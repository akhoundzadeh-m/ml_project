import json
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from math import ceil
from config import ClusterConfig

def load_character_crops(image_dir, label_dir, image_size):
    all_chars = []
    image_ids = [file.stem for file in image_dir.iterdir()]

    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.png"
        label_path = label_dir / f"{image_id}.json"

        if not label_path.exists() or not image_path.exists():
            continue

        with open(label_path, "r") as f:
            data = json.load(f)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        for annot in data["annotations"]:
            x, y, w, h = map(int, annot["boundingBox"].values())

            if w <= 0 or h <= 0 or x < 0 or y < 0:
                continue
            if y + h > image.shape[0] or x + w > image.shape[1]:
                continue

            cropped = image[y:y+h, x:x+w]
            if cropped.size == 0:
                continue
            
            cropped_resized = cv2.resize(cropped, image_size)
            all_chars.append(cropped_resized)

    return all_chars


def extract_features(images):
    features = []
    for image in images:
        hog_feat = hog(
            image, 
            pixels_per_cell=ClusterConfig.HOG_CELLS,
            cells_per_block=ClusterConfig.HOG_BLOCKS,
            orientations=ClusterConfig.HOG_ORIENTATIONS,
            block_norm='L2-Hys'
        )

        lbp = local_binary_pattern(
            image, 
            P=ClusterConfig.LBP_POINTS, 
            R=ClusterConfig.LBP_RADIUS, 
            method='uniform'
        )
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, ClusterConfig.LBP_POINTS + 3), density=True)

        moments = cv2.moments(image)
        hu = cv2.HuMoments(moments).flatten()
        hu_features = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        combined = np.concatenate([hog_feat, hist, hu_features])
        features.append(combined)
    
    return np.array(features)


def run_clustering(X, method):
    if method == "kmeans":
        model = KMeans(n_clusters=ClusterConfig.NUM_CLUSTERS, random_state=0)
    elif method == "dbscan":
        model = DBSCAN(
            eps=ClusterConfig.DBSCAN_EPS, 
            min_samples=ClusterConfig.DBSCAN_MIN_SAMPLES
        )
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=ClusterConfig.NUM_CLUSTERS)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
        
    labels = model.fit_predict(X)
    return labels, model


def evaluate_clustering(X, labels):
    mask = labels != -1
    if len(set(labels[mask])) <= 1:
        return -1, -1
        
    sil = silhouette_score(X[mask], labels[mask])
    cal = calinski_harabasz_score(X[mask], labels[mask])
    return sil, cal


def visualize_clusters(X, images, labels, method_name, save_path):
    unique_labels = sorted([l for l in set(labels) if l != -1])
    n_clusters = len(unique_labels)
    cols = 8
    rows = ceil(n_clusters / cols)
    
    fig, axs = plt.subplots(rows * 2, cols, figsize=ClusterConfig.VISUALIZATION_SIZE)
    axs = np.array(axs).reshape(rows * 2, cols)
    
    for i, k in enumerate(unique_labels):
        idxs = np.where(labels == k)[0]
        if len(idxs) < 2:
            continue

        cluster_feats = X[idxs]
        center = np.mean(cluster_feats, axis=0)
        dists = cdist([center], cluster_feats)[0]
        nearest = idxs[np.argmin(dists)]
        farthest = idxs[np.argmax(dists)]
        
        row = (i // cols) * 2
        col = i % cols
        
        axs[row, col].imshow(images[nearest], cmap='gray')
        axs[row, col].set_title(f"C{k}", fontsize=7)
        axs[row, col].axis('off')
        
        axs[row + 1, col].imshow(images[farthest], cmap='gray')
        axs[row + 1, col].axis('off')
    
    for ax in axs.flat:
        if not hasattr(ax, 'images') or not ax.images:
            ax.axis('off')
    
    fig.suptitle(f"{method_name} - Nearest (top) / Farthest (bottom)", fontsize=12)
    plt.tight_layout(h_pad=0.8)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)