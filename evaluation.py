import json
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import random
import math
import Levenshtein
from sklearn.metrics import classification_report
import config

def plot_detection_predictions(model, dataset, device, save_path, num_samples=4, thresh=0.5):
    model.eval()
    model.to(device)
    indices = random.sample(range(len(dataset)), num_samples)
    cols = min(2, num_samples)
    rows = math.ceil(num_samples / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, ax in zip(indices, axes.flat):
        image, target = dataset[idx]

        with torch.no_grad():
            prediction = model([image.to(device)])[0]
        
        ax.imshow(image.squeeze().cpu().numpy(), cmap="gray")

        for box in target["boxes"]:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min),
                width=x_max - x_min,
                height=y_max - y_min,
                linewidth=2,
                edgecolor="g",
                facecolor="none"
            )
            ax.add_patch(rect)
        
        for box, score in zip(prediction["boxes"], prediction["scores"]):
            if score > thresh:
                x_min, y_min, x_max, y_max = box.cpu().numpy()
                rect = patches.Rectangle(
                    (x_min, y_min),
                    width=x_max - x_min,
                    height=y_max - y_min,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(x_min, y_min, f"{score:.2f}", color="red")
            
        ax.set_title(f"ID: {target['image_id'].item()} — Ground Truth: Green, Pred: Red")
        ax.axis('off')
    
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_map(model, dataloader, device, iou_type="bbox"):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type=iou_type)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="mAP Calculation"):
            images = [image.to(device) for image in images]
            predictions = model(images)

            metric_targets = [{
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device)
            } for target in targets]

            metric.update(predictions, metric_targets)
    
    results = metric.compute()

    return {
        "map": results["map"].item(),
        "map_50": results["map_50"].item(),
        "map_75": results["map_75"].item()
    }


def plot_cls_predictions(model, dataset, device, save_path, num_samples=6):
    model.eval()
    model.to(device)

    image_ids = list(set([s["image_id"] for s in dataset.samples]))
    selected_ids = random.sample(image_ids, num_samples)

    cols = 2
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).flatten()

    for ax, image_id in zip(axes, selected_ids):
        image_path = dataset.image_dir / f"{image_id}.png"
        label_path = dataset.label_dir / f"{image_id}.json"

        if not image_path.exists() or not label_path.exists():
            ax.axis("off")
            continue

        image = Image.open(image_path).convert("L")
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Image: {image_id}", fontsize=12)

        with open(label_path, "r") as f:
            data = json.load(f)
        
        expression = []
        for annot in data["annotations"]:
            x, y, w, h = annot["boundingBox"].values()

            char_img = image.crop((x, y, x + w, y + h))
            char_tensor = dataset.transform(char_img)
            char_tensor = char_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(char_tensor)
                char_idx = torch.argmax(pred, dim=1).item()
                char = config.IDX_TO_CLASS.get(char_idx, "?")

            expression.append((x, char))

            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x, y - 5, char, color="blue", fontsize=10, weight="bold")
        
        pred_expr = "".join([c for _, c in sorted(expression, key=lambda x: x[0])])
        gt_expr = data.get("expression", "")
        ax.set_xlabel(f"Predicted: {pred_expr}\nGround Truth: {gt_expr}", fontsize=10)
        ax.axis("off")
    
    for ax in axes[len(selected_ids):]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_cls_model(model, dataset, device):
    model.eval()
    model.to(device)
    all_preds, all_labels, lev_distances = [], [], []
    
    image_groups = {}
    for sample in dataset.samples:
        image_id = sample["image_id"]
        image_groups.setdefault(image_id, []).append(sample)
    
    for image_id, samples in image_groups.items():
        samples = sorted(samples, key=lambda s: s["box"][0])
        gt_expr = ""
        pred_expr = ""

        for sample in samples:
            if sample["label"] == config.UNKNOWN_LABEL:
                continue

            image_path = dataset.image_dir / f"{image_id}.png"
            image = Image.open(image_path).convert("L")
            x_min, y_min, x_max, y_max = map(int, sample["box"])
            char_img = image.crop((x_min, y_min, x_max, y_max))
            char_tensor = dataset.transform(char_img)
            char_tensor = char_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(char_tensor)
                pred_idx = torch.argmax(logits, dim=1).item()
            
            gt_char = config.IDX_TO_CLASS.get(sample["label"], "?")
            pred_char = config.IDX_TO_CLASS.get(pred_idx, "?")
            gt_expr += gt_char
            pred_expr += pred_char

            all_labels.append(sample["label"])
            all_preds.append(pred_idx)
        
        lev_distances.append(Levenshtein.distance(gt_expr, pred_expr))
    
    class_names = ["Unknown"] + config.CHAR_CLASSES
    labels = list(range(len(class_names)))
    report = classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )

    print("Character Classification Report:")
    print(report)

    avg_lev = np.mean(lev_distances)
    print(f" - Average Levenshtein Distance: {avg_lev:.2f}")