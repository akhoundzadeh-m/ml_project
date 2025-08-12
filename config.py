import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random

class ODTConfig:
    # Reproducibility
    SEED = 42

    # Data
    IMAGE_SIZE = 512
    BATCH_SIZE = 1

    # Training
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005

    # Model
    ANCHOR_SIZES = ((32, 64, 128, 256),)
    ANCHOR_RATIOS = ((0.5, 1.0, 2.0),)
    NUM_CLASSES = 2

    @staticmethod
    def get_train_transform():
        return A.Compose([
            A.LongestMaxSize(max_size=512),
            A.PadIfNeeded(min_height=512, min_width=512, 
                        border_mode=0),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        clip=True,
        filter_invalid_bboxes=True
    ))

    @staticmethod
    def get_val_transform():
        return A.Compose([
            A.LongestMaxSize(max_size=512),
    A.PadIfNeeded(min_height=512, min_width=512, 
                  border_mode=0),
    ToTensorV2()
        ],
        bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        clip=True,
        filter_invalid_bboxes=True
    ))

    @staticmethod
    def get_inference_transform():
        return transforms.Compose([
            transforms.ToTensor()
        ])


CHAR_CLASSES = ['0','1','2','3','4','5','6','7','8','9','-','+','/','*','(',')']
CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(CHAR_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
UNKNOWN_LABEL = 0

class CLSConfig:
    # Reproducibility
    SEED = 42

    # Data
    IMAGE_SIZE = 48
    BATCH_SIZE = 16

    # Training
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4
    PSEUDO_START_EPOCH = 5
    MAX_PSEUDO_WEIGHT = 0.8
    CONF_THRESH = 0.95

    # Model
    NUM_CLASSES = len(CHAR_CLASSES) + 1 # +1 for unknown

    @staticmethod
    def get_train_transform():
        return transforms.Compose([
            ResizeAndPad(max_size=48, target_size=(48, 48)),
            AlbumentationsTransform(A.ElasticTransform(alpha=40, sigma=5, p=0.5)),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomInvert(p=0.1),
            transforms.ToTensor()
        ])
    
    @staticmethod
    def get_val_transform():
        return transforms.Compose([
            ResizeAndPad(max_size=48, target_size=(48, 48)),
            transforms.ToTensor()
        ])


class ResizeAndPad:
    def __init__(self, max_size, target_size, fill=0):
        self.max_size = max_size
        self.target_size = target_size
        self.fill = fill
    
    def __call__(self, img):
        w, h = img.size
        scale = self.max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, (new_h, new_w))

        pad_h = self.target_size[0] - new_h
        pad_w = self.target_size[1] - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img = TF.pad(img, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        return img
    
class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        # PIL -> NumPy
        img_np = np.array(img)
        # Apply Albumentations
        augmented = self.aug(image=img_np)
        img_np = augmented['image']
        # Back to PIL
        return Image.fromarray(img_np)
    

class ClusterConfig:
    # Processing parameters
    IMAGE_SIZE = 32
    NUM_CLUSTERS = 16 # 0-9, operators, brackets

    # Feature extraction
    LBP_RADIUS = 3
    LBP_POINTS = 24  # 8 * radius
    HOG_CELLS = (8, 8)
    HOG_BLOCKS = (2, 2)
    HOG_ORIENTATIONS = 9

    # Clustering methods
    METHODS = ["kmeans", "dbscan", "agglomerative"]
    DBSCAN_EPS = 2.5
    DBSCAN_MIN_SAMPLES = 4

    # Output
    VISUALIZATION_SIZE = (12, 8)


def set_seed(seed=42):
    """
    Set seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
