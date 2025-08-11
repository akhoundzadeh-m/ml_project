import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
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

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_train_transform():
        return A.Compose([
            A.LongestMaxSize(max_size=CLSConfig.IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=CLSConfig.IMAGE_SIZE,
                min_width=CLSConfig.IMAGE_SIZE,
                border_mode=0
            ),
            A.ElasticTransform(alpha=40, sigma=5, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.ToGray(p=0.2),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_val_transform():
        return A.Compose([
            A.LongestMaxSize(max_size=CLSConfig.IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=CLSConfig.IMAGE_SIZE,
                min_width=CLSConfig.IMAGE_SIZE,
                border_mode=0
            ),
            ToTensorV2()
        ])
    

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
