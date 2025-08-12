import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

class ODTDataset(Dataset):
    """
    Custom dataset for object detection task.

    Args:
        image_dir (Path): Directory containing images
        label_dir (Path): Directory containing JSON annotations
        transform (A.Compose): Albumentations transform pipeline
    """

    def __init__(self, image_dir: Path, label_dir: Path, transform :A.Compose=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_ids = [file.stem for file in image_dir.iterdir()]
        self.transform = transform
        self.default_transform = A.Compose([ToTensorV2()])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = self._load_image(image_id)
        target = self._load_annotations(image_id)

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=target["boxes"],
                labels=target["labels"]
            )
            image = transformed["image"]
            target["boxes"] = transformed["bboxes"]
            target["labels"] = transformed["labels"]
        else:
            image = self.default_transform(image=image)["image"]
        
        return self._postprocess(image, target, image_id)
    
    def _load_image(self, image_id):
        return np.array(Image.open(self.image_dir / f"{image_id}.png").convert("L"))
    
    def _load_annotations(self, image_id):
        with open(self.label_dir / f"{image_id}.json", "r") as f:
            data = json.load(f)
        
        boxes, labels = [], []
        for annot in data["annotations"]:
            x, y, w, h = annot["boundingBox"].values()
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
        
        return {"boxes": boxes, "labels": labels}
    
    def _postprocess(self, image, target, image_id):
        image = image.float() / 255.0 if image.dtype == torch.uint8 else image.float()

        boxes = torch.tensor(target["boxes"], dtype=torch.float32)
        labels = torch.tensor(target["labels"], dtype=torch.int64)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return image, {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([int(image_id)]),
            "area": areas,
            "iscrowd": torch.zeros_like(labels)
        }


def get_dataloaders(train_dir: Path, val_dir: Path,
                    train_transform: A.Compose=None, val_transform: A.Compose=None,
                    batch_size: int=1):
    """
    Create train and validation dataloaders.

    Args:
        train_dir (Path): Directory containing train dataset
        val_dir (Path): Directory containing validation dataset
        train_transform (A.Compose): Albumentations transform pipeline for train dataset
        val_transform (A.Compose): Albumentations transform pipeline for validation dataset
        batch_size (int): Number of samples in each batch

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders
    """

    train_ds = ODTDataset(
        image_dir=train_dir / "images",
        label_dir=train_dir / "labels",
        transform=train_transform
    )

    val_ds = ODTDataset(
        image_dir=val_dir / "images",
        label_dir=val_dir / "labels",
        transform=val_transform
    )

    train_dataloader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader


def collate_fn(batch):
    """
    Custom collate function for variable-sized bounding boxes.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


class CLSDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_ids = [file.stem for file in image_dir.iterdir()]
        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.image_dir / f"{sample['image_id']}.png"
        image = Image.open(image_path).convert("L")

        x_min, y_min, x_max, y_max = map(int, sample["box"])
        cropped = image.crop((x_min, y_min, x_max, y_max))
        return cropped, sample["label"]
    
    def _build_samples(self):
        samples = []
        for image_id in self.image_ids:
            label_path = self.label_dir / f"{image_id}.json"
            if not label_path.exists():
                continue

            with open(label_path, "r") as f:
                data = json.load(f)
            
            boxes = []
            for annot in data["annotations"]:
                x, y, w, h = annot["boundingBox"].values()
                boxes.append([x, y, x + w, y + h])
            
            expression = list(data.get("expression", ""))
            order = np.argsort([box[0] for box in boxes])
            boxes = [boxes[i] for i in order]

            for i, box in enumerate(boxes):
                label = config.CLASS_TO_IDX.get(
                    expression[i], config.UNKNOWN_LABEL
                ) if i < len(expression) else config.UNKNOWN_LABEL

                samples.append({
                    "image_id": image_id,
                    "box": box,
                    "label": label
                })
        
        return samples
            

class SupervisedCLSDataset(CLSDataset):
    def __init__(self, image_dir, label_dir, transform=None):
        super().__init__(image_dir, label_dir)
        self.transform = transform
        self.samples = [s for s in self.samples if s['label'] != config.UNKNOWN_LABEL]
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        if self.transform:
            image = self.transform(image)
        
        return image, label


class UnlabeledCLSDataset(CLSDataset):
    def __init__(self, image_dir, label_dir, transform=None):
        super().__init__(image_dir, label_dir)
        self.transform = transform
    
    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        if self.transform:
            image = self.transform(image)
        
        return image, self.samples[idx]["label"]