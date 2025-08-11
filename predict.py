import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import config

def detect_objects(model, test_image_dir, transform, save_path, device, thresh=0.5):
    model.to(device)
    model.eval()
    image_ids = [file.stem for file in test_image_dir.iterdir()]

    with open(save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "x", "y", "width", "height"])

        for image_id in tqdm(image_ids, desc="Detecting objects", leave=False):
            image_path = test_image_dir / f"{image_id}.png"
            image = Image.open(image_path).convert("L")

            transformed = transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(device)

            with torch.no_grad():
                predictions = model([image_tensor])[0]
            
            boxes = predictions["boxes"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()
            keep = scores > thresh

            if not np.any(keep):
                writer.writerow([image_id])
            else:
                for box in boxes[keep]:
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min

                    writer.writerow([image_id, x_min, y_min, width, height])


def predict_expressions(model, test_image_dir, detection_csv_path, transform, save_path, device):
    model.to(device)
    model.eval()

    detections = defaultdict(list)
    with open(detection_csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if not row: # Skip empty rows
                continue
            image_id = row[0]
            if len(row) == 1: # No detections
                detections[image_id] = []
            else:
                x, y, w, h = map(float, row[1:5])
                detections[image_id].append((x, y, w, h))
    
    with open(save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "expression"])

        for image_id, boxes in tqdm(detections.items(), desc="Predicting expressions"):
            image_path = test_image_dir / f"{image_id}.png"

            if not boxes:
                writer.writerow([image_id, ''])
                continue

            image = Image.opne(image_path).convert("L")
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                writer.writerow([image_id, ""])
                continue

            characters = []
            for box in sorted(boxes, key=lambda b: b[0]): # Sort by x-coordinate
                x, y, w, h = box

                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(image.shape[1], int(x + w))
                y2 = min(image.shape[0], int(y + h))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                char_img = image[y1:y2, x1:x2]

                try:
                    transformed = transform(image=char_img)
                    char_tensor = transformed['image'].unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(char_tensor)
                        pred_idx = torch.argmax(output, dim=1).item()

                    char = config.IDX_TO_CLASS.get(pred_idx, '?') if pred_idx > 0 else '?'
                    characters.append(char)
                except Exception as e:
                    print(f"Error processing box {box} in image {image_id}: {str(e)}")
                    characters.append('?')
            
            expression = "".join(characters)
            writer.writerow([image_id, expression])