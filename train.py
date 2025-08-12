import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm
import config
from config import CLSConfig

def train_step_faster_rcnn(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    
    return total_loss / len(dataloader)


def val_step_faster_rcnn(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation", leave=False):
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            model.train()
            loss_dict = model(images, targets)
            model.eval()
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    return total_loss / len(dataloader)


def train_faster_rcnn(model, train_dataloader, val_dataloader,
                      optimizer, scheduler, num_epochs, device,
                      model_save_path=None, loss_save_path=None):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_state = None
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_step_faster_rcnn(model, optimizer, train_dataloader, device)
        val_loss = val_step_faster_rcnn(model, val_dataloader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        print(f"Epoch [{epoch+1}/{num_epochs}] --> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if model_save_path:
        torch.save(best_model_state, model_save_path)
    
    if loss_save_path:
        save_losses(train_losses, val_losses, loss_save_path)


def get_pseudo_label_weight(epoch, num_epochs, max_weight=0.8):
    return min(max_weight, max_weight * (epoch / (num_epochs // 4)))


def generate_pseudo_labels(model, unlabeled_dataset, epoch, num_epochs, device):
    candidates = []
    model.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(unlabeled_dataset)), desc="Generating pseudo-labels", leave=False):
            image, _ = unlabeled_dataset[idx]
            image = image.unsqueeze(0).to(device)
            logits = model(image)
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            if conf.item() > CLSConfig.CONF_THRESH and pred.item() != config.UNKNOWN_LABEL:
                candidates.append((conf.item(), pred.item(), idx))
    
    candidates.sort(key=lambda x:x[0], reverse=True)
    max_candidates = len(candidates)
    frac = min(1.0, (epoch - CLSConfig.PSEUDO_START_EPOCH) / int(num_epochs - (num_epochs * 0.7)))
    k = int(max_candidates * frac)
    selected = candidates[:k]

    pseudo_labels = [label for _, label, _ in selected]
    pseudo_indices = [idx for _, _, idx in selected]

    print(f"[Epoch {epoch+1}] Pseudo-labels added: {len(pseudo_labels)} "
                  f"(from {max_candidates} candidates)")
    
    for i in range(len(pseudo_indices)):
        idx_in_unlabeled = pseudo_indices[i]
        unlabeled_dataset.samples[idx_in_unlabeled]['label'] = pseudo_labels[i] + 1000
    
    pseudo_subset = Subset(unlabeled_dataset, pseudo_indices)

    return pseudo_subset


def train_step_cls_model(model, dataloader, criterion, optimizer, epoch, num_epochs, device):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        sup_mask = labels < 1000
        pseudo_mask = ~sup_mask

        optimizer.zero_grad()
        outputs = model(images)

        loss_sup = criterion(outputs[sup_mask], labels[sup_mask]) if sup_mask.any() else 0
        if pseudo_mask.any():
            pseudo_labels = labels[pseudo_mask] - 1000
            loss_pseudo = criterion(outputs[pseudo_mask], pseudo_labels)
            weight = get_pseudo_label_weight(epoch, num_epochs, CLSConfig.MAX_PSEUDO_WEIGHT)
            loss = loss_sup + weight * loss_pseudo
        else:
            loss = loss_sup
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def val_step_cls_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total_loss += criterion(outputs, labels).item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total

    return total_loss / len(dataloader), accuracy



def train_cls_model(model, sup_dataset, unlabeled_dataset, val_dataloader,
                    criterion, optimizer, scheduler, batch_size, num_epochs, device,
                    model_save_path=None, loss_save_path=None):
    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float("inf")
    best_model_state = None
    model.to(device)

    for epoch in range(num_epochs):
        if epoch >= CLSConfig.PSEUDO_START_EPOCH:
            pseudo_dataset = generate_pseudo_labels(model, unlabeled_dataset, epoch, num_epochs, device)
            combined_dataset = ConcatDataset([sup_dataset, pseudo_dataset])
        else:
            combined_dataset =  sup_dataset

        train_dataloader = DataLoader(combined_dataset, batch_size, shuffle=True)
        train_loss = train_step_cls_model(model, train_dataloader, criterion, optimizer,
                                          epoch, num_epochs, device)
        val_loss, val_acc = val_step_cls_model(model, val_dataloader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        print(f"Epoch {epoch+1}/{num_epochs} --> "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        scheduler.step()

        if model_save_path:
            torch.save(best_model_state, model_save_path)
    
        if loss_save_path:
            save_losses(train_losses, val_losses, loss_save_path)


def save_losses(train_losses, val_losses, path):
    with open(path, "w") as f:
        f.write("Epoch, Train Loss, Val Loss\n")
        for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{epoch+1}, {t_loss:.6f}, {v_loss:.6f}\n")