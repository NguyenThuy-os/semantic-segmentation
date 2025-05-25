import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kagglehub
import csv

# Download dataset
path = kagglehub.dataset_download("huynhthethien/spectrogramsignal")
print("Path to dataset files:", path)

# Dataset
class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir)])
        self.transform = transform
        self.class_colors = {
            (2, 0, 0): 0,
            (127, 0, 0): 1,
            (248, 163, 191): 2
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_paths[idx])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, index in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = index

        if self.transform:
            augmented = self.transform(image=image, mask=label_mask)
            image = augmented['image']
            label_mask = augmented['mask']

        return image, label_mask.long()

# Data augmentation
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

# Load dataset
full_dataset = SemanticSegmentationDataset(
    image_dir=os.path.join(path, 'input'),
    label_dir=os.path.join(path, 'label'),
    transform=None
)

# Train/Val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model definition (unchanged)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x):
        w = self.global_pool(x)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

# Custom weight initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class LightSegNet(nn.Module):
    def __init__(self, n_classes):
        super(LightSegNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 26, 3, padding=1), nn.BatchNorm2d(26), nn.ReLU(),
            nn.Conv2d(26, 26, 3, padding=1), nn.BatchNorm2d(26), nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(26, 52, 3, padding=1), nn.BatchNorm2d(52), nn.ReLU(),
            nn.Conv2d(52, 52, 3, padding=1), nn.BatchNorm2d(52), nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(52, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            SEBlock(96, reduction=4),
            nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Dropout2d(0.3)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(96 + 52, 52, 3, padding=1), nn.BatchNorm2d(52), nn.ReLU(),
            nn.Conv2d(52, 52, 3, padding=1), nn.BatchNorm2d(52), nn.ReLU()
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(52 + 26, 26, 3, padding=1), nn.BatchNorm2d(26), nn.ReLU(),
            nn.Conv2d(26, 26, 3, padding=1), nn.BatchNorm2d(26), nn.ReLU()
        )

        self.classifier = nn.Conv2d(26, n_classes, 1)
        self.apply(init_weights)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec1 = self.up1(bottleneck)
        dec1 = self.decoder1(torch.cat([dec1, enc2], dim=1))
        dec2 = self.up2(dec1)
        dec2 = self.decoder2(torch.cat([dec2, enc1], dim=1))
        return self.classifier(dec2)

# Loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()

def combined_loss(logits, targets):
    ce = nn.CrossEntropyLoss()(logits, targets)
    dice = DiceLoss()(logits, targets)
    return ce + dice

# Training & evaluation loop
def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    acc = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        acc(preds, labels)
        iou(preds, labels)
        pbar.set_postfix({
            'Batch Loss': f'{loss.item():.4f}',
            'Mean Acc': f'{acc.compute():.4f}',
            'Mean IoU': f'{iou.compute():.4f}'
        })
    return running_loss / len(dataloader.dataset), acc.compute().cpu().numpy(), iou.compute().cpu().numpy()

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    acc = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            acc(preds, labels)
            iou(preds, labels)
            pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Mean Acc': f'{acc.compute():.4f}',
                'Mean IoU': f'{iou.compute():.4f}'
            })
    return running_loss / len(dataloader.dataset), acc.compute().cpu().numpy(), iou.compute().cpu().numpy()

def average_iou(iou_list):
    return np.mean([iou for _, iou in iou_list])

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightSegNet(n_classes=3).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
criterion = combined_loss

# CSV file setup for logging
csv_filename = "training_log.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_iou', 'val_iou'])

# Training loop
best_iou = 0
best_epoch = 0
best_state = None
iou_history = []

for epoch in range(10):
    print(f"\nEpoch {epoch + 1}")
    train_loss, train_acc, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, 3)
    val_loss, val_acc, val_iou = evaluate(model, val_loader, criterion, device, 3)
    iou_history.append((epoch + 1, val_iou))
    print(f"Epoch {epoch+1} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val IoU: {val_iou:.4f}")

    # Append epoch results to CSV
    with open(csv_filename, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([epoch+1, train_loss, val_loss, train_iou, val_iou])

    if val_iou > best_iou:
        best_iou = val_iou
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())

print(f"\nBest validation IoU: {best_iou:.4f} at epoch {best_epoch}")

# Save best model
torch.save(best_state, "best_model.pt")

# Plot IoU history  
epochs, val_ious = zip(*iou_history)
plt.plot(epochs, val_ious, marker='o')
plt.title('Validation IoU per Epoch')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.grid(True)
plt.show()
