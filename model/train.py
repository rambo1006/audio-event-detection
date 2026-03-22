import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SpeechCommandsDataset, KEYWORDS

DATA_PATH = "../data"
BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
DEVICE = torch.device("cpu")
NUM_CLASSES = len(KEYWORDS)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="training", leave=False):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="evaluating", leave=False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def train():
    print(f"device: {DEVICE}")
    print("loading datasets...")

    train_set = SpeechCommandsDataset(DATA_PATH, split='train', augment=False)
    val_set   = SpeechCommandsDataset(DATA_PATH, split='val',   augment=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    torch.manual_seed(42)
    model     = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {total_params:,}")

    best_val_acc = 0
    results = []

    print(f"\n{'epoch':<8} {'train loss':<12} {'train acc':<12} {'val loss':<12} {'val acc':<12}")
    print("-" * 56)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc     = evaluate(model, val_loader, criterion)
        scheduler.step(val_acc)

        results.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss,     'val_acc': val_acc
        })

        print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.3f} {val_loss:<12.4f} {val_acc:<12.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../results/best_model_v2.pt')
            print(f"         saved best model — val acc: {val_acc:.3f}")

    print(f"\ntraining complete!")
    print(f"best val accuracy: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    save_results(results)


def save_results(results):
    import matplotlib.pyplot as plt
    epochs     = [r['epoch']      for r in results]
    train_acc  = [r['train_acc']  for r in results]
    val_acc    = [r['val_acc']    for r in results]
    train_loss = [r['train_loss'] for r in results]
    val_loss   = [r['val_loss']   for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label='train loss')
    axes[0].plot(epochs, val_loss,   label='val loss')
    axes[0].set_title('Loss curve')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label='train accuracy')
    axes[1].plot(epochs, val_acc,   label='val accuracy')
    axes[1].set_title('Accuracy curve')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('../results/training_curves_v2.png', dpi=150)
    print("saved results/training_curves_v2.png")


if __name__ == "__main__":
    train()