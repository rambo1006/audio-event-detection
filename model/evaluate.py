import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from train import SimpleCNN
from data.dataset import SpeechCommandsDataset, KEYWORDS

DATA_PATH = "../data"
MODEL_PATH = "../results/best_model.pt"

def load_model():
    model = SimpleCNN(num_classes=len(KEYWORDS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def get_predictions(model, loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(KEYWORDS)))
    ax.set_yticks(range(len(KEYWORDS)))
    ax.set_xticklabels(KEYWORDS, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(KEYWORDS, fontsize=8)

    thresh = cm_normalized.max() / 2
    for i in range(len(KEYWORDS)):
        for j in range(len(KEYWORDS)):
            if cm_normalized[i, j] > 0.01:
                ax.text(j, i, f'{cm_normalized[i,j]:.0%}',
                       ha='center', va='center', fontsize=6,
                       color='white' if cm_normalized[i, j] > thresh else 'black')

    ax.set_title('Confusion Matrix — Audio Event Detection', fontsize=14, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig('../results/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("saved results/confusion_matrix.png")

def print_top_confusions(labels, preds, top_n=10):
    cm = confusion_matrix(labels, preds)
    np.fill_diagonal(cm, 0)

    confusions = []
    for i in range(len(KEYWORDS)):
        for j in range(len(KEYWORDS)):
            if cm[i, j] > 0:
                confusions.append((cm[i, j], KEYWORDS[i], KEYWORDS[j]))

    confusions.sort(reverse=True)

    print(f"\nTop {top_n} most confused pairs:")
    print(f"{'True':<12} {'Predicted':<12} {'Count'}")
    print("-" * 35)
    for count, true, pred in confusions[:top_n]:
        print(f"{true:<12} {pred:<12} {count}")

def run():
    print("loading test dataset...")
    test_set = SpeechCommandsDataset(DATA_PATH, split='test', augment=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    print("loading model...")
    model = load_model()

    print("running inference on test set...")
    labels, preds = get_predictions(model, test_loader)

    accuracy = (labels == preds).mean()
    print(f"\ntest accuracy: {accuracy*100:.2f}%")

    print_top_confusions(labels, preds)

    print("\ngenerating confusion matrix...")
    plot_confusion_matrix(labels, preds)

    print("\nper-class accuracy:")
    report = classification_report(labels, preds, target_names=KEYWORDS)
    print(report)

    with open('../results/classification_report.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(report)
    print("saved results/classification_report.txt")

if __name__ == "__main__":
    run()