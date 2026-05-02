import argparse
import csv
import os
import random
import time
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Argparse
def get_args():
    parser = argparse.ArgumentParser(description="VGG16-style FruitCNN Training")
    parser.add_argument("--train_dir",   type=str,   default="./imageset/train")
    parser.add_argument("--val_dir",     type=str,   default="./imageset/val")
    parser.add_argument("--test_dir",    type=str,   default="./imageset/test")
    parser.add_argument("--output_dir",  type=str,   default="./outputs/vgg16")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--save_model",  action="store_true")
    parser.add_argument("--model_path",  type=str,   default="best_model_vgg16.pth")
    parser.add_argument("--save_csv",    type=str,   default="results.csv")
    parser.add_argument("--results_dir", type=str,   default="./outputs")
    return parser.parse_args()


# Seed

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# VGG16-custom CNN
class VGG16Style(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()

        def conv_block(in_ch, out_ch, n_conv):
            layers = []
            for i in range(n_conv):
                layers += [
                    nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3,   64,  2),   # block1: 224→112
            conv_block(64,  128, 2),   # block2: 112→56
            conv_block(128, 256, 3),   # block3: 56→28
            conv_block(256, 512, 3),   # block4: 28→14
            conv_block(512, 512, 3),   # block5: 14→7
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# train/evaluation
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples, correct = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        correct       += (logits.argmax(1) == labels).sum().item()
    return total_loss / total_samples, correct / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_samples, correct = 0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        correct       += (logits.argmax(1) == labels).sum().item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    cm       = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss":             total_loss / total_samples,
        "accuracy":         correct / total_samples,
        "macro_f1":         macro_f1,
        "confusion_matrix": cm,
        "all_preds":        all_preds,
        "all_labels":       all_labels,
    }


# Dataset Exploration
def explore_dataset(dataset, class_names, output_dir):
    print("\n--- Dataset Exploration ---")
    targets = [s[1] for s in dataset.samples]
    counter = Counter(targets)
    counts  = [counter[i] for i in range(len(class_names))]

    print(f"{'Class':<40} {'Count':>6}")
    print("-" * 48)
    for i, name in enumerate(class_names):
        print(f"{name:<40} {counter[i]:>6}")

    imbalance_ratio = max(counts) / (min(counts) + 1e-9)
    print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio < 1.5:
        print("→ Dataset appears BALANCED.")
    elif imbalance_ratio < 3.0:
        print("→ Dataset is MILDLY IMBALANCED.")
    else:
        print("→ Dataset is SEVERELY IMBALANCED.")

    mean_count = np.mean(counts)
    colors = ["tomato" if c < mean_count * 0.85 or c > mean_count * 1.15 else "steelblue"
              for c in counts]

    plt.figure(figsize=(max(20, len(class_names) // 3), 6))
    plt.bar(class_names, counts, color=colors)
    plt.axhline(y=mean_count, color='gray', linestyle='--', label=f'Mean ({mean_count:.0f})')
    plt.xticks(rotation=90, ha='right', fontsize=6)
    plt.title("Class Distribution (red = deviates >15% from mean)")
    plt.ylabel("Number of Images")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150); plt.close()

    n_show  = min(30, len(class_names))
    n_cols  = 6
    n_rows  = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()
    for i in range(n_show):
        idx = targets.index(i)
        img, label = dataset[idx]
        img = img.numpy().transpose((1, 2, 0))
        img = np.clip(np.array([0.229, 0.224, 0.225]) * img
                      + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(class_names[label][:14], fontsize=7)
        axes[i].axis("off")
    for j in range(n_show, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "sample_images.png", dpi=150); plt.close()
    print("Saved exploration plots.\n")


# Learning Curve
def plot_learning_curve(train_losses, val_losses,
                        train_accs, val_accs,
                        train_f1s, val_f1s, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend()

    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs,   label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend()

    ax3.plot(train_f1s, label="Train Macro-F1")
    ax3.plot(val_f1s,   label="Val Macro-F1")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Macro-F1")
    ax3.set_title("Macro-F1"); ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


# Confusion Matrix

def plot_confusion_matrix(cm, title, save_path, classes):
    plt.figure(figsize=(20, 18))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title); plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, ha='right', fontsize=5)
    plt.yticks(tick_marks, classes, fontsize=5)
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


# Class-wise F1

def save_classwise_f1(all_labels, all_preds, class_names, save_path):
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in class_names:
            if name in report:
                r = report[name]
                writer.writerow([name,
                                  round(r["precision"], 4),
                                  round(r["recall"],    4),
                                  round(r["f1-score"],  4),
                                  int(r["support"])])
    print(f"Saved class-wise F1 → {save_path}")


# example of success and error
@torch.no_grad()
def save_prediction_samples(model, loader, class_names, device, output_dir, n_samples=10):
    model.eval()
    correct_saved, wrong_saved = [], []

    for images, labels in loader:
        if len(correct_saved) >= n_samples and len(wrong_saved) >= n_samples:
            break
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        for img, label, pred in zip(images.cpu(), labels.cpu(), preds.cpu()):
            if pred == label and len(correct_saved) < n_samples:
                correct_saved.append((img, label.item(), pred.item()))
            elif pred != label and len(wrong_saved) < n_samples:
                wrong_saved.append((img, label.item(), pred.item()))

    def denorm(t):
        t = t.numpy().transpose((1, 2, 0))
        return np.clip(np.array([0.229, 0.224, 0.225]) * t
                       + np.array([0.485, 0.456, 0.406]), 0, 1)

    for tag, samples in [("correct", correct_saved), ("wrong", wrong_saved)]:
        n = len(samples)
        if n == 0:
            continue
        fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
        if n == 1:
            axes = [axes]
        for ax, (img, label, pred) in zip(axes, samples):
            ax.imshow(denorm(img))
            title = f"GT: {class_names[label][:10]}\nPred: {class_names[pred][:10]}"
            ax.set_title(title, fontsize=7,
                         color="green" if tag == "correct" else "red")
            ax.axis("off")
        plt.suptitle(
            f"{'Correct' if tag == 'correct' else 'Wrong'} Predictions (VGG16-style)",
            fontsize=10
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"{tag}_predictions.png", dpi=150); plt.close()
    print("Saved prediction sample images.")


# Main
def main():
    args = get_args()
    set_seed(42)

    print("===== Arguments =====")
    for k, v in vars(args).items():
        print(f"  {k:<15}: {v}")
    print("=====================")

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    for d in [args.train_dir, args.val_dir, args.test_dir]:
        if not os.path.exists(d):
            print(f"Directory not found: {d}"); return

    #Transform size(224x224)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #Dataset load
    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(args.val_dir,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(args.test_dir,  transform=eval_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Classes   : {num_classes}")
    print(f"Train     : {len(train_dataset)}")
    print(f"Val       : {len(val_dataset)}")
    print(f"Test      : {len(test_dataset)}")

    explore_dataset(train_dataset, class_names, output_dir)
    
    # class_names.txt
    with open(results_dir / "class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    explore_dataset(train_dataset, class_names, output_dir)

    #DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    #model/optimzer
    model     = VGG16Style(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_params = count_parameters(model)
    print(f"\nModel: VGG16-style CNN (custom) | Parameters: {num_params:,}")
    print("Improvements over Baseline:")
    print("  - 5 conv blocks (vs 3) with BatchNorm")
    print("  - Deeper classifier with Dropout(0.5)")
    print("  - Input resolution 128 → 224")
    print("  - Scratch 학습 (from scratch)")

    run_id    = f"vgg16style_adam_lr{args.lr}_bs{args.batch_size}"
    curve_csv = output_dir / f"curve_{run_id}.csv"
    with open(curve_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "train_f1",
                                 "val_loss", "val_acc", "val_f1"])

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    train_f1s,    val_f1s    = [], []
    best_val_f1      = 0.0
    best_model_state = None
    total_train_time = 0.0

    print(f"\n--- Training ({run_id}) ---")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time

        train_metrics = evaluate(model, train_loader, criterion, device, num_classes)
        val_metrics   = evaluate(model, val_loader,   criterion, device, num_classes)

        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        train_accs.append(train_acc)
        val_accs.append(val_metrics["accuracy"])
        train_f1s.append(train_metrics["macro_f1"])
        val_f1s.append(val_metrics["macro_f1"])

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
              f"F1: {train_metrics['macro_f1']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val F1: {val_metrics['macro_f1']:.4f} | "
              f"Time: {epoch_time:.1f}s")

        with open(curve_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, train_acc, train_metrics["macro_f1"],
                val_metrics["loss"], val_metrics["accuracy"], val_metrics["macro_f1"]
            ])

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1      = val_metrics["macro_f1"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            if args.save_model:
                torch.save(best_model_state, output_dir / args.model_path)
            print(f"  → Best model saved (Val F1: {best_val_f1:.4f})")

    plot_learning_curve(train_losses, val_losses,
                        train_accs, val_accs,
                        train_f1s, val_f1s,
                        output_dir / f"learning_curve_{run_id}.png")

    #final evaluation
    model.load_state_dict(best_model_state)

    final_val    = evaluate(model, val_loader,  criterion, device, num_classes)
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)

    print(f"\n[Best Model - Val]  Acc: {final_val['accuracy']:.4f} | "
          f"Macro-F1: {final_val['macro_f1']:.4f}")
    print(f"[Final Test]        Acc: {test_metrics['accuracy']:.4f} | "
          f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Total Training Time: {total_train_time:.1f}s "
          f"({total_train_time / 60:.1f}min)")

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        title="VGG16-style CNN - Test Confusion Matrix",
        save_path=output_dir / f"confusion_matrix_{run_id}.png",
        classes=class_names
    )
    save_classwise_f1(test_metrics["all_labels"], test_metrics["all_preds"],
                      class_names, output_dir / f"classwise_f1_{run_id}.csv")
    save_prediction_samples(model, test_loader, class_names, device, output_dir)

    #CSV
    results_csv = results_dir / args.save_csv
    file_exists = results_csv.is_file()
    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "optimizer", "lr", "batch_size", "epochs",
                              "num_params", "total_train_time_s",
                              "best_val_f1", "val_acc", "val_f1",
                              "test_acc", "test_f1"])
        writer.writerow(["VGG16Style", "adam", args.lr, args.batch_size, args.epochs,
                         num_params, round(total_train_time, 1),
                         round(best_val_f1, 4),
                         round(final_val["accuracy"], 4),
                         round(final_val["macro_f1"], 4),
                         round(test_metrics["accuracy"], 4),
                         round(test_metrics["macro_f1"], 4)])

    print(f"\nDone! Results appended to {results_csv}")
    print(f"All outputs saved to {output_dir}")

if __name__ == "__main__":
    main()