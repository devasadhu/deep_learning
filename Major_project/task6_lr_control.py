"""
=============================================================================
TASK 6: INTELLIGENT LEARNING RATE CONTROL
=============================================================================
WCE Gastrointestinal Disease Classification Project

STRATEGIES IMPLEMENTED:
  1. ReduceLROnPlateau  — reduce LR when val loss stops improving
  2. CosineAnnealingLR  — smooth cosine decay of LR each epoch
  3. OneCycleLR         — warm-up + cosine annealing (fast convergence)

REQUIREMENTS:
  pip install torch torchvision matplotlib
=============================================================================
"""

import os
import json
import math
import time
from copy import deepcopy

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import (
        ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
    )
    import torchvision.transforms as transforms
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing dependency — {e}")
    print("  Run: pip install torch torchvision matplotlib")
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = r"C:\Users\devas\Desktop\DL LAB\Major_project"
SPLITS_DIR  = os.path.join(BASE_DIR, "task4_output", "splits", "kvasir_capsule")
OUTPUT_DIR  = os.path.join(BASE_DIR, "task6_output")

NUM_CLASSES   = 14
BATCH_SIZE    = 32
NUM_EPOCHS    = 25
INITIAL_LR    = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT_RATE  = 0.4
FREEZE_RATIO  = 0.7

# Windows requires num_workers=0 inside scripts (no __main__ fork guard).
# Set to 0 for safety; change to 4 only if running on Linux/Mac.
NUM_WORKERS = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR CODES
# ─────────────────────────────────────────────────────────────────────────────

class C:
    RED     = '\033[91m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN    = '\033[96m'
    WHITE   = '\033[97m'
    BOLD    = '\033[1m'
    RESET   = '\033[0m'

def cprint(text, colour='\033[97m'):
    print(f"{colour}{text}\033[0m")


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

def print_theory():
    cprint("\n" + "─"*80, C.BLUE)
    cprint("  INTELLIGENT LEARNING RATE CONTROL — KEY CONCEPTS", C.BOLD + C.CYAN)
    cprint("─"*80, C.BLUE)

    points = [
        ("Why LR Control Matters",
         "A constant LR is rarely optimal.\n"
         "    Too high → loss oscillates, never converges.\n"
         "    Too low  → extremely slow convergence, gets stuck in local minima.\n"
         "    Adaptive schedules automatically find the sweet spot."),

        ("Strategy 1: ReduceLROnPlateau",
         "Monitors validation loss after each epoch.\n"
         "    If val_loss doesn't improve for patience=3 epochs → LR × 0.5.\n"
         "    Most conservative, safest for medical datasets.\n"
         "    Best when: training is noisy or dataset is small."),

        ("Strategy 2: CosineAnnealingLR",
         "LR follows a cosine curve from LR_max → LR_min over T_max epochs.\n"
         "    Smooth decay — no sudden drops.\n"
         "    LR(t) = LR_min + 0.5*(LR_max−LR_min)*(1 + cos(π*t/T_max))\n"
         "    Best when: large datasets, stable training."),

        ("Strategy 3: OneCycleLR (Recommended)",
         "Warm-up: LR rises from 0 → LR_max over first 30% of steps.\n"
         "    Cool-down: LR falls back to 0 using cosine annealing.\n"
         "    Helps escape sharp minima, finds flatter, more generalisable solutions.\n"
         "    Best when: you want fast convergence (often 50% fewer epochs needed)."),
    ]
    for title, desc in points:
        cprint(f"\n  ▶ {title}:", C.YELLOW + C.BOLD)
        print(f"    {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODEL  (EfficientNet-B3, same as Task 5)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes, dropout=0.4, freeze_ratio=0.7):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    all_params = list(model.parameters())
    freeze_up_to = int(len(all_params) * freeze_ratio)
    for i, p in enumerate(all_params):
        p.requires_grad = (i >= freeze_up_to)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — JSON MANIFEST LOADER
# ─────────────────────────────────────────────────────────────────────────────

class ManifestDataset(torch.utils.data.Dataset):
    """
    Loads images listed in a Task-4 JSON manifest.
    Each entry: {"path": "...", "label": 0, "class": "..."}
    """
    def __init__(self, json_path, transform=None):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.transform = transform
        # Validate a few paths up front so errors surface early
        missing = [e['path'] for e in self.entries[:5]
                   if not os.path.exists(e['path'])]
        if missing:
            raise FileNotFoundError(
                f"Image paths in manifest don't exist on disk.\n"
                f"  First missing: {missing[0]}\n"
                f"  Check that task4_output/resized/ is intact and paths in\n"
                f"  the JSON match this machine's drive letter / folder layout."
            )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        from PIL import Image
        entry = self.entries[idx]
        img   = Image.open(entry['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, entry['label']


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def get_class_weights(json_path, num_classes, device):
    """Inverse-frequency class weights to handle imbalance."""
    with open(json_path) as f:
        entries = json.load(f)
    counts = [0] * num_classes
    for e in entries:
        counts[e['label']] += 1
    counts = [max(c, 1) for c in counts]
    total   = sum(counts)
    weights = [total / (num_classes * c) for c in counts]
    cprint(f"  Class weights (min={min(weights):.2f}, max={max(weights):.2f})", C.WHITE)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def make_loader(json_path, transform, shuffle, batch_size):
    ds = ManifestDataset(json_path, transform)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,   # 0 = safe on Windows
        pin_memory=torch.cuda.is_available(),
    ), len(ds)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scheduler_name, device):
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # OneCycleLR must step every BATCH
        if scheduler_name == "OneCycleLR":
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs    = model(images)
            loss       = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


def train_with_scheduler(scheduler_name, initial_state,
                         train_loader, val_loader, n_train,
                         num_epochs, initial_lr, weight_decay,
                         criterion, device):
    """Fresh model from initial_state, trained with the chosen scheduler."""

    model = build_model(NUM_CLASSES, DROPOUT_RATE, FREEZE_RATIO).to(device)
    model.load_state_dict(initial_state)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr, weight_decay=weight_decay
    )

    steps_per_epoch = len(train_loader)

    if scheduler_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
    elif scheduler_name == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer, max_lr=initial_lr,
            epochs=num_epochs, steps_per_epoch=steps_per_epoch
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    history = {"lr": [], "train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scheduler_name, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Record LR BEFORE epoch-level schedulers step
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)
        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 5))

        # Epoch-level schedulers step here
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        elif scheduler_name == "CosineAnnealingLR":
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        elapsed = time.time() - t0
        cprint(
            f"  Epoch {epoch:>3}/{num_epochs}  "
            f"lr={current_lr:.2e}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.1f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%  "
            f"({elapsed:.0f}s)",
            C.GREEN if val_loss == best_val_loss else C.WHITE
        )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(histories, num_epochs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, num_epochs + 1))

    colours = {
        "ReduceLROnPlateau": "#e74c3c",
        "CosineAnnealingLR": "#2ecc71",
        "OneCycleLR":        "#3498db",
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle("Task 6: Intelligent Learning Rate Control",
                 color='white', fontsize=16, fontweight='bold')

    for ax in axes.flat:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#444')
        ax.xaxis.grid(True, alpha=0.3, color='gray')
        ax.yaxis.grid(True, alpha=0.3, color='gray')

    # LR vs Epoch
    ax = axes[0, 0]
    for name, h in histories.items():
        ax.plot(epochs, h["lr"], label=name, color=colours[name], linewidth=2.5)
    ax.set_xlabel("Epoch", color='white')
    ax.set_ylabel("Learning Rate", color='white')
    ax.set_title("Learning Rate Schedule", color='white', fontsize=12)
    ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
    ax.set_yscale('log')

    # Training Loss
    ax = axes[0, 1]
    for name, h in histories.items():
        ax.plot(epochs, h["train_loss"], label=name, color=colours[name], linewidth=2.5)
    ax.set_xlabel("Epoch", color='white')
    ax.set_ylabel("Training Loss", color='white')
    ax.set_title("Training Loss Curves", color='white', fontsize=12)
    ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')

    # Validation Loss
    ax = axes[1, 0]
    for name, h in histories.items():
        ax.plot(epochs, h["val_loss"], label=name, color=colours[name],
                linewidth=2.5, linestyle='--')
    ax.set_xlabel("Epoch", color='white')
    ax.set_ylabel("Validation Loss", color='white')
    ax.set_title("Validation Loss Curves", color='white', fontsize=12)
    ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')

    # Validation Accuracy
    ax = axes[1, 1]
    for name, h in histories.items():
        ax.plot(epochs, [a * 100 for a in h["val_acc"]],
                label=name, color=colours[name], linewidth=2.5)
    ax.set_xlabel("Epoch", color='white')
    ax.set_ylabel("Validation Accuracy (%)", color='white')
    ax.set_title("Validation Accuracy", color='white', fontsize=12)
    ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    path = os.path.join(output_dir, "plot10_lr_control.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    cprint(f"  🖼  Saved: {path}", C.GREEN)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_report(histories, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    for name, h in histories.items():
        best_epoch = int(np.argmin(h["val_loss"])) + 1
        summary[name] = {
            "best_val_loss": round(min(h["val_loss"]), 4),
            "best_val_acc":  round(max(h["val_acc"]) * 100, 2),
            "best_epoch":    best_epoch,
            "final_lr":      h["lr"][-1],
            "history":       h,
        }

    best_scheduler = min(summary, key=lambda n: summary[n]["best_val_loss"])
    report = {
        "task":           "Task 6 - Learning Rate Control",
        "num_epochs":     NUM_EPOCHS,
        "initial_lr":     INITIAL_LR,
        "best_scheduler": best_scheduler,
        "schedulers":     summary,
    }
    path = os.path.join(output_dir, "task6_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    cprint(f"  📄 Report saved: {path}", C.GREEN)
    return summary, best_scheduler


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  (must be inside if __name__ == '__main__' on Windows)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cprint("\n" + "="*80, C.CYAN)
    cprint("  TASK 6: INTELLIGENT LEARNING RATE CONTROL", C.BOLD)
    cprint("  SVNIT Surat | Prof. Praveen Kumar Chandaliya", C.WHITE)
    cprint("="*80, C.CYAN)

    print_theory()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cprint(f"\n  Device  : {device}", C.CYAN)
    if device.type == "cuda":
        cprint(f"  GPU     : {torch.cuda.get_device_name(0)}", C.GREEN)
    else:
        cprint("  ⚠  No GPU detected. Training on CPU will be slow.", C.YELLOW)
        cprint(f"     Estimated time per epoch: ~3-8 min depending on dataset size.", C.YELLOW)

    # ── Verify split manifests exist ──────────────────────
    train_json = os.path.join(SPLITS_DIR, "train.json")
    val_json   = os.path.join(SPLITS_DIR,   "val.json")

    for path in [train_json, val_json]:
        if not os.path.exists(path):
            cprint(f"\n  ERROR: Split manifest not found: {path}", C.RED)
            cprint("  Make sure Task 4 ran successfully.", C.YELLOW)
            return

    # ── Load dataset sizes ─────────────────────────────────
    with open(train_json) as f:
        n_train = len(json.load(f))
    with open(val_json) as f:
        n_val = len(json.load(f))
    cprint(f"\n  Train: {n_train:,} images  |  Val: {n_val:,} images", C.GREEN)
    cprint(f"  Splits dir: {SPLITS_DIR}", C.WHITE)

    # ── Transforms + loaders ──────────────────────────────
    train_tf, val_tf = get_transforms()

    cprint("\n  Verifying image paths from manifest...", C.CYAN)
    try:
        train_loader, _ = make_loader(train_json, train_tf, shuffle=True,
                                      batch_size=BATCH_SIZE)
        val_loader,   _ = make_loader(val_json,   val_tf,   shuffle=False,
                                      batch_size=BATCH_SIZE)
    except FileNotFoundError as e:
        cprint(f"\n  ERROR: {e}", C.RED)
        return
    cprint("  ✅ Paths verified. All images accessible.", C.GREEN)

    # ── Class weights (handles imbalance in loss) ──────────
    cprint("\n  Computing class weights from training labels...", C.CYAN)
    class_weights = get_class_weights(train_json, NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Build model once, copy weights for each scheduler ─
    cprint("\n  Building EfficientNet-B3 base model...", C.CYAN)
    base_model    = build_model(NUM_CLASSES, DROPOUT_RATE, FREEZE_RATIO).to(device)
    initial_state = deepcopy(base_model.state_dict())
    trainable_p   = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    cprint(f"  Trainable parameters: {trainable_p:,}", C.GREEN)

    # ── Train with each scheduler ──────────────────────────
    histories = {}
    schedulers = ["ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR"]

    for scheduler_name in schedulers:
        cprint(f"\n{'='*80}", C.BLUE)
        cprint(f"  SCHEDULER: {scheduler_name}", C.BOLD + C.CYAN)
        cprint(f"  Epochs: {NUM_EPOCHS}  |  Initial LR: {INITIAL_LR}  |  "
               f"Batch: {BATCH_SIZE}  |  Device: {device}", C.WHITE)
        cprint(f"{'='*80}", C.BLUE)

        history = train_with_scheduler(
            scheduler_name, initial_state,
            train_loader, val_loader, n_train,
            NUM_EPOCHS, INITIAL_LR, WEIGHT_DECAY,
            criterion, device
        )
        histories[scheduler_name] = history

        best_acc = max(history['val_acc']) * 100
        cprint(f"\n  ✅ {scheduler_name} done.  "
               f"Best val_acc = {best_acc:.2f}%  |  "
               f"Best val_loss = {min(history['val_loss']):.4f}", C.GREEN + C.BOLD)

    # ── Plots + report ─────────────────────────────────────
    cprint("\n  Saving plots and report...", C.CYAN)
    generate_plots(histories, NUM_EPOCHS, OUTPUT_DIR)
    summary, best = save_report(histories, OUTPUT_DIR)

    # ── Final summary table ────────────────────────────────
    cprint("\n\n" + "="*80, C.CYAN)
    cprint("  TASK 6 RESULTS SUMMARY", C.BOLD + C.CYAN)
    cprint("="*80, C.CYAN)
    print()
    cprint(f"  {'Scheduler':<25} {'Best Val Loss':>14}  {'Best Acc (%)':>13}  "
           f"{'Best Epoch':>11}", C.BOLD)
    cprint("  " + "─"*68, C.WHITE)
    for name, info in summary.items():
        marker = "  ← BEST ✅" if name == best else ""
        col    = C.GREEN if name == best else C.WHITE
        cprint(
            f"  {name:<25} {info['best_val_loss']:>14.4f}  "
            f"{info['best_val_acc']:>12.2f}%  {info['best_epoch']:>11}{marker}",
            col
        )

    print()
    cprint(f"  Best scheduler : {best}", C.GREEN + C.BOLD)
    cprint(f"  → Use this in Task 7 for final model training.", C.WHITE)

    cprint("\n" + "="*80, C.CYAN)
    cprint("  TASK 6 COMPLETE ✅", C.GREEN + C.BOLD)
    cprint(f"  Outputs : {OUTPUT_DIR}", C.WHITE)
    cprint("  Files   : task6_report.json, plot10_lr_control.png", C.WHITE)
    cprint("\n  Next → Task 7: Model Training & Evaluation under 3 settings", C.YELLOW)
    cprint("="*80, C.CYAN)


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWS GUARD — required for PyTorch DataLoader with num_workers > 0
# Even with num_workers=0 this is good practice and costs nothing.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()