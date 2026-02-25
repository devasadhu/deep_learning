"""
=============================================================================
TASK 5: MODEL DESIGN USING TRANSFER LEARNING
=============================================================================
WCE Gastrointestinal Disease Classification Project

MODELS COMPARED:
  1. EfficientNet-B3   (efficient, ImageNet SOTA)
  2. MobileNetV3-Large (lightweight, mobile-friendly)
  3. ResNet101         (deep residual network, strong baseline)

STRATEGY:
  - Load ImageNet-pretrained weights
  - Freeze early layers (feature extractor)
  - Replace final classification head
  - Add Dropout + L2 regularisation
  - Print model summary (trainable vs frozen params)
  - Save model summaries and comparison table to JSON

REQUIREMENTS:
  pip install torch torchvision
=============================================================================
"""

import os
import json
import time

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision.models import (
        efficientnet_b3, EfficientNet_B3_Weights,
        mobilenet_v3_large, MobileNet_V3_Large_Weights,
        resnet101, ResNet101_Weights,
    )
except ImportError:
    print("ERROR: Run: pip install torch torchvision")
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = r"C:\Users\devas\Desktop\DL LAB\Major_project\task5_output"
SPLITS_DIR   = r"C:\Users\devas\Desktop\DL LAB\Major_project\task4_output\splits\kvasir_capsule"

NUM_CLASSES  = 14          # Kvasir-Capsule has 14 classes
DROPOUT_RATE = 0.4
L2_WEIGHT_DECAY = 1e-4     # Applied in optimizer (weight_decay param)
FREEZE_RATIO = 0.7         # Freeze first 70% of layers


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
    cprint("  TRANSFER LEARNING — KEY CONCEPTS", C.BOLD + C.CYAN)
    cprint("─"*80, C.BLUE)

    points = [
        ("Why Transfer Learning?",
         "Training CNNs from scratch needs millions of labelled images.\n"
         "    WCE datasets are too small (10–55K images) for that.\n"
         "    ImageNet-pretrained models already learned low-level features\n"
         "    (edges, textures, shapes) transferable to medical images."),

        ("Layer Freezing",
         "Early layers: generic features (edges, colour blobs) — FREEZE.\n"
         "    Late layers: task-specific features — FINE-TUNE.\n"
         "    We freeze the first 70% of layers and train the last 30%.\n"
         "    This prevents catastrophic forgetting of ImageNet knowledge."),

        ("Model Choice Rationale",
         "EfficientNet-B3: Best accuracy/parameter trade-off (SOTA 2019).\n"
         "    MobileNetV3: Lightest model, 5.4M params — real-time inference.\n"
         "    ResNet101: Proven deep baseline, easy to interpret gradients.\n"
         "    Three models allow direct comparison of depth vs efficiency."),

        ("Dropout + L2 Regularisation",
         "Dropout (p=0.4): randomly zeros neurons during training.\n"
         "    Prevents co-adaptation (each neuron learns independently).\n"
         "    L2 (weight_decay=1e-4): penalises large weights → generalization.\n"
         "    Critical for small medical datasets prone to overfitting."),

        ("Classification Head",
         "Original head replaced with: GlobalAvgPool → BN → Dropout → Linear.\n"
         "    Output size = NUM_CLASSES (14 for Kvasir-Capsule).\n"
         "    No softmax in head — CrossEntropyLoss applies it internally."),
    ]
    for title, desc in points:
        cprint(f"\n  ▶ {title}:", C.YELLOW + C.BOLD)
        print(f"    {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_efficientnet_b3(num_classes, dropout=0.4, freeze_ratio=0.7):
    """EfficientNet-B3 with custom classification head."""
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    # Freeze early layers
    all_params = list(model.parameters())
    freeze_up_to = int(len(all_params) * freeze_ratio)
    for i, param in enumerate(all_params):
        if i < freeze_up_to:
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_mobilenet_v3(num_classes, dropout=0.4, freeze_ratio=0.7):
    """MobileNetV3-Large with custom head."""
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    all_params = list(model.parameters())
    freeze_up_to = int(len(all_params) * freeze_ratio)
    for i, param in enumerate(all_params):
        if i < freeze_up_to:
            param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_resnet101(num_classes, dropout=0.4, freeze_ratio=0.7):
    """ResNet-101 with custom head."""
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

    all_params = list(model.parameters())
    freeze_up_to = int(len(all_params) * freeze_ratio)
    for i, param in enumerate(all_params):
        if i < freeze_up_to:
            param.requires_grad = False

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER COUNTER
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    return total, trainable, frozen


# ─────────────────────────────────────────────────────────────────────────────
# PRINT MODEL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_model_summary(name, model, total, trainable, frozen):
    cprint(f"\n  {'─'*70}", C.BLUE)
    cprint(f"  MODEL: {name}", C.BOLD + C.CYAN)
    cprint(f"  {'─'*70}", C.BLUE)

    cprint(f"\n  {'Parameter Type':<30} {'Count':>15}  {'%':>6}", C.BOLD)
    cprint("  " + "─"*55, C.WHITE)
    pct_train = trainable / total * 100
    pct_froz  = frozen    / total * 100
    cprint(f"  {'Total Parameters':<30} {total:>15,}  {'100.0%':>6}", C.WHITE)
    cprint(f"  {'Trainable (fine-tuned)':<30} {trainable:>15,}  {pct_train:>5.1f}%", C.GREEN)
    cprint(f"  {'Frozen (feature extractor)':<30} {frozen:>15,}  {pct_froz:>5.1f}%", C.YELLOW)

    memory_mb = (total * 4) / (1024 ** 2)   # float32 = 4 bytes
    cprint(f"\n  Model size (fp32)       : ~{memory_mb:.1f} MB", C.CYAN)
    cprint(f"  Dropout rate            : {DROPOUT_RATE}", C.WHITE)
    cprint(f"  L2 weight decay         : {L2_WEIGHT_DECAY}", C.WHITE)
    cprint(f"  Freeze ratio            : {FREEZE_RATIO*100:.0f}% of layers frozen", C.WHITE)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results):
    cprint("\n\n" + "="*80, C.CYAN)
    cprint("  MODEL COMPARISON TABLE — TASK 5", C.BOLD + C.CYAN)
    cprint("="*80, C.CYAN)
    print()

    cprint(f"  {'Model':<22} {'Total Params':>14}  {'Trainable':>12}  "
           f"{'Frozen':>12}  {'Size (MB)':>10}  {'ImageNet Acc':>13}", C.BOLD)
    cprint("  " + "─"*90, C.WHITE)

    imagenet_acc = {
        "EfficientNet-B3":   "82.0%",
        "MobileNetV3-Large": "75.3%",
        "ResNet-101":        "81.9%",
    }

    for name, info in results.items():
        total    = info['total']
        train    = info['trainable']
        frozen   = info['frozen']
        size_mb  = info['size_mb']
        acc      = imagenet_acc.get(name, "N/A")

        cprint(f"  {name:<22} {total:>14,}  {train:>12,}  {frozen:>12,}  "
               f"{size_mb:>9.1f}  {acc:>13}", C.WHITE)

    print()
    cprint("  RECOMMENDED: EfficientNet-B3 → best accuracy with reasonable size.", C.GREEN + C.BOLD)
    cprint("  MobileNetV3 → deploy on edge devices (raspberry pi, mobile phone).", C.WHITE)
    cprint("  ResNet-101  → strong interpretability, good gradient flow.", C.WHITE)
    print()
    cprint("  All models use:", C.YELLOW)
    cprint(f"    Dropout={DROPOUT_RATE}, L2 weight_decay={L2_WEIGHT_DECAY}, "
           f"freeze_ratio={FREEZE_RATIO}", C.WHITE)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_report(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "task":            "Task 5 - Transfer Learning",
        "num_classes":     NUM_CLASSES,
        "dropout":         DROPOUT_RATE,
        "l2_weight_decay": L2_WEIGHT_DECAY,
        "freeze_ratio":    FREEZE_RATIO,
        "models":          results,
        "recommendation":  "EfficientNet-B3 for best accuracy, "
                           "MobileNetV3 for deployment, ResNet-101 for baseline."
    }
    path = os.path.join(output_dir, "task5_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    cprint(f"\n  📄 Report saved: {path}", C.GREEN)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────

def generate_plot(results, output_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        cprint("  ⚠  matplotlib not found. Skipping plot.", C.YELLOW)
        return

    os.makedirs(output_dir, exist_ok=True)

    names      = list(results.keys())
    trainable  = [results[n]['trainable'] / 1e6 for n in names]
    frozen     = [results[n]['frozen']    / 1e6 for n in names]
    total      = [results[n]['total']     / 1e6 for n in names]
    sizes      = [results[n]['size_mb']          for n in names]

    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')

    # Left: stacked param bar
    ax = axes[0]
    ax.bar(x, frozen,    label='Frozen',    color='#e74c3c', alpha=0.85)
    ax.bar(x, trainable, label='Trainable', color='#2ecc71', alpha=0.85,
           bottom=frozen)
    for i, t in enumerate(total):
        ax.text(i, t + 0.2, f'{t:.1f}M', ha='center', color='white',
                fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, color='white', fontsize=11)
    ax.set_ylabel('Parameters (Millions)', color='white')
    ax.set_title('Frozen vs Trainable Parameters', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.yaxis.grid(True, alpha=0.3, color='gray')
    ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')

    # Right: model size
    ax2 = axes[1]
    bars = ax2.bar(x, sizes, color=['#3498db', '#9b59b6', '#f39c12'], alpha=0.85, width=0.5)
    for b, v in zip(bars, sizes):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                 f'{v:.0f} MB', ha='center', color='white', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, color='white', fontsize=11)
    ax2.set_ylabel('Model Size (MB, fp32)', color='white')
    ax2.set_title('Model Memory Footprint', color='white', fontsize=12)
    ax2.tick_params(colors='white')
    ax2.spines[:].set_color('#444')
    ax2.set_facecolor('#16213e')
    ax2.yaxis.grid(True, alpha=0.3, color='gray')

    plt.suptitle('Task 5: Transfer Learning — Model Comparison', color='white',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "plot9_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    cprint(f"  🖼  Saved: {path}", C.GREEN)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

MODEL_BUILDERS = {
    "EfficientNet-B3":   build_efficientnet_b3,
    "MobileNetV3-Large": build_mobilenet_v3,
    "ResNet-101":        build_resnet101,
}

def main():
    cprint("\n" + "="*80, C.CYAN)
    cprint("  TASK 5: TRANSFER LEARNING MODEL DESIGN", C.BOLD)
    cprint("  SVNIT Surat | Prof. Praveen Kumar Chandaliya", C.WHITE)
    cprint("="*80, C.CYAN)

    print_theory()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cprint(f"\n  Device: {device}", C.CYAN)
    if device.type == "cuda":
        cprint(f"  GPU: {torch.cuda.get_device_name(0)}", C.GREEN)

    cprint("\n\n  Building models...", C.CYAN)
    cprint("─"*80, C.BLUE)

    results = {}

    for name, builder in MODEL_BUILDERS.items():
        cprint(f"\n  Loading {name} with ImageNet weights...", C.CYAN)
        t0 = time.time()
        model = builder(NUM_CLASSES, DROPOUT_RATE, FREEZE_RATIO)
        model = model.to(device)
        elapsed = time.time() - t0

        total, trainable, frozen = count_params(model)
        size_mb = (total * 4) / (1024 ** 2)

        print_model_summary(name, model, total, trainable, frozen)
        cprint(f"  Build time: {elapsed:.2f}s", C.WHITE)

        results[name] = {
            "total":        total,
            "trainable":    trainable,
            "frozen":       frozen,
            "size_mb":      round(size_mb, 1),
            "trainable_pct": round(trainable / total * 100, 1),
        }

        # Save model weights
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model_path = os.path.join(OUTPUT_DIR, f"{name.replace('-', '_').replace(' ', '_')}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes":      NUM_CLASSES,
            "dropout":          DROPOUT_RATE,
            "freeze_ratio":     FREEZE_RATIO,
        }, model_path)
        cprint(f"  💾 Model saved: {model_path}", C.GREEN)

    print_comparison_table(results)
    save_report(results, OUTPUT_DIR)
    generate_plot(results, OUTPUT_DIR)

    cprint("\n" + "="*80, C.CYAN)
    cprint("  TASK 5 COMPLETE ✅", C.GREEN + C.BOLD)
    cprint(f"  Outputs: {OUTPUT_DIR}", C.WHITE)
    cprint("\n  Next → Task 6: Intelligent Learning Rate Control", C.YELLOW)
    cprint("="*80, C.CYAN)


if __name__ == "__main__":
    main()