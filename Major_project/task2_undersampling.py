"""
=============================================================================
TASK 2: UNDER-SAMPLING FOR CLASS IMBALANCE HANDLING
=============================================================================
WCE Gastrointestinal Disease Classification Project

STRATEGY:
  - Apply under-sampling ONLY to Kvasir-Capsule (3,434:1 imbalance)
  - Cap majority classes at UPPER_CAP = 3000
  - Preserve ALL minority class samples (never under-sample rare classes)
  - Copy balanced subset to task2_output/kvasir_capsule_balanced/
  - Generate before/after report and plots

=============================================================================
"""

import os
import json
import random
import shutil

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

KVASIR_CAPSULE_ROOT = r"C:\Users\devas\Desktop\DL LAB\Major_project\datasets\kvasir_capsule"
OUTPUT_DIR          = r"C:\Users\devas\Desktop\DL LAB\Major_project\task2_output"
UPPER_CAP           = 3000    # Max images per class after under-sampling
RANDOM_SEED         = 42
IMG_EXTENSIONS      = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

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

def bar(count, max_count, width=35):
    filled = int((count / max_count) * width) if max_count > 0 else 0
    return '█' * filled + '░' * (width - filled)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATASET FROM DISK
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(root_path):
    class_images = {}
    for cls in sorted(os.listdir(root_path)):
        cls_path = os.path.join(root_path, cls)
        if not os.path.isdir(cls_path):
            continue
        images = [
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
        if images:
            class_images[cls] = images
    return class_images

# ─────────────────────────────────────────────────────────────────────────────
# UNDER-SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def apply_undersampling(class_images, cap, seed):
    random.seed(seed)
    sampled = {}
    info = {}
    for cls, images in class_images.items():
        original = len(images)
        if original > cap:
            selected = random.sample(images, cap)
            action = "REDUCED"
        else:
            selected = images
            action = "KEPT"
        sampled[cls] = selected
        info[cls] = {
            "original": original,
            "sampled":  len(selected),
            "removed":  original - len(selected),
            "action":   action
        }
    return sampled, info

# ─────────────────────────────────────────────────────────────────────────────
# PRINT THEORY — WHY UNDER-SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def print_theory():
    cprint("\n" + "─"*80, C.BLUE)
    cprint("  WHY UNDER-SAMPLING?", C.BOLD + C.CYAN)
    cprint("─"*80, C.BLUE)
    points = [
        ("Problem",
         "Kvasir-Capsule has 3,434:1 imbalance. Normal Clean Mucosa = 72.7%\n"
         "    of all data. A naive model predicting 'normal' always gets 73%\n"
         "    accuracy but detects ZERO diseases — clinically useless."),
        ("Solution",
         "Randomly remove samples from MAJORITY classes only.\n"
         "    Minority classes (Polyp=55, Hematin=12, Ampulla=10) are NEVER touched."),
        ("Cap = 3,000",
         "Reduces imbalance from 3,434:1 to ~55:1.\n"
         "    Dataset shrinks from 47K to ~16K images (faster training).\n"
         "    3,000 images per class is still sufficient for deep learning."),
        ("Remaining imbalance",
         "Task 3 (augmentation) will further address extreme minority classes.\n"
         "    Class weights during training handle residual imbalance."),
    ]
    for title, desc in points:
        cprint(f"\n  ▶ {title}:", C.YELLOW + C.BOLD)
        print(f"    {desc}")

# ─────────────────────────────────────────────────────────────────────────────
# PRINT RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_report(info, cap):
    print()
    cprint("─"*80, C.BLUE)
    cprint("  BEFORE vs AFTER UNDER-SAMPLING", C.BOLD + C.CYAN)
    cprint("─"*80, C.BLUE)
    print()

    max_orig = max(v["original"] for v in info.values())
    total_before = total_after = total_removed = 0

    cprint(f"  {'Class':<30} {'Before':>7}  {'After':>7}  {'Removed':>8}  Status", C.BOLD)
    cprint("  " + "─"*72, C.WHITE)

    for cls, v in sorted(info.items(), key=lambda x: -x[1]['original']):
        orig    = v["original"]
        sampled = v["sampled"]
        removed = v["removed"]
        action  = v["action"]
        total_before  += orig
        total_after   += sampled
        total_removed += removed

        if action == "REDUCED":
            col    = C.RED
            status = f"{C.RED}▼ REDUCED{C.RESET}"
        elif orig < 100:
            col    = C.MAGENTA
            status = f"{C.MAGENTA}⚠ MINORITY{C.RESET}"
        else:
            col    = C.GREEN
            status = f"{C.GREEN}✓ KEPT{C.RESET}"

        bar_str = bar(sampled, max_orig)
        print(f"  {col}{cls:<30}{C.RESET} {orig:>7,}  {sampled:>7,}  {removed:>8,}  {status}")

    cprint("  " + "─"*72, C.WHITE)
    cprint(f"  {'TOTAL':<30} {total_before:>7,}  {total_after:>7,}  {total_removed:>8,}", C.BOLD)

    print()
    before_ratio = max(v["original"] for v in info.values()) / min(v["original"] for v in info.values())
    after_ratio  = max(v["sampled"]  for v in info.values()) / min(v["sampled"]  for v in info.values())
    reduction    = (total_removed / total_before) * 100
    improvement  = before_ratio / after_ratio

    cprint(f"  Imbalance ratio : {before_ratio:>8,.0f}:1  →  {after_ratio:,.0f}:1", C.CYAN)
    cprint(f"  Dataset size    : {total_before:,} → {total_after:,} ({reduction:.1f}% reduction)", C.CYAN)
    cprint(f"  Improvement     : {improvement:.0f}x less imbalanced ✅", C.GREEN + C.BOLD)
    print()
    cprint("  ⚠  Minority classes untouched — Task 3 will augment them:", C.YELLOW)
    cprint("     Polyp (55), Blood-Hematin (12), Ampulla of Vater (10)", C.YELLOW)

# ─────────────────────────────────────────────────────────────────────────────
# COPY BALANCED DATASET
# ─────────────────────────────────────────────────────────────────────────────

def copy_balanced_dataset(sampled, output_dir):
    balanced_dir = os.path.join(output_dir, "kvasir_capsule_balanced")
    total = sum(len(v) for v in sampled.values())
    cprint(f"\n  Copying {total:,} selected images to output...", C.CYAN)
    for cls, images in sampled.items():
        cls_out = os.path.join(balanced_dir, cls)
        os.makedirs(cls_out, exist_ok=True)
        for src in images:
            shutil.copy2(src, os.path.join(cls_out, os.path.basename(src)))
    cprint(f"  ✅ Balanced dataset saved to:", C.GREEN)
    cprint(f"     {balanced_dir}", C.WHITE)
    return balanced_dir

# ─────────────────────────────────────────────────────────────────────────────
# SAVE JSON REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_json_report(info, cap, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    before_counts = {k: v["original"] for k, v in info.items()}
    after_counts  = {k: v["sampled"]  for k, v in info.items()}
    report = {
        "task":             "Task 2 - Under-sampling",
        "cap_threshold":    cap,
        "random_seed":      RANDOM_SEED,
        "before_total":     sum(before_counts.values()),
        "after_total":      sum(after_counts.values()),
        "images_removed":   sum(before_counts.values()) - sum(after_counts.values()),
        "reduction_pct":    round((1 - sum(after_counts.values()) / sum(before_counts.values())) * 100, 2),
        "imbalance_before": round(max(before_counts.values()) / min(before_counts.values()), 1),
        "imbalance_after":  round(max(after_counts.values())  / min(after_counts.values()),  1),
        "per_class":        info
    }
    path = os.path.join(output_dir, "task2_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    cprint(f"  📄 JSON report saved: {path}", C.GREEN)

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(info, cap, output_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        cprint("  ⚠  matplotlib not found. Run: pip install matplotlib", C.YELLOW)
        return

    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(info.keys(), key=lambda c: -info[c]['original'])
    before  = [info[c]['original'] for c in classes]
    after   = [info[c]['sampled']  for c in classes]
    x       = np.arange(len(classes))
    width   = 0.4

    # ── Plot 5: Before vs After ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')

    ax = axes[0]
    ax.bar(x - width/2, before, width, label='Before', color='#e74c3c', alpha=0.85)
    ax.bar(x + width/2, after,  width, label='After',  color='#2ecc71', alpha=0.85)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8, color='white')
    ax.set_ylabel('Number of Images (log scale)', color='white')
    ax.set_title(f'Kvasir-Capsule: Before vs After Under-sampling\nCap = {cap:,}', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.axhline(y=cap, color='#f39c12', linestyle='--', linewidth=1.5, label=f'Cap={cap:,}')
    ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
    ax.yaxis.grid(True, alpha=0.3, color='gray')

    ax2 = axes[1]
    before_ratio = max(before) / min(before)
    after_ratio  = max(after)  / min(after)
    bars2 = ax2.bar(
        [f'Before\n{before_ratio:,.0f}:1', f'After\n{after_ratio:,.0f}:1'],
        [before_ratio, after_ratio],
        color=['#e74c3c', '#2ecc71'], alpha=0.85, width=0.5
    )
    for b, v in zip(bars2, [before_ratio, after_ratio]):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
                 f'{v:,.0f}:1', ha='center', color='white', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Imbalance Ratio (max/min)', color='white')
    ax2.set_title('Imbalance Ratio: Before vs After', color='white', fontsize=12)
    ax2.tick_params(colors='white')
    ax2.spines[:].set_color('#444')
    ax2.set_facecolor('#16213e')
    ax2.yaxis.grid(True, alpha=0.3, color='gray')

    plt.tight_layout(pad=3)
    p1 = os.path.join(output_dir, "plot5_undersampling_before_after.png")
    plt.savefig(p1, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    cprint(f"  🖼  Saved: {p1}", C.GREEN)

    # ── Plot 6: After distribution horizontal bar ─────────
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    sorted_cls    = sorted(info.keys(), key=lambda c: info[c]['sampled'])
    sorted_counts = [info[c]['sampled'] for c in sorted_cls]
    bar_cols      = ['#e74c3c' if info[c]['action'] == 'REDUCED' else
                     ('#9b59b6' if info[c]['sampled'] < 100 else '#2ecc71')
                     for c in sorted_cls]

    ax.barh(sorted_cls, sorted_counts, color=bar_cols, alpha=0.85)
    for i, (cls, v) in enumerate(zip(sorted_cls, sorted_counts)):
        ax.text(v + 20, i, f'{v:,}', va='center', color='white', fontsize=8)
    ax.axvline(x=cap, color='#f39c12', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Number of Images', color='white')
    ax.set_title(f'Class Distribution After Under-sampling (Cap={cap:,})', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.xaxis.grid(True, alpha=0.3, color='gray')
    ax.legend(handles=[
        mpatches.Patch(color='#e74c3c', label='Under-sampled (was above cap)'),
        mpatches.Patch(color='#2ecc71', label='Kept as-is'),
        mpatches.Patch(color='#9b59b6', label='Minority class (<100 images)'),
        plt.Line2D([0],[0], color='#f39c12', linestyle='--', label=f'Cap={cap:,}'),
    ], facecolor='#2d2d2d', edgecolor='white', labelcolor='white', loc='lower right')

    plt.tight_layout()
    p2 = os.path.join(output_dir, "plot6_after_distribution.png")
    plt.savefig(p2, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    cprint(f"  🖼  Saved: {p2}", C.GREEN)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cprint("\n" + "="*80, C.CYAN)
    cprint("  TASK 2: UNDER-SAMPLING | Kvasir-Capsule Dataset", C.BOLD)
    cprint("  SVNIT Surat | Prof. Praveen Kumar Chandaliya", C.WHITE)
    cprint("="*80, C.CYAN)

    if not os.path.exists(KVASIR_CAPSULE_ROOT):
        cprint(f"\n  ERROR: Dataset not found at:\n  {KVASIR_CAPSULE_ROOT}", C.RED)
        return

    print_theory()

    cprint(f"\n  Loading Kvasir-Capsule from:\n  {KVASIR_CAPSULE_ROOT}", C.CYAN)
    class_images = load_dataset(KVASIR_CAPSULE_ROOT)
    cprint(f"  Found {len(class_images)} classes, "
           f"{sum(len(v) for v in class_images.values()):,} total images ✅", C.GREEN)

    cprint(f"\n  Applying under-sampling (cap = {UPPER_CAP:,}, seed = {RANDOM_SEED})...", C.CYAN)
    sampled, info = apply_undersampling(class_images, UPPER_CAP, RANDOM_SEED)

    print_report(info, UPPER_CAP)

    print()
    cprint("─"*80, C.BLUE)
    cprint("  Copy selected images to task2_output/kvasir_capsule_balanced/?", C.BOLD)
    cprint("  This physically copies ~16K images to a new balanced folder.", C.WHITE)
    cprint("  Required for Tasks 3 and 4 to work on the balanced dataset.", C.WHITE)
    ans = input("\n  Copy files? (y/n): ").strip().lower()
    if ans == 'y':
        copy_balanced_dataset(sampled, OUTPUT_DIR)
    else:
        cprint("  Skipping file copy. Report and plots will still be saved.", C.YELLOW)

    cprint("\n  Saving outputs...", C.CYAN)
    save_json_report(info, UPPER_CAP, OUTPUT_DIR)
    generate_plots(info, UPPER_CAP, OUTPUT_DIR)

    cprint("\n" + "="*80, C.CYAN)
    cprint("  TASK 2 COMPLETE ✅", C.GREEN + C.BOLD)
    cprint(f"  Outputs in: {OUTPUT_DIR}", C.WHITE)
    cprint("  Files: task2_report.json, plot5_*.png, plot6_*.png", C.WHITE)
    cprint("\n  Next → Task 3: Augmentation for minority classes", C.YELLOW)
    cprint("  Target classes: Polyp (55), Blood-Hematin (12), Ampulla of Vater (10)", C.YELLOW)
    cprint("="*80, C.CYAN)

if __name__ == "__main__":
    main()
