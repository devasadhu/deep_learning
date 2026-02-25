"""
=============================================================================
TASK 4: PREPROCESSING — RESIZE, NORMALIZE, TRAIN/VAL/TEST SPLIT
=============================================================================
WCE Gastrointestinal Disease Classification Project

WHAT THIS DOES:
  1. Resize all images to 224x224 (EfficientNetV2 standard input size)
  2. Verify images are valid (not corrupted)
  3. Create stratified Train/Val/Test splits for all 5 datasets
     - 70% Train | 15% Validation | 15% Test
     - Stratified: each split has proportional class representation
  4. Save split manifests as JSON (image path + label per split)
  5. Compute and save normalization statistics (mean, std per channel)
  6. Generate preprocessing report and plots

NOTE ON NORMALIZATION:
  We use ImageNet pretrained statistics for EfficientNetV2:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
  Actual pixel-level normalization happens in the DataLoader (Task 5),
  not here. Here we just verify and compute dataset statistics.
=============================================================================
"""

import os
import json
import random
import shutil
import math
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Run: pip install Pillow numpy")
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE = r"C:\Users\devas\Desktop\DL LAB\Major_project"

# Input datasets
DATASETS = {
    "kvasir_capsule": os.path.join(BASE, "task3_output", "kvasir_capsule_augmented"),
    "kvasir_v2":      os.path.join(BASE, "datasets", "kvasir_v2"),
    "CVC-ClinicDB":   os.path.join(BASE, "datasets", "CVC-ClinicDB"),
    "ETIS-Larib":     os.path.join(BASE, "datasets", "ETIS-Larib"),
    "KID":            os.path.join(BASE, "datasets", "KID"),
}

# Output
OUTPUT_DIR   = os.path.join(BASE, "task4_output")
RESIZED_DIR  = os.path.join(OUTPUT_DIR, "resized")   # resized images
SPLITS_DIR   = os.path.join(OUTPUT_DIR, "splits")    # JSON split manifests

TARGET_SIZE  = (224, 224)   # EfficientNetV2 input
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42
IMG_EXTS     = {'.jpg', '.jpeg', '.png', '.bmp'}

# ImageNet normalization stats (used by EfficientNetV2 pretrained weights)
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
    cprint("  WHY THESE PREPROCESSING STEPS?", C.BOLD + C.CYAN)
    cprint("─"*80, C.BLUE)
    points = [
        ("Resize to 224×224",
         "EfficientNetV2 expects fixed-size input.\n"
         "    All 5 datasets have different resolutions (336×336, 720×576, 1225×966 etc.)\n"
         "    224×224 is the standard input size for pretrained EfficientNetV2 on ImageNet."),
        ("Stratified Train/Val/Test split (70/15/15)",
         "Stratified = each class has 70/15/15 split proportionally.\n"
         "    Prevents a class appearing ONLY in train but not in test.\n"
         "    Critical for imbalanced datasets — random split can miss minority classes entirely."),
        ("Why 70/15/15?",
         "Standard split for medical imaging benchmarks.\n"
         "    15% test is enough for statistical significance with 15K+ images.\n"
         "    Larger train set needed because of class imbalance."),
        ("Normalization (ImageNet stats)",
         "EfficientNetV2 was pretrained on ImageNet.\n"
         "    Using ImageNet mean/std aligns our data with pretrained weights.\n"
         "    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]\n"
         "    Applied in DataLoader during training (not hardcoded into image files)."),
        ("Split manifests as JSON",
         "Store {filepath, label, split} for each image.\n"
         "    DataLoader reads these lists — no data copying needed.\n"
         "    Reproducible: same seed = same split every time."),
    ]
    for title, desc in points:
        cprint(f"\n  ▶ {title}:", C.YELLOW + C.BOLD)
        print(f"    {desc}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: RESIZE IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def resize_dataset(ds_name, src_root, dst_root):
    """Resize all images in a dataset to TARGET_SIZE."""
    cprint(f"\n  [{ds_name}] Resizing to {TARGET_SIZE[0]}×{TARGET_SIZE[1]}...", C.CYAN)

    if not os.path.exists(src_root):
        cprint(f"  ERROR: Source not found: {src_root}", C.RED)
        return {}, []

    class_map   = {}   # class_name -> label_index
    errors      = []
    total       = 0
    skipped     = 0
    label_idx   = 0

    for cls in sorted(os.listdir(src_root)):
        cls_src = os.path.join(src_root, cls)
        if not os.path.isdir(cls_src):
            continue

        cls_dst = os.path.join(dst_root, ds_name, cls)
        os.makedirs(cls_dst, exist_ok=True)
        class_map[cls] = label_idx
        label_idx += 1

        images = [f for f in os.listdir(cls_src)
                  if os.path.splitext(f)[1].lower() in IMG_EXTS]

        for fname in images:
            src_path = os.path.join(cls_src, fname)
            # Save as .jpg for consistency
            dst_name = os.path.splitext(fname)[0] + '.jpg'
            dst_path = os.path.join(cls_dst, dst_name)

            if os.path.exists(dst_path):
                skipped += 1
                total   += 1
                continue

            try:
                img = Image.open(src_path).convert('RGB')
                img = img.resize(TARGET_SIZE, Image.BILINEAR)
                img.save(dst_path, 'JPEG', quality=95)
                total += 1
            except Exception as e:
                errors.append(f"{src_path}: {e}")

        n = len(images)
        cprint(f"    {cls:<35} {n:>5} images → resized", C.WHITE)

    cprint(f"  Done. {total} images resized, {skipped} skipped (already done), "
           f"{len(errors)} errors.", C.GREEN)
    return class_map, errors

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(ds_name, resized_root, class_map, seed=42):
    """
    Create stratified train/val/test split.
    Returns dict: {'train': [...], 'val': [...], 'test': [...]}
    Each entry: {'path': str, 'label': int, 'class': str}
    """
    random.seed(seed)
    splits = {'train': [], 'val': [], 'test': []}

    ds_root = os.path.join(resized_root, ds_name)
    if not os.path.exists(ds_root):
        cprint(f"  ERROR: Resized dataset not found: {ds_root}", C.RED)
        return splits, {}

    split_info = {}

    for cls, label in sorted(class_map.items(), key=lambda x: x[1]):
        cls_path = os.path.join(ds_root, cls)
        if not os.path.exists(cls_path):
            continue

        images = sorted([
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if os.path.splitext(f)[1].lower() in IMG_EXTS | {'.jpg'}
        ])

        random.shuffle(images)
        n       = len(images)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val   = max(1, int(n * VAL_RATIO))
        n_test  = n - n_train - n_val

        # Ensure at least 1 in each split for tiny classes
        if n_test < 1:
            n_test  = 1
            n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1

        train_imgs = images[:n_train]
        val_imgs   = images[n_train:n_train + n_val]
        test_imgs  = images[n_train + n_val:]

        for img_path in train_imgs:
            splits['train'].append({'path': img_path, 'label': label, 'class': cls})
        for img_path in val_imgs:
            splits['val'].append({'path': img_path, 'label': label, 'class': cls})
        for img_path in test_imgs:
            splits['test'].append({'path': img_path, 'label': label, 'class': cls})

        split_info[cls] = {
            'total': n,
            'train': len(train_imgs),
            'val':   len(val_imgs),
            'test':  len(test_imgs),
            'label': label
        }

    # Shuffle splits
    for split in splits:
        random.shuffle(splits[split])

    return splits, split_info

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SAVE SPLIT MANIFESTS
# ─────────────────────────────────────────────────────────────────────────────

def save_splits(ds_name, splits, split_info, class_map, splits_dir):
    ds_split_dir = os.path.join(splits_dir, ds_name)
    os.makedirs(ds_split_dir, exist_ok=True)

    # Save each split
    for split_name, entries in splits.items():
        path = os.path.join(ds_split_dir, f"{split_name}.json")
        with open(path, 'w') as f:
            json.dump(entries, f, indent=2)

    # Save class map
    class_map_path = os.path.join(ds_split_dir, "class_map.json")
    with open(class_map_path, 'w') as f:
        json.dump(class_map, f, indent=2)

    # Save split info
    info_path = os.path.join(ds_split_dir, "split_info.json")
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    cprint(f"  📄 Splits saved to: {ds_split_dir}", C.GREEN)

# ─────────────────────────────────────────────────────────────────────────────
# PRINT SPLIT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_split_table(ds_name, splits, split_info):
    cprint(f"\n  {ds_name} — Train/Val/Test Split:", C.BOLD)
    cprint(f"  {'Class':<30} {'Total':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}", C.CYAN)
    cprint("  " + "─"*58, C.WHITE)

    for cls, info in sorted(split_info.items(), key=lambda x: -x[1]['total']):
        print(f"  {cls:<30} {info['total']:>6}  {info['train']:>6}  "
              f"{info['val']:>5}  {info['test']:>5}")

    cprint("  " + "─"*58, C.WHITE)
    totals = {s: len(splits[s]) for s in splits}
    cprint(f"  {'TOTAL':<30} {sum(totals.values()):>6}  "
           f"{totals['train']:>6}  {totals['val']:>5}  {totals['test']:>5}", C.BOLD)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE DATASET STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(ds_name, resized_root, sample_limit=500):
    """
    Compute mean and std of pixel values across a sample of images.
    """
    ds_path = os.path.join(resized_root, ds_name)
    if not os.path.exists(ds_path):
        return None

    all_images = []
    for root, dirs, files in os.walk(ds_path):
        for f in files:
            if f.endswith('.jpg'):
                all_images.append(os.path.join(root, f))

    # Sample for speed
    random.seed(42)
    sample = random.sample(all_images, min(sample_limit, len(all_images)))

    means = []
    stds  = []

    for img_path in sample:
        try:
            img = Image.open(img_path).convert('RGB')
            arr = np.array(img).astype(np.float32) / 255.0
            means.append(arr.mean(axis=(0,1)))
            stds.append(arr.std(axis=(0,1)))
        except:
            continue

    if not means:
        return None

    mean = np.mean(means, axis=0).tolist()
    std  = np.mean(stds,  axis=0).tolist()
    return {'mean': mean, 'std': std, 'sample_size': len(sample)}

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(all_results, output_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        cprint("  ⚠  matplotlib not found. Skipping plots.", C.YELLOW)
        return

    os.makedirs(output_dir, exist_ok=True)

    # Plot: Train/Val/Test split sizes per dataset
    ds_names = list(all_results.keys())
    train_counts = [all_results[ds]['splits']['train'] for ds in ds_names]
    val_counts   = [all_results[ds]['splits']['val']   for ds in ds_names]
    test_counts  = [all_results[ds]['splits']['test']  for ds in ds_names]

    x     = np.arange(len(ds_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    b1 = ax.bar(x - width, train_counts, width, label='Train (70%)', color='#2ecc71', alpha=0.85)
    b2 = ax.bar(x,         val_counts,   width, label='Val   (15%)', color='#f39c12', alpha=0.85)
    b3 = ax.bar(x + width, test_counts,  width, label='Test  (15%)', color='#e74c3c', alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 20,
                    f'{int(h):,}', ha='center', va='bottom',
                    color='white', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=15, ha='right', color='white', fontsize=10)
    ax.set_ylabel('Number of Images', color='white')
    ax.set_title('Task 4: Train/Val/Test Split Sizes Across All 5 Datasets', color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.yaxis.grid(True, alpha=0.3, color='gray')
    ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    p = os.path.join(output_dir, "plot8_train_val_test_splits.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    cprint(f"  🖼  Saved: {p}", C.GREEN)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE MASTER REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_master_report(all_results, stats_dict, output_dir):
    report = {
        "task":           "Task 4 - Preprocessing",
        "target_size":    list(TARGET_SIZE),
        "train_ratio":    TRAIN_RATIO,
        "val_ratio":      VAL_RATIO,
        "test_ratio":     TEST_RATIO,
        "random_seed":    RANDOM_SEED,
        "imagenet_mean":  IMAGENET_MEAN,
        "imagenet_std":   IMAGENET_STD,
        "datasets":       all_results,
        "pixel_stats":    stats_dict,
    }
    path = os.path.join(output_dir, "task4_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    cprint(f"  📄 Master report: {path}", C.GREEN)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cprint("\n" + "="*80, C.CYAN)
    cprint("  TASK 4: PREPROCESSING — RESIZE + NORMALIZE + SPLIT", C.BOLD)
    cprint("  SVNIT Surat | Prof. Praveen Kumar Chandaliya", C.WHITE)
    cprint("="*80, C.CYAN)

    print_theory()

    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR,  exist_ok=True)

    all_results = {}
    stats_dict  = {}

    for ds_name, src_root in DATASETS.items():

        cprint(f"\n{'='*80}", C.BLUE)
        cprint(f"  DATASET: {ds_name}", C.BOLD + C.CYAN)
        cprint(f"{'='*80}", C.BLUE)

        if not os.path.exists(src_root):
            cprint(f"  ⚠  Skipping — not found: {src_root}", C.YELLOW)
            continue

        # ── Step 1: Resize ────────────────────────────────
        class_map, errors = resize_dataset(ds_name, src_root, RESIZED_DIR)
        if not class_map:
            continue
        if errors:
            cprint(f"  ⚠  {len(errors)} errors during resize:", C.YELLOW)
            for e in errors[:5]:
                print(f"    {e}")

        # ── Step 2: Stratified split ──────────────────────
        cprint(f"\n  Creating stratified 70/15/15 split...", C.CYAN)
        splits, split_info = stratified_split(ds_name, RESIZED_DIR, class_map, RANDOM_SEED)

        # ── Step 3: Print table ───────────────────────────
        print_split_table(ds_name, splits, split_info)

        # ── Step 4: Save splits ───────────────────────────
        save_splits(ds_name, splits, split_info, class_map, SPLITS_DIR)

        # ── Step 5: Compute pixel stats (sample) ─────────
        cprint(f"\n  Computing pixel statistics (sample of 500 images)...", C.CYAN)
        stats = compute_stats(ds_name, RESIZED_DIR)
        if stats:
            stats_dict[ds_name] = stats
            cprint(f"  Pixel mean (RGB): {[round(v,3) for v in stats['mean']]}", C.WHITE)
            cprint(f"  Pixel std  (RGB): {[round(v,3) for v in stats['std']]}", C.WHITE)
            cprint(f"  ImageNet mean:    {IMAGENET_MEAN}  ← will use for normalization", C.YELLOW)

        all_results[ds_name] = {
            "source":     src_root,
            "classes":    class_map,
            "num_classes": len(class_map),
            "splits": {
                "train": len(splits['train']),
                "val":   len(splits['val']),
                "test":  len(splits['test']),
                "total": len(splits['train']) + len(splits['val']) + len(splits['test']),
            },
            "errors": len(errors)
        }

    # ── Final summary ─────────────────────────────────────
    print()
    cprint("="*80, C.CYAN)
    cprint("  FINAL SUMMARY — ALL DATASETS", C.BOLD + C.CYAN)
    cprint("="*80, C.CYAN)
    print()
    cprint(f"  {'Dataset':<20} {'Classes':>8}  {'Train':>7}  {'Val':>6}  {'Test':>6}  {'Total':>7}", C.BOLD)
    cprint("  " + "─"*65, C.WHITE)

    grand_train = grand_val = grand_test = 0
    for ds, info in all_results.items():
        tr = info['splits']['train']
        va = info['splits']['val']
        te = info['splits']['test']
        grand_train += tr
        grand_val   += va
        grand_test  += te
        print(f"  {ds:<20} {info['num_classes']:>8}  {tr:>7,}  {va:>6,}  {te:>6,}  {tr+va+te:>7,}")

    cprint("  " + "─"*65, C.WHITE)
    grand_total = grand_train + grand_val + grand_test
    cprint(f"  {'GRAND TOTAL':<20} {'':>8}  {grand_train:>7,}  {grand_val:>6,}  "
           f"{grand_test:>6,}  {grand_total:>7,}", C.BOLD)

    print()
    cprint("  Normalization (apply in DataLoader, Task 5):", C.YELLOW + C.BOLD)
    cprint(f"    mean = {IMAGENET_MEAN}", C.WHITE)
    cprint(f"    std  = {IMAGENET_STD}", C.WHITE)
    cprint("    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])", C.WHITE)

    # Save outputs
    print()
    cprint("  Saving outputs...", C.CYAN)
    save_master_report(all_results, stats_dict, OUTPUT_DIR)
    generate_plots(all_results, OUTPUT_DIR)

    cprint("\n" + "="*80, C.CYAN)
    cprint("  TASK 4 COMPLETE ✅", C.GREEN + C.BOLD)
    cprint(f"  Resized images : {RESIZED_DIR}", C.WHITE)
    cprint(f"  Split manifests: {SPLITS_DIR}", C.WHITE)
    cprint(f"  Report         : {OUTPUT_DIR}", C.WHITE)
    cprint("\n  Next → Task 5: EfficientNetV2 Teacher Model", C.YELLOW)
    cprint("  The split JSONs will be used by the DataLoader in Task 5.", C.YELLOW)
    cprint("="*80, C.CYAN)


if __name__ == "__main__":
    main()
