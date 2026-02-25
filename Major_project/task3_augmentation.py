"""
=============================================================================
TASK 3: DATA AUGMENTATION FOR MINORITY CLASSES
=============================================================================
WCE Gastrointestinal Disease Classification Project

STRATEGY:
  - Augment ONLY the 3 extreme minority classes in Kvasir-Capsule
  - Source: task2_output/kvasir_capsule_balanced/ (from Task 2)
  - Target counts:
      Polyp            55  →  500  (medically most critical)
      Blood - Hematin  12  →  300
      Ampulla of Vater 10  →  200
  - Augmentation techniques (all medically safe for GI images):
      Random horizontal flip
      Random vertical flip
      Random rotation (up to 30 degrees)
      Random brightness/contrast shift
      Random zoom (crop and resize)
      Random Gaussian noise
  - Augmented images saved alongside originals in balanced folder
  - Other classes left completely untouched

=============================================================================
"""

import os
import json
import random
import shutil
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
except ImportError:
    print("ERROR: Missing dependencies. Run:")
    print("  pip install Pillow numpy")
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Input: balanced dataset from Task 2
BALANCED_ROOT = r"C:\Users\devas\Desktop\DL LAB\Major_project\task2_output\kvasir_capsule_balanced"

# Output: augmented dataset (copy of balanced + new augmented images)
OUTPUT_ROOT   = r"C:\Users\devas\Desktop\DL LAB\Major_project\task3_output\kvasir_capsule_augmented"

# Report output
REPORT_DIR    = r"C:\Users\devas\Desktop\DL LAB\Major_project\task3_output"

RANDOM_SEED   = 42
IMG_EXTS      = {'.jpg', '.jpeg', '.png', '.bmp'}

# Target image count per minority class after augmentation
AUGMENTATION_TARGETS = {
    "Polyp":            500,   # 55  → 500  (most critical: cancer risk)
    "Blood - Hematin":  300,   # 12  → 300
    "Ampulla of Vater": 200,   # 10  → 200
}

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
# AUGMENTATION FUNCTIONS
# Each function takes a PIL Image and returns a new augmented PIL Image
# ─────────────────────────────────────────────────────────────────────────────

def aug_horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def aug_vertical_flip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def aug_rotate(img):
    angle = random.uniform(-30, 30)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)

def aug_brightness(img):
    factor = random.uniform(0.7, 1.4)
    return ImageEnhance.Brightness(img).enhance(factor)

def aug_contrast(img):
    factor = random.uniform(0.7, 1.4)
    return ImageEnhance.Contrast(img).enhance(factor)

def aug_zoom(img):
    """Random crop then resize back to original size."""
    w, h   = img.size
    factor = random.uniform(0.75, 0.95)
    new_w  = int(w * factor)
    new_h  = int(h * factor)
    left   = random.randint(0, w - new_w)
    top    = random.randint(0, h - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)

def aug_gaussian_noise(img):
    """Add subtle Gaussian noise."""
    arr  = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def aug_color_jitter(img):
    """Shift hue/saturation slightly."""
    factor = random.uniform(0.8, 1.2)
    return ImageEnhance.Color(img).enhance(factor)

def aug_sharpness(img):
    factor = random.uniform(0.5, 2.0)
    return ImageEnhance.Sharpness(img).enhance(factor)

# Pool of all augmentation functions
AUG_FUNCTIONS = [
    aug_horizontal_flip,
    aug_vertical_flip,
    aug_rotate,
    aug_brightness,
    aug_contrast,
    aug_zoom,
    aug_gaussian_noise,
    aug_color_jitter,
    aug_sharpness,
]

def apply_random_augmentations(img, n_transforms=2):
    """Apply n_transforms randomly chosen augmentations."""
    transforms = random.sample(AUG_FUNCTIONS, min(n_transforms, len(AUG_FUNCTIONS)))
    for t in transforms:
        img = t(img)
    return img

# ─────────────────────────────────────────────────────────────────────────────
# LOAD IMAGES FROM CLASS FOLDER
# ─────────────────────────────────────────────────────────────────────────────

def get_image_paths(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENT ONE CLASS
# ─────────────────────────────────────────────────────────────────────────────

def augment_class(class_name, src_folder, dst_folder, target_count):
    """
    Generate augmented images until we reach target_count total.
    Copies originals first, then generates synthetic images.
    """
    os.makedirs(dst_folder, exist_ok=True)
    original_paths = get_image_paths(src_folder)
    original_count = len(original_paths)

    if original_count == 0:
        cprint(f"  ERROR: No images found in {src_folder}", C.RED)
        return 0, 0

    # Copy all originals first
    for src in original_paths:
        dst = os.path.join(dst_folder, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    current_count = original_count
    needed        = target_count - current_count
    generated     = 0

    if needed <= 0:
        cprint(f"  {class_name}: already has {original_count} images (target={target_count})", C.YELLOW)
        return original_count, 0

    cprint(f"\n  Augmenting {class_name}: {original_count} → {target_count} (+{needed} synthetic)", C.CYAN)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    while generated < needed:
        # Pick a random source image (cycle through originals)
        src_path = original_paths[generated % original_count]

        try:
            img = Image.open(src_path).convert('RGB')

            # Apply 2-3 random augmentations
            n_aug = random.randint(2, 3)
            aug_img = apply_random_augmentations(img, n_aug)

            # Save with augmented filename
            base, ext = os.path.splitext(os.path.basename(src_path))
            aug_name  = f"{base}_aug_{generated:04d}{ext}"
            aug_path  = os.path.join(dst_folder, aug_name)
            aug_img.save(aug_path, quality=95)
            generated += 1

            if generated % 50 == 0:
                print(f"    Generated {generated}/{needed}...", end='\r')

        except Exception as e:
            cprint(f"  Warning: Could not process {src_path}: {e}", C.YELLOW)
            continue

    print(f"    Generated {generated}/{needed}... done!    ")
    return original_count, generated

# ─────────────────────────────────────────────────────────────────────────────
# COPY NON-MINORITY CLASSES UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

def copy_class_unchanged(class_name, src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    images = get_image_paths(src_folder)
    copied = 0
    for src in images:
        dst = os.path.join(dst_folder, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
    return len(images)

# ─────────────────────────────────────────────────────────────────────────────
# PRINT THEORY
# ─────────────────────────────────────────────────────────────────────────────

def print_theory():
    cprint("\n" + "─"*80, C.BLUE)
    cprint("  WHY AUGMENTATION?", C.BOLD + C.CYAN)
    cprint("─"*80, C.BLUE)
    points = [
        ("Problem",
         "After under-sampling, 3 classes remain critically small:\n"
         "    Polyp (55 images), Blood-Hematin (12), Ampulla of Vater (10).\n"
         "    Deep learning models cannot learn from so few examples."),
        ("Solution",
         "Generate synthetic images via geometric and photometric transforms.\n"
         "    Applied ONLY to minority classes — majority classes untouched."),
        ("Why these transforms are safe for medical images",
         "Flips/rotations: GI lesions can appear at any orientation in WCE.\n"
         "    Brightness/contrast: lighting varies across capsule positions.\n"
         "    Zoom: lesion size varies with camera distance.\n"
         "    Noise: WCE naturally has image noise from the capsule sensor."),
        ("What we DON'T do",
         "No colour inversion (would change tissue colour diagnostically).\n"
         "    No extreme distortion (would corrupt lesion morphology).\n"
         "    No SMOTE (works for tabular data, not medical images)."),
        ("Target counts",
         "Polyp: 55 → 500   (most critical — polyps cause colorectal cancer)\n"
         "    Blood-Hematin: 12 → 300\n"
         "    Ampulla of Vater: 10 → 200"),
    ]
    for title, desc in points:
        cprint(f"\n  ▶ {title}:", C.YELLOW + C.BOLD)
        print(f"    {desc}")

# ─────────────────────────────────────────────────────────────────────────────
# PRINT FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(results):
    print()
    cprint("─"*80, C.BLUE)
    cprint("  TASK 3 RESULTS — FINAL CLASS DISTRIBUTION", C.BOLD + C.CYAN)
    cprint("─"*80, C.BLUE)
    print()
    cprint(f"  {'Class':<30} {'Before':>7}  {'After':>7}  {'Synthetic':>10}  Status", C.BOLD)
    cprint("  " + "─"*72, C.WHITE)

    total_before = total_after = total_synthetic = 0
    max_after = max(v['after'] for v in results.values())

    for cls, v in sorted(results.items(), key=lambda x: -x[1]['after']):
        before    = v['before']
        after     = v['after']
        synthetic = v['synthetic']
        total_before    += before
        total_after     += after
        total_synthetic += synthetic

        if synthetic > 0:
            col    = C.MAGENTA
            status = f"{C.MAGENTA}★ AUGMENTED{C.RESET}"
        else:
            col    = C.GREEN
            status = f"{C.GREEN}✓ COPIED{C.RESET}"

        print(f"  {col}{cls:<30}{C.RESET} {before:>7,}  {after:>7,}  {synthetic:>10,}  {status}")

    cprint("  " + "─"*72, C.WHITE)
    cprint(f"  {'TOTAL':<30} {total_before:>7,}  {total_after:>7,}  {total_synthetic:>10,}", C.BOLD)
    print()

    after_counts  = {k: v['after'] for k, v in results.items()}
    after_ratio   = max(after_counts.values()) / min(after_counts.values())
    before_counts = {k: v['before'] for k, v in results.items()}
    before_ratio  = max(before_counts.values()) / min(before_counts.values())

    cprint(f"  Imbalance ratio : {before_ratio:>8,.0f}:1  →  {after_ratio:,.1f}:1", C.CYAN)
    cprint(f"  Total synthetic : {total_synthetic:,} new images generated", C.CYAN)
    cprint(f"  Dataset size    : {total_before:,} → {total_after:,} images", C.CYAN)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE JSON REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_report(results, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    report = {
        "task": "Task 3 - Augmentation",
        "augmentation_targets": AUGMENTATION_TARGETS,
        "random_seed": RANDOM_SEED,
        "per_class": results,
        "total_before":    sum(v['before']    for v in results.values()),
        "total_after":     sum(v['after']     for v in results.values()),
        "total_synthetic": sum(v['synthetic'] for v in results.values()),
    }
    path = os.path.join(report_dir, "task3_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    cprint(f"  📄 Report saved: {path}", C.GREEN)

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(results, report_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        cprint("  ⚠  matplotlib not found. Skipping plots.", C.YELLOW)
        return

    os.makedirs(report_dir, exist_ok=True)

    classes   = sorted(results.keys(), key=lambda c: -results[c]['after'])
    before    = [results[c]['before']    for c in classes]
    originals = [results[c]['before']    for c in classes]
    synthetic = [results[c]['synthetic'] for c in classes]
    after     = [results[c]['after']     for c in classes]

    x     = np.arange(len(classes))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')

    # Left: stacked bar — original + synthetic
    ax = axes[0]
    ax.bar(x, originals, width*1.8, label='Original',  color='#2ecc71', alpha=0.85)
    ax.bar(x, synthetic, width*1.8, label='Synthetic', color='#9b59b6', alpha=0.85,
           bottom=originals)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8, color='white')
    ax.set_ylabel('Number of Images', color='white')
    ax.set_title('Task 3: Original vs Augmented Images per Class', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.yaxis.grid(True, alpha=0.3, color='gray')
    ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')

    # Right: minority classes zoom
    ax2 = axes[1]
    minority_classes = list(AUGMENTATION_TARGETS.keys())
    m_before  = [results[c]['before']    for c in minority_classes]
    m_after   = [results[c]['after']     for c in minority_classes]
    m_synth   = [results[c]['synthetic'] for c in minority_classes]
    mx        = np.arange(len(minority_classes))

    ax2.bar(mx - 0.2, m_before, 0.35, label='Before Aug', color='#e74c3c', alpha=0.85)
    ax2.bar(mx + 0.2, m_after,  0.35, label='After Aug',  color='#2ecc71', alpha=0.85)
    for i, (b, a) in enumerate(zip(m_before, m_after)):
        ax2.text(i - 0.2, b + 2, str(b), ha='center', color='white', fontsize=10, fontweight='bold')
        ax2.text(i + 0.2, a + 2, str(a), ha='center', color='white', fontsize=10, fontweight='bold')
    ax2.set_xticks(mx)
    ax2.set_xticklabels(minority_classes, fontsize=10, color='white')
    ax2.set_ylabel('Number of Images', color='white')
    ax2.set_title('Minority Classes: Before vs After Augmentation', color='white', fontsize=12)
    ax2.tick_params(colors='white')
    ax2.spines[:].set_color('#444')
    ax2.yaxis.grid(True, alpha=0.3, color='gray')
    ax2.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
    ax2.set_facecolor('#16213e')

    plt.tight_layout(pad=3)
    p = os.path.join(report_dir, "plot7_augmentation_results.png")
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    cprint(f"  🖼  Saved: {p}", C.GREEN)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cprint("\n" + "="*80, C.CYAN)
    cprint("  TASK 3: AUGMENTATION FOR MINORITY CLASSES", C.BOLD)
    cprint("  SVNIT Surat | Prof. Praveen Kumar Chandaliya", C.WHITE)
    cprint("="*80, C.CYAN)

    if not os.path.exists(BALANCED_ROOT):
        cprint(f"\n  ERROR: Task 2 output not found at:\n  {BALANCED_ROOT}", C.RED)
        cprint("  Run task2_undersampling.py first.", C.YELLOW)
        return

    print_theory()

    # Get all classes from balanced dataset
    all_classes = sorted([
        d for d in os.listdir(BALANCED_ROOT)
        if os.path.isdir(os.path.join(BALANCED_ROOT, d))
    ])

    cprint(f"\n\n  Found {len(all_classes)} classes in balanced dataset.", C.CYAN)
    cprint(f"  Minority classes to augment: {list(AUGMENTATION_TARGETS.keys())}", C.MAGENTA)
    cprint(f"  Other classes: copied unchanged", C.GREEN)

    print()
    cprint("─"*80, C.BLUE)
    cprint("  PROCESSING CLASSES", C.BOLD + C.CYAN)
    cprint("─"*80, C.BLUE)

    results = {}

    for cls in all_classes:
        src_folder = os.path.join(BALANCED_ROOT, cls)
        dst_folder = os.path.join(OUTPUT_ROOT, cls)
        before     = len(get_image_paths(src_folder))

        if cls in AUGMENTATION_TARGETS:
            target = AUGMENTATION_TARGETS[cls]
            orig, synthetic = augment_class(cls, src_folder, dst_folder, target)
            after = orig + synthetic
        else:
            cprint(f"  Copying {cls} ({before:,} images)...", C.WHITE)
            after     = copy_class_unchanged(cls, src_folder, dst_folder)
            synthetic = 0

        results[cls] = {
            "before":    before,
            "after":     after,
            "synthetic": synthetic,
            "target":    AUGMENTATION_TARGETS.get(cls, before)
        }

    print_final_report(results)

    cprint("\n  Saving outputs...", C.CYAN)
    save_report(results, REPORT_DIR)
    generate_plots(results, REPORT_DIR)

    cprint("\n" + "="*80, C.CYAN)
    cprint("  TASK 3 COMPLETE ✅", C.GREEN + C.BOLD)
    cprint(f"  Augmented dataset: {OUTPUT_ROOT}", C.WHITE)
    cprint(f"  Report: {REPORT_DIR}", C.WHITE)
    cprint("\n  Next → Task 4: Preprocessing", C.YELLOW)
    cprint("  (Resize all images to 224x224, normalize, train/val/test split)", C.YELLOW)
    cprint("="*80, C.CYAN)

if __name__ == "__main__":
    main()
