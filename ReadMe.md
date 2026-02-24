# WCE Gastrointestinal Disease Classification

Deep learning-based classification of GI diseases across 5 datasets
using Knowledge Distillation (EfficientNetV2 teacher → student model).

## Datasets
| Dataset | Classes | Images | Modality |
|---|---|---|---|
| Kvasir-Capsule | 14 | 47,238 | WCE |
| KVASIR v2 | 8 | 8,000 | Colonoscopy |
| CVC-ClinicDB | 1 | 612 | Colonoscopy |
| ETIS-Larib | 1 | 196 | Colonoscopy |
| KID | 4 | 370 | WCE |

## Progress
- [x] Task 1: Dataset exploration and imbalance analysis
- [x] Task 2: Under-sampling (47K → 14.7K, imbalance 3434:1 → 300:1)
- [x] Task 3: Augmentation for minority classes (300:1 → 18.9:1)
- [ ] Task 4: Preprocessing (resize 224×224, train/val/test split)
- [ ] Task 5: EfficientNetV2 teacher model
- [ ] Task 6: Learning rate scheduling
- [ ] Task 7: Training + Knowledge Distillation + Evaluation