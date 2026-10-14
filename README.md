# Trash Detection with RF-DETR

Object detection on household-waste images using **RF-DETR** (Roboflow's real-time detection transformer), fine-tuned from COCO-pretrained weights. Final project for the Methods of Research / deep learning course.

The model detects and classifies one item per image into six classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

## Repository contents

| File | Purpose |
|---|---|
| `01_dataset_inspection.ipynb` | Downloads the dataset and checks it: split sizes, class balance, box sizes, label errors, duplicate/leakage check, sample images with boxes |
| `02_dataset_cleaning.ipynb` | Fixes the issues found in step 01 and writes the cleaned dataset |
| `03_model_and_training.ipynb` | Model architecture overview, from-scratch Hungarian matcher and loss, augmentation, three training experiments, early stopping, checkpointing, training curves |
| `04_evaluation.ipynb` | Test-set evaluation (accuracy, precision, recall, IoU, mAP), confusion matrix, prediction visualizations, error analysis |
| `requirements.txt` | Python dependencies |

## Setup

The notebooks are written for **Google Colab with a GPU runtime** (Runtime > Change runtime type > T4 GPU). Each notebook installs what it needs in its first cell.

1. Get a Roboflow API key from your Roboflow account settings.
2. In Colab, open the key icon in the left sidebar (Secrets), add a secret named `ROBOFLOW_API_KEY`, paste the key, and turn on notebook access. **Never paste the key into a notebook or commit it to GitHub.**
3. Checkpoints and results are saved to `MyDrive/rf-detr-trash` on Google Drive, so they survive a Colab runtime reset.

To run locally instead, use Python 3.10 or newer with a CUDA GPU and `pip install -r requirements.txt`. The notebooks use Colab-only features (`google.colab.drive` and `userdata`), which you would replace with local paths and an environment variable.

## How to run

Run the notebooks in order:

1. **`01_dataset_inspection.ipynb`**: review the tables and image grids.
2. **`02_dataset_cleaning.ipynb`**: creates `/content/trash-clean`.
3. **`03_model_and_training.ipynb`**: trains the experiments. Each experiment is its own cell, and an interrupted run resumes when the cell is run again. At the end it prints which run did best on the **validation** set.
4. **`04_evaluation.ipynb`**: set `RUN_NAME` to the best run and run all cells. Figures and a `test_results.json` file are saved to Drive.

## Dataset

| | |
|---|---|
| Source | Roboflow Universe, project `trash-detection-kcsnu-1kabv` (workspace `princesss-workspace-2dq44`), version 1 |
| Format | COCO bounding boxes |
| Size | 2,524 images of 512x384 pixels (about 1 object per image) |
| Split | 70 / 15 / 15 (train 1,763, valid 379, test 377 after cleaning) |
| Classes | cardboard, glass, metal, paper, plastic, trash |

Check the dataset's page on Roboflow Universe for its license and original source, and cite it in the report.

**Cleaning applied in notebook 02**

- Removed an unused placeholder category from the export and renumbered the classes to ids 0 to 5. Without this, the class count and label mapping can be off by one.
- Removed the 5 images that had no annotations.
- Recorded the dataset version, download date, and final counts in `dataset_info.json`.

**Checks that passed in notebook 01:** no overlap between splits (by file hash and by source image name), no zero-size or out-of-bounds boxes, no duplicate boxes.

## Model and training

- **Model:** `RFDETRNano` from the [`rfdetr`](https://github.com/roboflow/rf-detr) library (384 px input, 2 decoder layers, about 30.5 million parameters), fine-tuned from COCO-pretrained weights. A DINOv2 vision-transformer backbone with windowed attention, a multi-scale projector, and a deformable-attention decoder with two-stage query selection. Class and box prediction heads are a linear layer and a small MLP. A first attempt with the larger Small model for 20 epochs was stopped because it was too slow, and the project switched to Nano with 10 epochs.
- **Training:** AdamW, separate learning rates for backbone and decoder, gradient clipping, exponential moving average of weights, mixed precision. Loss is IoU-aware classification loss plus L1 and GIoU box losses, with Hungarian matching of predictions to ground truth.
- **Preprocessing:** resize to the model's square input size (384 x 384 for Nano), normalize with ImageNet mean and standard deviation.
- **Experiments:**

| Run | Augmentation | LR schedule |
|---|---|---|
| `exp_a_baseline_noaug` | none | step decay |
| `exp_b_augmentation` | flips, rotation up to 20 degrees, brightness/contrast | step decay |
| `exp_c_aug_cosine` | same as above | cosine with 1 warm-up epoch |

- **Settings for all runs:** 10 epochs, batch size 8 with gradient accumulation 2 (effective 16), learning rate 1e-4 (decoder) and 1.5e-4 (backbone), seed 42, on a Colab T4 GPU.
- **Early stopping** on validation mAP@0.5:0.95 (patience 3), **checkpointing** every 5 epochs plus the best weights, and validation loss logged each epoch. Early stopping did not shorten any run.
- `03_model_and_training.ipynb` also contains a small from-scratch Hungarian matcher and loss, verified to give the same assignments as the library's matcher.

## Evaluation

Metrics are computed on the **test split** with code written for this project, and mAP with `pycocotools`:

- A prediction is matched to a ground-truth box one-to-one with the Hungarian algorithm on IoU, and counts only if IoU is at least 0.5.
- **TP:** matched and correct class. **FP:** unmatched prediction, or matched with the wrong class. **FN:** unmatched ground-truth box, or matched with the wrong class.
- **Precision** = TP / (TP + FP). **Recall** = TP / (TP + FN).
- **Accuracy** is the share of matched boxes that have the correct class.
- **mAP@0.5** and **mAP@0.5:0.95** are computed over all confidence levels.
- The confidence threshold for precision and recall is chosen on the validation set (best F1) and then applied unchanged to the test set.

### Validation results (training experiments)

| Run | Val mAP@0.5:0.95 | Val mAP@0.5 | Best epoch | Training time |
|---|---|---|---|---|
| A: no augmentation | 0.9218 | 0.9457 | 10 | 17.6 min |
| B: augmentation | 0.9196 | 0.9505 | 10 | 23.0 min |
| C: augmentation + cosine schedule | 0.9168 | 0.9445 | 7 | 24.5 min |

The three runs are effectively tied (a spread of about 0.005 mAP, one run each). Neither augmentation nor the cosine schedule gave a measurable improvement. Run A was selected as the final model because it has the highest validation mAP@0.5:0.95, with a margin that is within noise.

### Test results

Final model: run A (`exp_a_baseline_noaug`), evaluated once on the test split (377 images, 382 boxes). The confidence threshold 0.70 was chosen on the validation split. Precision, recall, F1, accuracy and mean IoU use IoU 0.5.

| Metric | Value |
|---|---|
| Accuracy (after matching) | 0.956 |
| Precision | 0.956 |
| Recall | 0.911 |
| F1 | 0.933 |
| Mean IoU of matched boxes | 0.971 |
| mAP@0.5 | 0.953 |
| mAP@0.5:0.95 | 0.924 |

The library's own evaluation gives mAP@0.5 0.952 and mAP@0.5:0.95 0.922, which agrees with our pycocotools numbers. Per-class AP@0.5 ranges from 0.985 (glass) to 0.852 (trash, only 15 test boxes). Of the 34 missed ground-truth boxes, 18 were not detected and 16 were detected with the wrong class (mostly glass, metal and trash confusions). CPU inference took about 660 ms per image.

## Limitations

- Every image is a single object on a plain background at 512x384 pixels, so results will not transfer directly to cluttered real-world scenes.
- The `trash` class is a catch-all with few samples (15 test boxes), so its per-class numbers are noisy.
- Results come from a single training run per experiment with one seed, so small differences between runs may not be meaningful.
- Each run was limited to 10 epochs and used the smaller Nano model to save time. Runs A and B were still at their best score at the final epoch, so longer training might improve results slightly.

## References

1. Robinson, I., et al. (2025). *RF-DETR: Neural Architecture Search for Real-Time Detection Transformers.* arXiv:2511.09554. https://arxiv.org/abs/2511.09554
2. Roboflow. *rf-detr* (code and documentation). https://github.com/roboflow/rf-detr
3. Carion, N., et al. (2020). *End-to-End Object Detection with Transformers (DETR).* ECCV.
4. Zhu, X., et al. (2021). *Deformable DETR: Deformable Transformers for End-to-End Object Detection.* ICLR.
5. Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.*
6. Lin, T.-Y., et al. (2014). *Microsoft COCO: Common Objects in Context.* ECCV. (evaluation protocol, via `pycocotools`)
7. Roboflow Universe dataset: Trash Detection (`trash-detection-kcsnu-1kabv`), version 1. Add the author, URL, and access date from the dataset page.
