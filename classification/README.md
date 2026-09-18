# AID x4 classification evaluation

This code reproduces the x4 classification comparison using a **fixed
classifier**. One checkpoint is loaded once and evaluated on LR x4, MAG-ViT x4,
and FunSR x4 images. Do not retrain the classifier separately for the three
conditions.

## Classification checkpoints

- [MaxViT-T checkpoint](https://drive.google.com/file/d/1SUH3LO9TKbPZmxe3149lD90gRIRk3WMH/view?usp=sharing)
  - Filename: `maxvit_t_aid_lr_x2.pth`

- [ResNeXt-101 checkpoint](https://drive.google.com/file/d/1B1c7NzaRemEO-3JtGFlEGEVmnElj_7YX/view?usp=sharing)
  - Filename: `resnext101_32x8d_aid_lr_x2.pth`
 
## AID classification test images

Download the prepared AID test images:

[AID test images on Google Drive](https://drive.google.com/drive/folders/1ROlATzYSOmPfZkUG3XqTv5OmCBjIZK_q?usp=sharing)

After downloading and extracting, use:

- `LR_x4` for LR ×4 classification.
- `out_x4` for MAG-ViT ×4 classification.
--

The name `LR_x2` describes the data used to train the classifier. The same
checkpoint can be evaluated on all x4 image conditions. This keeps the
classifier fixed and makes the image-quality comparison fair.

## Expected dataset format

Each input may use either of these layouts:

```text
flat_folder/Airport_1.png
flat_folder/airport_2.png
```

or:

```text
root/Airport/image_1.png
root/BareLand/image_2.png
```

The script fails on unrecognized labels instead of silently excluding images.
By default, it also requires exactly 2,000 images in every condition.

## Run MaxViT-T

```bash
bash classification/run_x4_evaluation.sh \
  maxvit_t \
  /data/Image_restoration/magvitpaper/classification/models/Maxvit_class/maxvit_original_LR_x2/maxvit_t_best.pth \
  /absolute/path/to/_LR_x4 \
  /absolute/path/to/_MAGViT_x4 \
  /data/Image_restoration/Datasets/RS_AID_data/AID-dataset/test/out_x4_funsr_full \
  results/table_viii_x4/maxvit \
  cuda:4
```

## Run ResNeXt-101

```bash
bash classification/run_x4_evaluation.sh \
  resnext101_32x8d \
  /data/Image_restoration/magvitpaper/classification/models/Resnet_x/Resnet_x_original_LR_x2/resnext101_32x8d_best.pth \
  /absolute/path/to/AID_LR_x4 \
  /absolute/path/to/AID_MAGViT_x4 \
  /data/Image_restoration/Datasets/RS_AID_data/AID-dataset/test/out_x4_funsr_full \
  results/table_viii_x4/resnext101 \
  cuda:4
```

Change the two placeholder dataset paths to their real locations. The FunSR x4
path above comes from the original ResNeXt evaluation script.

## Outputs

For each classifier, the output directory contains:

- `table_viii_x4_results.csv`: aggregate accuracy for all three conditions;
- `run_metadata.json`: checkpoint SHA-256, classes, device, and preprocessing;
- one directory per condition containing per-image predictions, a confusion
  matrix, a classification report, and a JSON summary.

For a 2,000-image test set, the paper's reported x4 results correspond to:

| Classifier | LR x4 | MAG-ViT x4 | FunSR x4 |
|---|---:|---:|---:|
| MaxViT-T | 1910/2000 (95.50%) | 1919/2000 (95.95%) | 1877/2000 (93.85%) |
| ResNeXt-101 | 1835/2000 (91.75%) | 1864/2000 (93.20%) | 1832/2000 (91.60%) |

These values should only be claimed when they are reproduced by the saved CSV
outputs using the published test split.

## Files to publish

Commit this `classification/` directory and the generated aggregate result CSVs
to GitHub. Store the large `.pth` checkpoint files in a GitHub Release, Zenodo,
or another persistent model host. Add their download URLs and SHA-256 hashes to
the repository README. Do not commit large checkpoints directly to Git history.

The repository needs **two** classification checkpoints to reproduce both
classifier columns in Table VIII: one MaxViT-T checkpoint and one ResNeXt-101
checkpoint. It does not need separate classifier checkpoints for LR x4,
MAG-ViT x4, and FunSR x4.
