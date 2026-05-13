# MDC25 — Supervised Microscopy Denoising

Submission code for the [AI4Life-MDC25 Supervised Denoising Challenge](https://ai4life-mdc25.grand-challenge.org/).

A Restormer-based model for denoising microscopy TIFF images, packaged as a Grand Challenge Docker container.

## Project layout

```
.
├── mdc_model.py          # Restormer model definition
├── train.py              # Training loop (MSE loss + checkpointing)
├── inference.py          # Container entrypoint: denoises images in INPUT_PATH
├── create_model.py       # JIT-package a model for distribution
├── create_test.py        # Sample a small subset of training data for quick tests
├── evaluate.py           # PSNR / SSIM evaluation against ground truth
├── show_pic.py           # Visualize noisy / denoised / GT triplets
├── requirements.txt
├── Dockerfile            # Submission image (pytorch/pytorch base, linux/amd64)
├── do_build.sh           # Build the submission container
├── do_test_run.sh        # Build + run a forward pass on test/input
├── do_save.sh            # Save container as gzip tarball for upload
```

Datasets (`noisy_train/`, `gt_train/`, `gt_test/`) and trained checkpoints (`*.pth`) are not included in the repo — see the challenge page for downloads.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+ and PyTorch with CUDA.

## Training

Place the training pairs under `./noisy_train/` (noisy `.tif`) and `./gt_train/` (clean `.tif`), then:

```bash
python train.py
```

Checkpoints are written to `./ckpt2/ckpt_epoch{N}.pth`. Note: `train.py` currently has an absolute `resume_path` near the top — edit it (or set to `None` and remove the resume block) to start from scratch.

## Evaluation

```bash
python evaluate.py   # edit the paths at the bottom of the script for your run
```

Reports per-image PSNR and SSIM against the ground-truth folder.

## Inference (local)

```bash
python inference.py   # reads from INPUT_PATH, writes to OUTPUT_PATH (see env vars in script)
```

## Docker submission

```bash
./do_build.sh        # build the linux/amd64 image
./do_test_run.sh     # smoke test: runs container on test/input/interface_0
./do_save.sh         # produce a gzip tarball ready to upload to Grand Challenge
```

The container expects model weights to be available either inside the image (under `resources/`) or mounted at `/opt/ml/model` (handled by `do_test_run.sh`).

## Citation / acknowledgement

The model architecture follows [Restormer](https://github.com/swz30/Restormer) (Zamir et al., CVPR 2022).
