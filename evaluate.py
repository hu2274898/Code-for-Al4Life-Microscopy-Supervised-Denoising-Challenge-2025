from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import tifffile
import numpy as np
from pathlib import Path

# 设置 ground truth 和预测文件夹
gt_dir = Path("/data/birth/lmx/work/Class_projects/hxt/work/AI4Life-MDC25-example-submission-main/gt_test")
pred_dir = Path("/data/birth/lmx/work/Class_projects/hxt/work/AI4Life-MDC25-example-submission-main/output/images/image-stack-denoised")

all_psnr = []
all_ssim = []

# 遍历所有 gt 文件
for gt_file in sorted(gt_dir.glob("*.tif")):
    pred_file = pred_dir / gt_file.name
    if not pred_file.exists():
        print(f"Warning: {pred_file} not found, skipped.")
        continue

    gt = tifffile.imread(gt_file).astype(np.float32)
    pred = tifffile.imread(pred_file).astype(np.float32)

    psnr_list = []
    ssim_list = []

    # 遍历 D 维的每一层
    for i in range(gt.shape[0]):
        psnr_val = psnr(gt[i], pred[i], data_range=gt[i].max() - gt[i].min())
        ssim_val = ssim(gt[i], pred[i], data_range=gt[i].max() - gt[i].min())
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    all_psnr.append(np.mean(psnr_list))
    all_ssim.append(np.mean(ssim_list))

print(f"Average PSNR across all files: {np.mean(all_psnr):.4f}")
print(f"Average SSIM across all files: {np.mean(all_ssim):.4f}")
