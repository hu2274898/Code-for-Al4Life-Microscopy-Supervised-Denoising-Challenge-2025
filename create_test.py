import random
from pathlib import Path
import shutil

# 路径
img_dir = Path('./noisy')
gt_dir  = Path('./gt')
test_img_dir = Path('./noisy_test')
test_gt_dir  = Path('./gt_test')

# 创建测试集文件夹
test_img_dir.mkdir(exist_ok=True)
test_gt_dir.mkdir(exist_ok=True)

# 获取所有图片
img_files = sorted(img_dir.glob('*.tif'))

# 设置测试集比例
test_ratio = 0.2  # 20%做测试集
test_count = int(len(img_files) * test_ratio)

# 随机选择测试集
test_imgs = random.sample(img_files, test_count)

for img_file in test_imgs:
    # 对应 gt 文件
    gt_file = gt_dir / img_file.name
    
    # 移动图片和 gt
    shutil.move(str(img_file), str(test_img_dir / img_file.name))
    shutil.move(str(gt_file), str(test_gt_dir / gt_file.name))

print(f'Moved {test_count} images to test set.')
