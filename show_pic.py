from pathlib import Path
import tifffile 
import numpy as np
import matplotlib.pyplot as plt
image_path = '/data/birth/lmx/work/Class_projects/hxt/work/AI4Life-MDC25-example-submission-main/test/input/interface_0/images/image-stack-unstructured-noise/00001.tif'
save_path = '/data/birth/lmx/work/Class_projects/hxt/work/AI4Life-MDC25-example-submission-main'

def save_image(image_path, save_path, slice_index=2):
    image_path = Path(image_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    img = tifffile.imread(image_path)
    slice_img = img[slice_index]  # 选取某一层

    plt.figure(figsize=(6,6))
    plt.imshow(slice_img)
    plt.axis('off')
    plt.savefig(save_path / f'slice_{slice_index}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved slice_{slice_index}.png at {save_path}")


def read_image(image_path: Path) -> np.ndarray:
    """
    Read and preprocess input image.
    Modify this function to implement your own image loading and preprocessing pipeline.
    """
    print(f"Reading image: {image_path}")
    input_array = tifffile.imread(image_path)
    input_array = input_array.astype(np.float32)
    print(f"Loaded image shape: {input_array.shape}")
    return input_array

read_image(image_path)
save_image(image_path,save_path)