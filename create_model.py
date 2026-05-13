"""
This script creates a simple example torch model and saves it as a jit model.
The model will be saved in the resources/ directory.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur


class SimpleModel(nn.Module):
    """A simple example torch model containing only a gaussian blur"""

    def __init__(self):
        super().__init__()
        self.transform = GaussianBlur(kernel_size=3, sigma=(1.0, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Check if input is 3D
            if len(x.shape) == 3:  # (D, H, W)
                # Process each slice separately
                output = []
                for slice_idx in range(x.shape[0]):
                    slice_2d = x[slice_idx:slice_idx+1]  # Add channel dim
                    processed_slice = self.transform(slice_2d)
                    output.append(processed_slice.squeeze(0))  # Remove channel dim
                return torch.stack(output)
            else:  # 2D input
                return self.transform(x.unsqueeze(0)).squeeze(0)


def create_model(model_path: Path):
    """Create and save an example jit model"""
    model = SimpleModel()
    
    jit_model = torch.jit.script(model) 
    
    print(f'Saving model to: {model_path.absolute()}')
    torch.jit.save(jit_model, model_path)


if __name__ == "__main__":
    model_path = Path(__file__).parent / "resources/my_model.pth"
    create_model(model_path)
