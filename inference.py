"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""
from os import device_encoding
from pathlib import Path
import json
import tifffile 
import numpy as np
import torch
from torch import nn
from typing import List
import torch
import torch.nn.functional as F

class SafeResRestormer(nn.Module):
    def __init__(self, model, downsample_times=3, factor=2):
        super().__init__()
        self.model = model
        self.downsample_times = downsample_times
        self.factor = factor
        self.total_factor = factor ** downsample_times  # 总下采样倍数

    def forward(self, x):
        # 记录原始的图片尺寸
        orig_shape = x.shape  # [B, H, W] 或 [B*D, H, W]
        H, W = orig_shape[-2], orig_shape[-1]

        # 计算 pad 后的尺寸，使 H, W 能被 total_factor 整除
        pad_H = (self.total_factor - H % self.total_factor) % self.total_factor
        pad_W = (self.total_factor - W % self.total_factor) % self.total_factor

        # pad 在右和下方
        x = F.pad(x, (0, pad_W, 0, pad_H), mode='reflect')

        # 调用原模型 forward
        out = self.model(x)

        # 裁回原始尺寸
        out = out[..., :H, :W]
        return out
from torch import nn
from typing import List
import torch
import numbers
from einops import rearrange
import torch.nn.functional as F
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
class Block(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Block, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
# class BottleneckBlock(nn.Module):
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
#         super(BottleneckBlock, self).__init__()

#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention(dim, num_heads, bias)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))

#         return x
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, stride=stride)

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super(TransformerBlock, self).__init__()

        self.cov = nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1,bias=True)
        self.act = nn.GELU()
    def forward(self, x):
        res = x
        res = self.cov(res)
        res = self.act(res)
        res = self.cov(res)

        return x+res

class restormer(nn.Module):
    def __init__(self,
                in_chans: int,
                out_chans: int,
                n_feat0: int,
                kernel_size=3,
                bias=False,
                 num_blocks=[1,1,1,2]
                 ):
        super().__init__()
        # Feature extraction
        self.feat_extract = conv(in_chans, n_feat0, kernel_size, bias=bias)
        dim = n_feat0
        head = 1
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'
        #self.fre1 = FreModule(dim * 2 ** 3, num_heads=head, bias=bias)
        #self.blocks_down = []
        # for i in range(1):
        #     self.blocks_down.append(ResnetBlock(dim * 2 ** 2))
        # self.body_down = nn.Sequential(*self.blocks_down)
        # for i in range(1):
        #     self.blocks_up.append(ResnetBlock(dim * 2 ** 2))
        # self.body_up = nn.Sequential(*self.blocks_down)
        # Encoder - 3 DownBlocks
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1)) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        # Bottleneck
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2)) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to
        self.latent = nn.Sequential(*[Block(dim=int(dim * 2 ** 3),num_heads=1,ffn_expansion_factor=ffn_expansion_factor,bias=bias,LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        #self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 3)) for i in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2)) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1)) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1)) for i in range(num_blocks[0])])

        # OutConv
        self.conv_last = conv(n_feat0 * 2, out_chans, 5, bias=bias)

    def forward(self, x):
        inp_enc_level1 = self.feat_extract(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)


        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)


        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return self.conv_last(out_dec_level1)+x
# Constants for the location of the input and output files, please do not modify! 
INPUT_PATH = Path("/input/images/image-stack-unstructured-noise")
OUTPUT_PATH = Path("/output/images/image-stack-denoised")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================================
# USER-CUSTOMIZABLE FUNCTIONS
# Modify these functions to implement your own inference pipeline
# ============================================================================

def load_model():
    """
    Load your model. You have two options for the model path:
    1. Save model as a part of the Docker-container image in the resources/ directory.
        It will be available at the /opt/app/resources directory at runtime.
    2. Upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). 
        The resources in the tarball will be extracted to /opt/ml/model directory at runtime.
    """
    # Option 1: part of the Docker-container image: resources/
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = Path("/opt/app/resources/ckpt_epoch53.pth")

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). 
    # The resources in the tarball will be extracted to `model_dir` at runtime.
    # Example: model_path = Path("/opt/ml/model/my_model.pth")
    model = restormer(in_chans=1, out_chans=1, n_feat0=6)
    checkpoint = torch.load(model_path,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    safe_model = SafeResRestormer(model, downsample_times=3, factor=2)
    safe_model.to(device)  # 确保 wrapper 在同一个 device
    safe_model.eval()      # wrapper eval
    print(f"Loading model: {model_path}")
    return safe_model

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


def run_inference(model, input_tensor):
    """
    Run inference on the input tensor.
    Modify this function to implement your own inference logic.
    """
    print("Running inference...")
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(1).float().to(next(model.parameters()).device)
    print(f"Input shape: {input_tensor.shape}")
    with torch.no_grad():
        output = model(input_tensor).squeeze().cpu().numpy()
    print(f"Output shape: {output.shape}")
    return output


def save_output(array, output_path):
    """
    Save the processed array.
    """
    print(f"Saving output to: {output_path}")
    with tifffile.TiffWriter(output_path) as out:
        out.write(
            array,
            resolutionunit=2 # This flag is important for the GC to process the output correctly! 
        )


def inference_handler():
    """
    Main handler for processing images with unstructured noise.
    This is where you should implement your main inference pipeline.
    """
    # Show torch cuda info
    _show_torch_cuda_info()

    # Load your model
    model = load_model()

    # Load and prepare input
    input_files = sorted(INPUT_PATH.glob("*.tif"))
    print(f"Reading input files: {input_files}")

    for input_file in input_files:
        input_array = read_image(input_file)
        input_array = input_array
        # Run inference
        result = run_inference(model, input_array)
        # Save output
        output_path = OUTPUT_PATH / input_file.name
        save_output(result, output_path)

    return 0


# ============================================================================
# UTILITY FUNCTIONS
# These functions handle the interface with the Grand Challenge platform
# ============================================================================

def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    print(f"Interface key: {interface_key}")

    handler = inference_handler

    # Call the handler
    return handler()


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(INPUT_PATH.parent.parent / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
