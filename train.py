import torch
import random
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import tifffile
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from mdc_model import restormer
class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        
        if reduced:
            return 1 - S.mean()
        else:
            return 1 - S
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class tiffdata(Dataset):
    def __init__(self,img_dir,gt_dir):
        self.img_dir = Path(img_dir)
        self.gt_dir = Path(gt_dir)
        self.img = sorted(self.img_dir.glob('*.tif'))
        self.gt = sorted(self.gt_dir.glob('*.tif'))
        assert len(self.img) == len(self.gt)
    def __len__(self):
        return len(self.img)
    def __getitem__(self,idx):
        img = tifffile.imread(self.img[idx]).astype(np.float32)
        gt = tifffile.imread(self.gt[idx]).astype(np.float32)
        return img, gt
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
def train():
    setup_seed(42)
    resume_path = '/data/birth/lmx/work/Class_projects/hxt/work/mdc25/ckpt2/ckpt_epoch65.pth'
    img_dir = './noisy_train'
    gt_dir = './gt_train'
    train_data = tiffdata(img_dir,gt_dir)
    train_load = DataLoader(train_data,batch_size=31, shuffle=True, num_workers=6)
    print('data has been loaded')
    model = restormer(in_chans=1, out_chans=1, n_feat0=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.00005)
    log_file = open('./log.txt','w')
    best_val = 100
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val = float(checkpoint['loss'])
   # 继续用之前的最优loss
    print(f"Resumed from {resume_path}, start_epoch={start_epoch}, best_val={best_val}")
    for epoch in tqdm(range(start_epoch,120)):
        print('epoch:',epoch)
        loss_total = 0
        model.train()
        for i,(img,gt) in enumerate(tqdm(train_load, desc=f'Epoch {epoch}', leave=False)):
            # img = np.expand_dims(img, axis=1)  # [C, H, W] or [C, D, H, W]
            # gt  = np.expand_dims(gt, axis=1)
            B, D, H, W = img.shape
            img = img.reshape(B*D, H, W)
            gt  = gt.reshape(B*D, H, W)
            img = img.unsqueeze(1).float().to(device)
            gt  = gt.unsqueeze(1).float().to(device)
            #[4,16,64,64]

            output = model(img)
            loss = F.mse_loss(output,gt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_total += loss.item()
            
        avg_loss = loss_total / len(train_load)
        valid_loss = "{:.5f}".format(avg_loss)
        print(valid_loss)
        log_file.write(valid_loss + '\n')
        ckpt_path = f'./ckpt2/ckpt_epoch{epoch}.pth'
        if(avg_loss < best_val):
            best_val = avg_loss
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            },ckpt_path)
    return 'finish'
            


if __name__ == '__main__':
    train()