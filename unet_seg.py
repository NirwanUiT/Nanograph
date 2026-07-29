"""
Lightweight U-Net foreground segmenter for Nanograph.

Trained on independent fluorescence-organelle data (NMI org/seg) and applied
zero-shot to other modalities (e.g. Aaron mitochondria) as a learned foreground
prior that recovers dim/thin structure the classical Otsu/Frangi cascade misses.

Fully convolutional: handles arbitrary input sizes (padded to a multiple of 16
at inference). Kept dependency-light (pure torch) so it imports inside the
encoder pipeline without extra packages.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4-level U-Net, single-channel in, single-logit out."""
    def __init__(self, base=32):
        super().__init__()
        self.e1 = _block(1, base)
        self.e2 = _block(base, base * 2)
        self.e3 = _block(base * 2, base * 4)
        self.e4 = _block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bott = _block(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = _block(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = _block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bott(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def _pad_to_mult(x, m=16):
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    return F.pad(x, (0, pw, 0, ph)), (h, w)


@torch.no_grad()
def unet_predict(model, image_gray, device='cpu', thresh=0.5, return_prob=False):
    """Run the U-Net on a uint8 grayscale image; return a uint8 {0,255} mask.

    Normalises per-image to [0,1] with a robust 1-99 percentile stretch so it is
    insensitive to absolute intensity scale (key for cross-dataset transfer).
    """
    model.eval()
    img = image_gray.astype(np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    t = torch.from_numpy(img)[None, None].to(device)
    t, (h, w) = _pad_to_mult(t)
    prob = torch.sigmoid(model(t))[0, 0, :h, :w].cpu().numpy()
    if return_prob:
        return prob
    return (prob >= thresh).astype(np.uint8) * 255


def load_unet(ckpt_path, device='cpu', base=32):
    model = UNet(base=base).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    model.eval()
    return model
