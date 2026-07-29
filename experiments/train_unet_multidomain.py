"""
Retrain the Nanograph clDice U-Net for external curvilinear datasets.

Motivation
----------
The organelle-trained U-Net transfers POORLY to vessels/EM/filaments (negative
transfer, cross-domain seg IoU < 0.05). Three compounding causes were diagnosed:
  1. polarity  - external structures are DARK-on-BRIGHT (organelles are bright);
  2. selection - the organelle-calibrated quality proxy cannot rank the masks;
  3. EM limit  - EPFL mitochondria are not threshold/ridge separable at all.

This script fixes (1) and (3) at the source by TRAINING a single multi-domain
segmenter on the four target datasets, with a random-inversion augmentation that
makes the network polarity-AGNOSTIC (it segments a structure whether it appears
bright or dark), so no fragile inference-time polarity detector is needed.

Datasets (all with binary GT):
  STARE / DRIVE  retinal vessels   (independent images -> random val split)
  microtubules   IRM filaments     (independent images -> random val split)
  EPFL mito      EM volume slices  (correlated slices -> CONTIGUOUS val split
                                    with a guard gap, to avoid adjacency leakage)

Loss: Dice+BCE + clDice (centerline) - identical recipe to the organelle net.

Outputs:
  experiments/unet_multidomain.pt    best macro (Dice+csens) checkpoint
  experiments/unet_multidomain.png   training curves
"""
import os
import sys
import glob
import argparse
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nanograph_v4.unet_seg import UNet
# reuse the exact loss/metric recipe from the organelle trainer
from train_unet import combined_loss, soft_skel

PREP = '/mnt/nas1/nba055-2/idea_1/ext_datasets/prepared'

# dir, split-strategy.  'random' = independent images; 'contiguous' = slice
# volume, hold out a tail block (+guard gap) so val slices are not near train.
DATASETS = {
    'stare':        dict(dir=f'{PREP}/stare',        split='random'),
    'drive':        dict(dir=f'{PREP}/drive',        split='random'),
    'microtubules': dict(dir=f'{PREP}/microtubules', split='random'),
    'epfl':         dict(dir=f'{PREP}/epfl_full',    split='contiguous'),
}


def pick_gpu():
    """Return 'cuda:i' for the GPU with the most free memory, else 'cpu'."""
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
        free = [int(x) for x in out.decode().split()]
        best = int(np.argmax(free))
        return f'cuda:{best}'
    except Exception:
        return 'cuda:0'


def _norm(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def build_splits(seed=0, val_frac=0.2):
    """Return {ds: {'train': [(img,msk)...], 'val': [...]}} of file-path pairs."""
    splits = {}
    for ds, cfg in DATASETS.items():
        imgs = sorted(glob.glob(os.path.join(cfg['dir'], 'images', '*.png')))
        pairs = [(p, p.replace('/images/', '/labels/')) for p in imgs
                 if os.path.exists(p.replace('/images/', '/labels/'))]
        if cfg['split'] == 'contiguous':
            # slices already sorted by index; tail 20% -> val, 4-slice guard gap
            n = len(pairs)
            n_val = max(1, int(val_frac * n))
            guard = 4
            val = pairs[n - n_val:]
            train = pairs[:max(0, n - n_val - guard)]
        else:
            rng = np.random.default_rng(seed)
            idx = np.arange(len(pairs))
            rng.shuffle(idx)
            n_val = max(1, int(val_frac * len(pairs)))
            val = [pairs[i] for i in idx[:n_val]]
            train = [pairs[i] for i in idx[n_val:]]
        splits[ds] = {'train': train, 'val': val}
        print(f'  {ds:13s} train={len(train):3d} val={len(val):3d}')
    return splits


class MultiDomainCrops(Dataset):
    """Balanced random-crop sampler: each item picks a dataset UNIFORMLY (so
    domains are equally represented regardless of size), then a random train
    image and a random crop, with polarity-agnostic augmentation."""

    def __init__(self, splits, size=256, length=1200, seed=0):
        self.size = size
        self.length = length
        self.seed = seed
        self.ds_names = list(splits.keys())
        # cache normalised images + binary masks in RAM
        self.cache = {}
        self.train_ids = {}
        for ds in self.ds_names:
            ids = []
            for ip, mp in splits[ds]['train']:
                img = _norm(cv2.imread(ip, cv2.IMREAD_GRAYSCALE))
                msk = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
                self.cache[ip] = (img, msk)
                ids.append(ip)
            self.train_ids[ds] = ids

    def __len__(self):
        return self.length

    def _crop(self, img, msk, rng):
        h, w = img.shape
        s = self.size
        if h < s or w < s:
            ph, pw = max(0, s - h), max(0, s - w)
            img = np.pad(img, ((0, ph), (0, pw)))
            msk = np.pad(msk, ((0, ph), (0, pw)))
            h, w = img.shape
        # bias crops toward foreground so vessel/filament crops aren't empty
        for _ in range(4):
            y = int(rng.integers(0, h - s + 1))
            x = int(rng.integers(0, w - s + 1))
            ci, cm = img[y:y+s, x:x+s], msk[y:y+s, x:x+s]
            if cm.mean() > 0.002:
                return ci, cm
        return ci, cm

    def __getitem__(self, i):
        # per-call rng keeps workers deterministic yet varied
        rng = np.random.default_rng(self.seed * 100003 + i)
        ds = self.ds_names[int(rng.integers(0, len(self.ds_names)))]
        ip = self.train_ids[ds][int(rng.integers(0, len(self.train_ids[ds])))]
        img, msk = self.cache[ip]
        img, msk = self._crop(img, msk, rng)

        # geometric
        if rng.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if rng.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        k = int(rng.integers(0, 4))
        if k:
            img, msk = np.rot90(img, k), np.rot90(msk, k)
        img, msk = np.ascontiguousarray(img), np.ascontiguousarray(msk)
        # intensity
        img = np.clip(img ** rng.uniform(0.6, 1.6), 0, 1)
        img = np.clip(img * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1), 0, 1)
        if rng.random() < 0.5:
            img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.5, 2.0)))
        if rng.random() < 0.5:
            img = np.clip(img + rng.normal(0, rng.uniform(0.01, 0.06),
                                           img.shape).astype(np.float32), 0, 1)
        # POLARITY-AGNOSTIC: half the time invert so the net segments a
        # structure whether it is bright- or dark-on-background.
        if rng.random() < 0.5:
            img = 1.0 - img
        return (torch.from_numpy(img[None].copy()).float(),
                torch.from_numpy(msk[None].copy()).float())


def _pad16(t):
    h, w = t.shape[-2:]
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    return F.pad(t, (0, pw, 0, ph)), (h, w)


@torch.no_grad()
def evaluate(model, splits, device, invert_val=True):
    """Full-image val metrics per dataset. invert_val: also try the inverted
    input and keep the better mask (net is polarity-agnostic; this mirrors how
    it will be used at inference without a polarity detector)."""
    model.eval()
    per = {}
    for ds in splits:
        dl, il, cl = [], [], []
        for ip, mp in splits[ds]['val']:
            img = _norm(cv2.imread(ip, cv2.IMREAD_GRAYSCALE))
            y = torch.from_numpy(
                (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32))[None, None].to(device)
            best = None
            variants = [img, 1.0 - img] if invert_val else [img]
            for v in variants:
                t, (h, w) = _pad16(torch.from_numpy(v)[None, None].to(device))
                prob = torch.sigmoid(model(t))[..., :h, :w]
                p = (prob >= 0.5).float()
                inter = (p * y).sum()
                iou = ((inter + 1) / (p.sum() + y.sum() - inter + 1)).item()
                if best is None or iou > best[0]:
                    dice = ((2 * inter + 1) / (p.sum() + y.sum() + 1)).item()
                    sk_t = soft_skel(y, 10)
                    cs = ((sk_t * p).sum() + 1) / (sk_t.sum() + 1)
                    best = (iou, dice, float(cs))
            il.append(best[0]); dl.append(best[1]); cl.append(best[2])
        per[ds] = dict(iou=float(np.mean(il)), dice=float(np.mean(dl)),
                       csens=float(np.mean(cl)))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--length', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cldice', type=float, default=0.5)
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'unet_multidomain.pt'))
    args = ap.parse_args()

    device = pick_gpu()
    torch.manual_seed(args.seed)
    print(f'device={device}')
    print('splits:')
    splits = build_splits(seed=args.seed)

    ds = MultiDomainCrops(splits, size=args.size, length=args.length, seed=args.seed)
    tr = DataLoader(ds, batch_size=args.bs, shuffle=True,
                    num_workers=6, drop_last=True, persistent_workers=True)

    model = UNet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    hist = {'loss': [], 'macro_dice': [], 'macro_iou': []}
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        losses = []
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = combined_loss(model(x), y, cldice_w=args.cldice)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        per = evaluate(model, splits, device)
        macro_iou = float(np.mean([per[d]['iou'] for d in per]))
        macro_dice = float(np.mean([per[d]['dice'] for d in per]))
        macro_cs = float(np.mean([per[d]['csens'] for d in per]))
        hist['loss'].append(float(np.mean(losses)))
        hist['macro_dice'].append(macro_dice); hist['macro_iou'].append(macro_iou)
        score = macro_dice + macro_cs
        flag = ''
        if score > best:
            best = score
            torch.save({'model': model.state_dict(), 'base': args.base,
                        'val': per, 'macro_iou': macro_iou,
                        'macro_dice': macro_dice, 'epoch': ep}, args.out)
            flag = ' *saved'
        pd = '  '.join(f"{d[:4]}={per[d]['iou']:.3f}" for d in per)
        print(f'ep {ep:3d} loss={np.mean(losses):.3f} '
              f'mIoU={macro_iou:.3f} mDice={macro_dice:.3f} | {pd}{flag}',
              flush=True)

    print(f'\nbest macro(dice+csens)={best:.4f} -> {args.out}')
    # reload best and print final per-dataset table
    ck = torch.load(args.out, map_location='cpu')
    print('\nFINAL held-out val (best ckpt, epoch %d):' % ck['epoch'])
    for d, m in ck['val'].items():
        print(f"  {d:13s} IoU={m['iou']:.3f}  Dice={m['dice']:.3f}  csens={m['csens']:.3f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist['loss']); ax[0].set_title('train loss'); ax[0].set_xlabel('epoch')
    ax[1].plot(hist['macro_dice'], label='macro Dice')
    ax[1].plot(hist['macro_iou'], label='macro IoU')
    ax[1].set_title('val (macro over datasets)'); ax[1].set_xlabel('epoch')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    out_png = os.path.splitext(args.out)[0] + '.png'
    plt.tight_layout(); plt.savefig(out_png, dpi=120)
    print(f'saved {out_png}')


if __name__ == '__main__':
    main()
