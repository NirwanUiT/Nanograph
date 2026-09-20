#!/usr/bin/env python3
"""T4: reproduce the organelle U-Net held-out split and write the stems.

Split definition (experiments/train_unet.py): sorted basenames of org/*.png,
shuffled with np.random.default_rng(0), first int(0.15*n) = validation.
"""
import glob
import os
import sys

import numpy as np

ORG = '/mnt/nas1/nba055-2/idea_1/nmi_data/org'
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', 'paper', 'heldout_organelle.txt')

ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ORG, '*.png')))
rng = np.random.default_rng(0)
rng.shuffle(ids)
n_val = max(1, int(0.15 * len(ids)))
stems = [os.path.splitext(i)[0] for i in ids[:n_val]]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    f.write('\n'.join(stems) + '\n')
print(f'wrote {OUT} ({len(stems)} stems of {len(ids)})')
