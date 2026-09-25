#!/usr/bin/env python3
"""
T18 bake-off: Allen training tiles -> nnU-Net v2 raw dataset (Dataset501_AllenMito),
with OUR train/val split as fold 0 (splits_final.json), and the test cells as
an inference folder. Masks become 0/1 labels; images stay uint8 PNG.

Usage: python allen_nnunet_prep.py --stage data   (before planning)
       python allen_nnunet_prep.py --stage split  (after planning, writes splits_final.json)
       python allen_nnunet_prep.py --stage export --pred-in DIR --pred-out DIR
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np
import pandas as pd

ROOT = '/mnt/nas1/nba055-2/idea_1/archive_demo/allen'
NN = '/mnt/nas1/nba055-2/idea_1/archive_demo/nnunet'
NAME = 'Dataset501_AllenMito'


def data():
    M = pd.read_csv(f'{ROOT}/trainset/manifest.csv')
    M = M[M.split.isin(['train', 'val'])]
    raw = f'{NN}/nnUNet_raw/{NAME}'
    for d in ('imagesTr', 'labelsTr'):
        os.makedirs(f'{raw}/{d}', exist_ok=True)
    for i in M.id:
        shutil.copyfile(f'{ROOT}/trainset/images/{i}.png', f'{raw}/imagesTr/{i}_0000.png')
        m = cv2.imread(f'{ROOT}/trainset/masks/{i}.png', cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f'{raw}/labelsTr/{i}.png', (m > 0).astype(np.uint8))
    json.dump({'channel_names': {'0': 'TOMM20'}, 'labels': {'background': 0, 'mitochondria': 1},
               'numTraining': len(M), 'file_ending': '.png'}, open(f'{raw}/dataset.json', 'w'), indent=1)
    test = f'{NN}/test_in'
    os.makedirs(test, exist_ok=True)
    for p in os.listdir(f'{ROOT}/testset/img'):
        shutil.copyfile(f'{ROOT}/testset/img/{p}', f'{test}/{p[:-4]}_0000.png')
    print(f'{len(M)} training tiles, {len(os.listdir(test))} test cells')


def split():
    M = pd.read_csv(f'{ROOT}/trainset/manifest.csv')
    s = [{'train': sorted(M[M.split == 'train'].id), 'val': sorted(M[M.split == 'val'].id)}]
    json.dump(s, open(f'{NN}/nnUNet_preprocessed/{NAME}/splits_final.json', 'w'))
    print(f"fold 0: {len(s[0]['train'])} train / {len(s[0]['val'])} val")


def export(src, dst):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for p in os.listdir(src):
        if p.endswith('.png'):
            m = cv2.imread(os.path.join(src, p), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(dst, p), (m > 0).astype(np.uint8) * 255)
            n += 1
    print(f'{n} predictions -> {dst}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['data', 'split', 'export'])
    ap.add_argument('--pred-in')
    ap.add_argument('--pred-out')
    a = ap.parse_args()
    {'data': data, 'split': split}.get(a.stage, lambda: export(a.pred_in, a.pred_out))()
