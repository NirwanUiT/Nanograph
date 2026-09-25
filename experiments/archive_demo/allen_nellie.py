#!/usr/bin/env python3
"""
T18 bake-off: Nellie (Lefebvre et al., Nat Methods 2025), zero-shot, on the
unmasked slab image of every test cell (allen_export_test.py). Only Nellie's
filtering (multiscale Frangi) and labelling steps are used; the union of its
instance labels is the mask, written as <out>/<CellId>.png (0/255).

Runs in the Nellie venv (it has its own dependencies):
    /var/tmp/nba055/venvs/nellie/bin/python experiments/archive_demo/allen_nellie.py
"""
import argparse
import contextlib
import io
import logging
import os
import shutil
import tempfile

import numpy as np

TEST = '/mnt/nas1/nba055-2/idea_1/archive_demo/allen/testset'
UM = 0.108333


def one(args):
    cid, out = args
    import cv2
    import tifffile
    from nellie.im_info.verifier import FileInfo, ImInfo
    from nellie.segmentation.filtering import Filter
    from nellie.segmentation.labelling import Label
    logging.disable(logging.CRITICAL)
    im = cv2.imread(f'{TEST}/img_raw/{cid}.png', cv2.IMREAD_GRAYSCALE)
    tmp = tempfile.mkdtemp(dir=os.environ.get('TMPDIR', '/var/tmp'))
    try:
        p = os.path.join(tmp, f'{cid}.tif')
        tifffile.imwrite(p, im)
        with contextlib.redirect_stdout(io.StringIO()):
            fi = FileInfo(p, output_dir=os.path.join(tmp, 'out'))
            fi.find_metadata()
            fi.load_metadata()
            fi.change_axes('YX')
            fi.change_dim_res('X', UM)
            fi.change_dim_res('Y', UM)
            ii = ImInfo(fi)
            Filter(ii, remove_edges=False, device='cpu').run()
            Label(ii, device='cpu').run()
            lab = np.asarray(ii.get_memmap(ii.pipeline_paths['im_instance_label']))
        cv2.imwrite(os.path.join(out, f'{cid}.png'), ((lab.reshape(im.shape) > 0) * 255).astype(np.uint8))
        return cid, ''
    except Exception as ex:
        return cid, f'{type(ex).__name__}: {ex}'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/nas1/nba055-2/idea_1/archive_demo/allen/preds/nellie')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids = sorted(p[:-4] for p in os.listdir(f'{TEST}/img_raw'))
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(one, [(c, a.out) for c in ids])
    errs = [f'{c}: {e}' for c, e in res if e]
    print(f'{len(ids) - len(errs)}/{len(ids)} cells segmented; errors: {len(errs)}')
    for e in errs[:10]:
        print(' ', e)


if __name__ == '__main__':
    main()
