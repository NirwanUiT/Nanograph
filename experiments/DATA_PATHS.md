# Data roots used by the paper runs

All paths are on the vs-c2 NAS. None of these datasets are committed to the repo.

| Dataset | Images | Masks |
|---|---|---|
| Organelle main set (726, in-house acquisition) | `/mnt/nas1/nba055-2/idea_1/nmi_data/org` | `/mnt/nas1/nba055-2/idea_1/nmi_data/seg` |
| Cells3D membrane (60) | `/mnt/nas1/nba055-2/idea_1/datasets/cells3d_membrane/images` | none (scikit-image sample data) |
| Cells3D nuclei (60) | `/mnt/nas1/nba055-2/idea_1/datasets/cells3d_nuclei/images` | none |
| Retina tiles (25) | `/mnt/nas1/nba055-2/idea_1/datasets/retina/images` | none |
| Fluorescence cells (5) | `/mnt/nas1/nba055-2/idea_1/datasets/cell/images` | none |
| STARE (20) | `/mnt/nas1/nba055-2/idea_1/ext_datasets/prepared/stare/images` | `.../stare/labels` |
| DRIVE (20) | `/mnt/nas1/nba055-2/idea_1/ext_datasets/prepared/drive/images` | `.../drive/labels` |
| EPFL mito EM (10) | `/mnt/nas1/nba055-2/idea_1/ext_datasets/prepared/epfl_mito/images` | `.../epfl_mito/labels` |
| IRM microtubules (66) | `/mnt/nas1/nba055-2/idea_1/ext_datasets/prepared/microtubules/images` | `.../microtubules/labels` |
| Temporal clip (73, same acquisition as main set) | `/mnt/nas1/nba055-2/idea_1/mito_aaron/images` | `/mnt/nas1/nba055-2/idea_1/mito_aaron/masks` |
| STED TOM20 (345, Zenodo 14215838) | `/mnt/nas1/nba055-2/idea_1/public_mito/sted_prep/images` | none (denoising GT only) |
| MITO MIP tiles (228, Zenodo 7724799) | `/mnt/nas1/nba055-2/idea_1/public_mito/mito_mip_tiles/images` | `.../mito_mip_tiles/masks` |
| MITO full frames (41) | `/mnt/nas1/nba055-2/idea_1/public_mito/mito_zenodo/images` | `.../mito_zenodo/masks` |
| MITO in-focus tiles (271) | `/mnt/nas1/nba055-2/idea_1/public_mito/mito_zenodo_tiles/images` | `.../mito_zenodo_tiles/masks` |

U-Net training data = the organelle main set (85/15 seed-0 split; see
`experiments/write_heldout.py`). Curvilinear multi-domain training data =
STARE/DRIVE/EPFL/microtubules prepared dirs above.
