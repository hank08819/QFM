"""Build a model on reference atlases, project a new atlas and cluster it. Usage:
python quickstart.py model.npz new_atlas.h5ad k ref1.h5ad [ref2.h5ad ...]"""
import sys
import numpy as np
from kqefm import KQEFM, cluster

out, atlas, k, refs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
model = KQEFM.build_from_h5ad(refs); model.save(out)
E = model.project_h5ad(atlas); np.save("scores.npy", E)
np.save("labels.npy", cluster(E, k)); print("done")
