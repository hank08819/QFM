"""Command line: kqefm build | kqefm project | kqefm cluster."""
import argparse
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kqefm", description="Quantum foundation model for single-cell atlases")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="fit the frozen model on reference .h5ad atlases")
    b.add_argument("out", help="model file (.npz)"); b.add_argument("refs", nargs="+", help="reference .h5ad files")
    b.add_argument("--cells-per-atlas", type=int, default=230000); b.add_argument("--directions", type=int, default=132)
    p = sub.add_parser("project", help="project a new atlas onto a frozen model")
    p.add_argument("model"); p.add_argument("atlas", help=".h5ad file"); p.add_argument("out", help="scores (.npy)")
    c = sub.add_parser("cluster", help="Q-PSC on projected scores")
    c.add_argument("scores", help=".npy from 'kqefm project'"); c.add_argument("k", type=int); c.add_argument("out", help="labels (.npy)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        from .model import KQEFM
        KQEFM.build_from_h5ad(a.refs, cells_per_atlas=a.cells_per_atlas, n_directions=a.directions).save(a.out)
    elif a.cmd == "project":
        from .model import KQEFM
        np.save(a.out, KQEFM.load(a.model).project_h5ad(a.atlas))
    elif a.cmd == "cluster":
        from .cluster import cluster
        np.save(a.out, cluster(np.load(a.scores), a.k))


if __name__ == "__main__":
    main()
