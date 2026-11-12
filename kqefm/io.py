"""Reading counts and gene identifiers from .h5ad files (raw counts are taken from .raw when present)."""
import numpy as np
import scipy.sparse as sp


def open_atlas(path):
    """Open an .h5ad file in backed mode; returns (AnnData, count matrix, var table)."""
    import anndata as ad
    H = ad.read_h5ad(path, backed="r"); raw = H.raw is not None
    return H, (H.raw.X if raw else H.X), (H.raw.var if raw else H.var)


def gene_ids(var, sym2ens=None):
    """Ensembl identifiers of the genes of an atlas: the var index when it holds Ensembl IDs, otherwise gene symbols
    (column 'feature_name' or the index) mapped through `sym2ens`; unmapped genes get ''."""
    idx = var.index.astype(str)
    if idx.str.startswith("ENS").mean() > 0.5:
        return np.array(idx)
    names = var["feature_name"].astype(str).values if "feature_name" in var else np.array(idx)
    sym2ens = sym2ens or {}
    return np.array([sym2ens.get(s, "") for s in names])


def symbol_map(var):
    """Symbol -> Ensembl identifier pairs from an atlas whose var index holds Ensembl IDs and has 'feature_name'."""
    if "feature_name" not in var:
        return {}
    return {s: e for s, e in zip(var["feature_name"].astype(str), var.index.astype(str)) if str(e).startswith("ENS")}


def read_rows(X, rows, chunk=20000):
    """Rows of a (backed) count matrix as one float32 CSR matrix; `rows` must be sorted."""
    parts = []
    for i in range(0, len(rows), chunk):
        M = X[rows[i:i + chunk]]
        parts.append(sp.csr_matrix(M if sp.issparse(M) else np.asarray(M), dtype=np.float32))
    return sp.vstack(parts, format="csr")
