"""How well does the frozen reference span a new atlas? (computed before clustering, without labels)

share_total: variance of the atlas captured by the frozen directions, relative to its own leading directions
(same number). Optionally, with annotations, share_between: the same ratio for the variance between annotated groups.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds
from .preprocess import to_universe, prep


def transfer_share(model, M, ids, labels=None, max_cells=30000, seed=0):
    """Return dict(share_total=..., share_between=... or None) for count matrix M (cells x genes, identifiers `ids`)."""
    M = sp.csr_matrix(M)
    if M.shape[0] > max_cells:
        M = M[np.sort(np.random.RandomState(seed).choice(M.shape[0], max_cells, replace=False))]
        if labels is not None:
            labels = np.asarray(labels)[np.sort(np.random.RandomState(seed).choice(len(labels), max_cells, replace=False))]
    Z = prep(to_universe(M, ids, model.pos), model.w).astype(np.float64); n = Z.shape[0]; m = np.asarray(Z.mean(0)).ravel()
    Vt = model.Vt; L = Vt.shape[0]
    cap = (((Z @ Vt.T) - (m @ Vt.T)[None, :]) ** 2).sum()
    A = LinearOperator(Z.shape, matvec=lambda x: Z @ np.ravel(x) - (m @ np.ravel(x)),
                       rmatvec=lambda u: Z.T @ np.ravel(u) - m * np.sum(u), dtype=np.float64)
    _, S, Wt = svds(A, k=L, v0=np.ones(min(Z.shape)))
    out = {"share_total": float(cap / (S ** 2).sum()), "share_between": None}
    if labels is not None:
        labs, inv = np.unique(np.asarray(labels), return_inverse=True); cnt = np.bincount(inv)
        G = sp.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(len(labs), n))
        T = np.asarray((G @ Z).todense()) / cnt[:, None] - m[None, :]
        out["share_between"] = float((cnt * ((T @ Vt.T) ** 2).sum(1)).sum() / (cnt * ((T @ Wt.T) ** 2).sum(1)).sum())
    return out
