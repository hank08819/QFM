"""Gene universe and preprocessing shared by the reference and every new atlas.

Each cell becomes z_g = w_g log(1 + c_g) / || (w_h log(1 + c_h))_h ||_2, with inverse document frequency weights
w_g = log((n + 1) / (d_g + 1)) / max_h log((n + 1) / (d_h + 1)) computed once on the reference (d_g: reference cells
with a nonzero count of gene g). New atlases always use the reference weights.
"""
import numpy as np
import scipy.sparse as sp


def to_universe(M, ids, pos):
    """Reorder the columns of a cells x genes count matrix M (gene identifiers `ids`) to the model gene universe `pos`
    (identifier -> column). Genes outside the universe are dropped; universe genes absent from M stay zero."""
    j = np.array([pos.get(g, -1) for g in ids]); keep = np.where(j >= 0)[0]
    M = sp.csr_matrix(M)[:, keep].tocoo()
    return sp.csr_matrix((M.data.astype(np.float32), (M.row, j[keep][M.col])), shape=(M.shape[0], len(pos)), dtype=np.float32)


def idf_weights(R):
    """Inverse document frequency weights of the columns of the reference count matrix R (cells x universe genes)."""
    R = sp.csr_matrix(R); n = R.shape[0]
    ndet = np.bincount(R.indices[R.data != 0], minlength=R.shape[1]).astype(np.float64)
    w = np.log((n + 1) / (ndet + 1))
    return (w / w.max()).astype(np.float32)


def prep(M, w):
    """log1p, IDF weighting and unit Euclidean length per cell (rows of a sparse count matrix)."""
    M = sp.csr_matrix(M, dtype=np.float32, copy=True)
    np.log1p(M.data, out=M.data); M.data *= w[M.indices]
    sq = np.zeros(M.shape[0]); nz = np.diff(M.indptr) > 0
    sq[nz] = np.add.reduceat(M.data.astype(np.float64) ** 2, M.indptr[:-1][nz])
    nrm = np.sqrt(sq); nrm[nrm == 0] = 1
    M.data /= np.repeat(nrm, np.diff(M.indptr)).astype(np.float32)
    return M
