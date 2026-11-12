"""The frozen model: one exact decomposition of a reference, stored and reused unchanged on every new atlas.

Build (once): sample cells from each reference atlas (fixed seed), map genes to one Ensembl universe, compute IDF weights,
preprocess, and take the leading right singular vectors of the implicitly centered matrix with ARPACK started from a
fixed vector of ones. The model is (genes, w, mu, V, s, n).
Use (zero-shot): map a new atlas to the same genes, preprocess with the reference weights and project,
e_j = sqrt(n) (z - mu)^T v_j / s_j, which gives unit variance over the reference cells. No step draws a random number.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds
from .preprocess import to_universe, idf_weights, prep
from .io import open_atlas, gene_ids, symbol_map, read_rows


class KQEFM:
    """Frozen model. Attributes: genes (universe), w (IDF weights), mu (reference mean), Vt (directions x genes),
    s (singular values), n_ref (reference cells), sym2ens (symbol -> Ensembl map used for symbol-indexed atlases)."""

    def __init__(self, genes, w, mu, Vt, s, n_ref, sym2ens=None):
        self.genes = np.asarray(genes); self.w = np.asarray(w, np.float32); self.mu = np.asarray(mu, np.float64)
        self.Vt = np.asarray(Vt, np.float64); self.s = np.asarray(s, np.float64); self.n_ref = int(n_ref)
        self.sym2ens = dict(sym2ens or {}); self.pos = {g: i for i, g in enumerate(self.genes)}
        self._P = (self.Vt.T / self.s[None, :]) * np.sqrt(self.n_ref); self._off = self.mu @ self._P

    @property
    def n_directions(self):
        return len(self.s)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, matrices, gene_lists, n_directions=132, sym2ens=None, verbose=True):
        """Fit the model on reference count matrices (cells x genes, one per atlas) with their Ensembl gene lists."""
        t0 = time.time()
        genes = np.array(sorted(set().union(*[set(np.asarray(g)[np.asarray(g) != ""]) for g in gene_lists])))
        pos = {g: i for i, g in enumerate(genes)}
        R = sp.vstack([to_universe(M, g, pos) for M, g in zip(matrices, gene_lists)], format="csr")
        w = idf_weights(R); Z = prep(R, w); del R
        n = Z.shape[0]; mu = np.asarray(Z.mean(0)).ravel().astype(np.float64); ZT = Z.T.tocsr()
        A = LinearOperator(Z.shape, matvec=lambda x: Z @ np.ravel(x).astype(np.float64) - float(mu @ np.ravel(x)),
                           rmatvec=lambda u: ZT @ np.ravel(u).astype(np.float64) - mu * float(np.sum(u)), dtype=np.float64)
        m = min(Z.shape); v0 = np.ones(m) / np.sqrt(m)
        U, s, Vt = svds(A, k=n_directions, tol=1e-6, v0=v0, solver="arpack", maxiter=100000)
        o = np.argsort(s)[::-1]; U, s, Vt = U[:, o], s[o], Vt[o]
        sg = np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(len(s))]); Vt = Vt * sg[:, None]
        if verbose:
            print(f"KQE-FM built: {n:,} reference cells x {len(genes):,} genes, {n_directions} directions, {time.time() - t0:.0f} s", flush=True)
        return cls(genes, w, mu, Vt, s, n, sym2ens)

    @classmethod
    def build_from_h5ad(cls, paths, cells_per_atlas=230000, seed=0, n_directions=132, verbose=True):
        """Fit the model on .h5ad reference atlases: `cells_per_atlas` cells drawn from each (all cells when fewer)."""
        mats, glists, sym2ens = [], [], {}
        for p in paths:
            H, X, var = open_atlas(p); sym2ens.update(symbol_map(var))
            rows = np.sort(np.random.RandomState(seed).choice(H.n_obs, min(cells_per_atlas, H.n_obs), replace=False))
            mats.append(read_rows(X, rows)); glists.append(gene_ids(var, sym2ens))
            if verbose:
                print(f"reference {p}: {len(rows):,} cells", flush=True)
        return cls.build(mats, glists, n_directions=n_directions, sym2ens=sym2ens, verbose=verbose)

    # ------------------------------------------------------------------ use
    def project(self, M, ids, chunk=100000):
        """Whitened scores (cells x n_directions) of a count matrix M with gene identifiers `ids` (Ensembl IDs)."""
        M = sp.csr_matrix(M); E = np.empty((M.shape[0], self.n_directions), np.float32)
        for i in range(0, M.shape[0], chunk):
            Z = prep(to_universe(M[i:i + chunk], ids, self.pos), self.w)
            E[i:i + chunk] = (Z @ self._P - self._off).astype(np.float32)
        return E

    def project_h5ad(self, path, rows=None, chunk=100000):
        """Whitened scores of the cells of an .h5ad file (all cells, or the given row indices, in that order)."""
        H, X, var = open_atlas(path); ids = gene_ids(var, self.sym2ens)
        rows = np.arange(H.n_obs) if rows is None else np.asarray(rows)
        o = np.argsort(rows); inv = np.empty_like(o); inv[o] = np.arange(len(o)); rs = rows[o]
        E = np.empty((len(rows), self.n_directions), np.float32)
        for i in range(0, len(rs), chunk):
            E[i:i + chunk] = self.project(read_rows(X, rs[i:i + chunk]), ids)
        return E[inv]

    def gene_coverage(self, ids):
        """Share of an atlas's genes that are in the model universe."""
        return float(np.mean([g in self.pos for g in ids]))

    # ------------------------------------------------------------------ storage
    def save(self, path):
        np.savez(path, genes=self.genes, w=self.w, mu=self.mu, Vt=self.Vt, s=self.s, n_ref=self.n_ref,
                 sym_keys=np.array(list(self.sym2ens.keys())), sym_vals=np.array(list(self.sym2ens.values())))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        sym = dict(zip(z["sym_keys"], z["sym_vals"])) if "sym_keys" in z else {}
        return cls(z["genes"], z["w"], z["mu"], z["Vt"], z["s"], int(z["n_ref"]), sym)
