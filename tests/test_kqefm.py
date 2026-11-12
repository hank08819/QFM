import numpy as np
import scipy.sparse as sp
import pytest
from kqefm import KQEFM, transfer_share


def toy(seed, n=400, p=300, k=4):
    rs = np.random.RandomState(seed); y = rs.randint(k, size=n); prog = rs.gamma(1.0, 1.0, size=(k, p))
    C = rs.poisson(prog[y] * 0.8); C[:, rs.rand(p) < 0.1] = 0
    genes = np.array([f"ENSG{j:011d}" for j in range(p)])
    return sp.csr_matrix(C.astype(np.float32)), genes, y


def test_build_is_deterministic_and_projection_is_whitened():
    (A, ga, _), (B, gb, _) = toy(0), toy(1)
    m1 = KQEFM.build([A, B], [ga, gb], n_directions=8, verbose=False)
    m2 = KQEFM.build([A, B], [ga, gb], n_directions=8, verbose=False)
    assert np.array_equal(m1.Vt, m2.Vt) and np.array_equal(m1.s, m2.s)
    E = m1.project(sp.vstack([A, B]), ga)
    assert E.shape == (800, 8)
    assert np.allclose((E ** 2).mean(0) - E.mean(0) ** 2, 1.0, atol=1e-3)


def test_new_atlas_with_missing_and_extra_genes(tmp_path):
    (A, ga, _), (C, gc, y) = toy(0), toy(2)
    m = KQEFM.build([A], [ga], n_directions=6, verbose=False)
    gc = gc.copy(); gc[:20] = "ENSG_NOT_IN_UNIVERSE"
    E1 = m.project(C, gc); E2 = m.project(C, gc)
    assert np.array_equal(E1, E2) and E1.shape == (C.shape[0], 6)
    m.save(tmp_path / "m.npz"); m2 = KQEFM.load(tmp_path / "m.npz")
    assert np.array_equal(m2.project(C, gc), E1)
    sh = transfer_share(m, C, gc, labels=y)
    assert 0 < sh["share_total"] <= 1 + 1e-9 and 0 < sh["share_between"] <= 1 + 1e-9


def test_cluster_runs_when_sckqe2_is_installed():
    pytest.importorskip("sckqe2"); pytest.importorskip("torch")
    from kqefm import cluster
    (A, ga, y) = toy(3)
    m = KQEFM.build([A], [ga], n_directions=12, verbose=False)
    E = m.project(A, ga)
    l1 = cluster(E, 4); l2 = cluster(E, 4)
    assert np.array_equal(l1, l2) and len(np.unique(l1)) == 4
