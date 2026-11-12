"""Zero-shot clustering of projected cells, using the quantum encoding and Q-PSC of scKQE-2.

Each of the whitened scores turns one qubit (22 blocks of six qubits for 132 directions, three re-uploading layers of
R_Y and R_Z rotations, s = 0.19, no entangling gate); cells are compared by the exact fidelity of their product states,
joined to their certified 15 nearest states, and partitioned by Q-PSC into k clusters.
"""
import numpy as np


def bloch(E, s=0.19):
    """Bloch vectors (cells, n_directions, 3) of the projected scores."""
    from sckqe2.qstate import bloch_chunked
    return bloch_chunked(np.asarray(E), s)


def quantum_features(E, s=0.19, qubits=6, backend="closed_form"):
    """The 726 quantum features per cell (for 132 directions): Bloch vectors and within-block <Z_i Z_j>."""
    from sckqe2.encode import encode
    return encode(np.asarray(E), s=s, qubits=qubits, backend=backend)


def cluster(E, k, s=0.19, n_neighbors=15, seed=0, return_graph=False, **knn_kw):
    """Q-PSC on the projected scores E: certified exact nearest states, fidelity-weighted graph, Leiden, merge to k.
    Returns integer labels (and, with return_graph=True, the graph, neighbors, distances and certificates)."""
    from sckqe2.qpsc import qpsc
    out = qpsc(np.asarray(E), k, s=s, n_neighbors=n_neighbors, seed=seed, return_graph=return_graph, **knn_kw)
    if return_graph:
        labels, res, G, I, D, cert = out
        return np.asarray(labels), G, I, D, cert
    return np.asarray(out[0])
