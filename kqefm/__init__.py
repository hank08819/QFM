"""KQE-FM: a quantum foundation model for single-cell atlases, computed once and applied unchanged to new atlases.

The model is one exact decomposition of a public reference. Every new cell is projected onto its frozen directions,
encoded into a product state of qubits without entanglement, and clustered by exact quantum fidelity.
No step is trained and no step draws a random number: the same input always returns the same output.
"""
from .model import KQEFM
from .cluster import cluster, bloch, quantum_features
from .diagnostic import transfer_share
from .preprocess import to_universe, idf_weights, prep

__version__ = "1.0.0"
__author__ = "Henry Han"
__all__ = ["KQEFM", "cluster", "bloch", "quantum_features", "transfer_share", "to_universe", "idf_weights", "prep"]
