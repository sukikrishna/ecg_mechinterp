"""ecg_mechinterp: mechanistic interpretability of ECG foundation models.

Merges three prior analyses (a multi-model PTB-XL probing/CKA/SAE study, a single-model
CLEF SAE convergence study on PTB-DB, and a structured RQ1-4 stability-and-bias study on
PTB-DB) into one dataset-agnostic, model-agnostic library. See docs/provenance.md for where
each module's methodology came from, and docs/findings.md for the consolidated results that
motivated this design.
"""

__version__ = "0.1.0"
