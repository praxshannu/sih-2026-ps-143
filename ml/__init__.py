"""SENTINEL machine-learning code: training, datasets, metrics, synthetic data.

Sub-packages are importable from the repository root (``pythonpath = ["."]``
in ``pyproject.toml``)::

    from ml.synth import profile_from_spec
    from ml.training.train_detector import train
    from ml.training.transforms import build_train_transforms

``ml/training`` also supports flat imports (``from dataset import ...``)
because the original modules put their own directory on ``sys.path``; that
still works and is what the per-directory tests rely on.
"""
