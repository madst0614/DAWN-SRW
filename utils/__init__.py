"""
DAWN Utilities

Data processing, training, and checkpoint utilities.
"""

# Lazy imports — PyTorch utils loaded only when accessed (allows torch-free JAX usage)
def __getattr__(name):
    _data_names = {"CacheLoader"}
    _training_names = {"CheckpointManager", "TrainingMonitor", "format_time", "count_parameters"}
    _checkpoint_names = {
        "VERSION_PARAM_CHANGES", "strip_compile_prefix", "categorize_keys",
        "load_checkpoint_smart", "print_load_info", "load_optimizer_state",
    }

    if name in _data_names:
        from . import data
        return getattr(data, name)
    if name in _training_names:
        from . import training
        return getattr(training, name)
    if name in _checkpoint_names:
        from . import checkpoint
        return getattr(checkpoint, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Data utilities
    "CacheLoader",
    # Training utilities
    "CheckpointManager",
    "TrainingMonitor",
    "format_time",
    "count_parameters",
    # Checkpoint utilities
    "VERSION_PARAM_CHANGES",
    "strip_compile_prefix",
    "categorize_keys",
    "load_checkpoint_smart",
    "print_load_info",
    "load_optimizer_state",
]
