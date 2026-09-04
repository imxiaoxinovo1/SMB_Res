"""Small model registry used by training scripts."""

_REGISTRY = {}


def register(name):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_models():
    return list(_REGISTRY.keys())
