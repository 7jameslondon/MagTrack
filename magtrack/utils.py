from functools import lru_cache

from ._cupy import cp, is_cupy_available

@lru_cache(maxsize=1)
def check_cupy():
    if not is_cupy_available():
        return False

    try:
        import cupy as cupy_mod  # type: ignore

        if not cupy_mod.cuda.is_available():
            return False

        cupy_mod.random.randint(0, 1, size=(1,))  # Test cupy
    except Exception:  # pragma: no cover - defensive fallback
        return False
    else:
        return True