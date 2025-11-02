from warnings import filterwarnings

filterwarnings('ignore', category=FutureWarning, message='cupyx.jit.rawkernel')

from magtrack.core import *  # noqa: F401,F403
from ._cupy import check_cupy

__all__ = list(globals().get('__all__', []))
__all__.append('check_cupy')
