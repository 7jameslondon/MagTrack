from warnings import filterwarnings

filterwarnings('ignore', category=FutureWarning, message='cupyx.jit.rawkernel')

from magtrack.core import *  # noqa: F401,F403
from magtrack._cupy import check_cupy
