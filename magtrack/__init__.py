from warnings import filterwarnings

filterwarnings('ignore', category=FutureWarning, message='cupyx.jit.rawkernel')

from magtrack.core import *
import magtrack.utils