# try:
#     from pkg_resources import declare_namespace
#     declare_namespace(__name__)
# except ImportError:
#     from pkgutil import extend_path
#     __path__ = extend_path(__path__, __name__)

from os.path import dirname, basename, isfile, join
import glob
from .__version__ import __version__

modules = glob.glob(join(dirname(__file__), "*.py"))

__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]


# Read the __version__.py file
# with open('__version__.txt', 'r') as f:
#     ver = f.read()
# ver = ('__version__.txt', 'r').read_text()
#
# __version__ = '0.0.3'

# with open("__version__.txt", 'r') as fh:
#     ver = fh.read().strip()
#
# __version__ = '0.0.3'
print(__version__)
