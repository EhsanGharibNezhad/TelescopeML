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


# print('TelescopeML',__version__)
