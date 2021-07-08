import os.path as osp
import sys
from sys import platform as _platform

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

if _platform == "win32" or _platform == "win64":
    lib_path = osp.join(this_dir, '\src\lib')
else:
    lib_path = osp.join(this_dir, 'src/lib')

# Add lib to PYTHONPATH
add_path(lib_path)
