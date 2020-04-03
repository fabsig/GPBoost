# coding: utf-8
"""Find the path to GPBoost dynamic library files."""
import os

from platform import system


def find_lib_path():
    """Find the path to GPBoost library files.

    Returns
    -------
    lib_path: list of strings
       List of all found library paths to GPBoost.
    """
    if os.environ.get('GPBOOST_BUILD_DOC', False):
        # we don't need lib_gpboost while building docs
        return []

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [curr_path,
                os.path.join(curr_path, '../../'),
                os.path.join(curr_path, 'compile'),
                os.path.join(curr_path, '../compile'),
                os.path.join(curr_path, '../../lib/')]
    if system() in ('Windows', 'Microsoft'):
        dll_path.append(os.path.join(curr_path, '../compile/Release/'))
        dll_path.append(os.path.join(curr_path, '../compile/windows/x64/DLL/'))
        dll_path.append(os.path.join(curr_path, '../../Release/'))
        dll_path.append(os.path.join(curr_path, '../../windows/x64/DLL/'))
        dll_path = [os.path.join(p, 'lib_gpboost.dll') for p in dll_path]
    else:
        dll_path = [os.path.join(p, 'lib_gpboost.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if not lib_path:
        dll_path = [os.path.realpath(p) for p in dll_path]
        raise Exception('Cannot find gpboost library file in following paths:\n' + '\n'.join(dll_path))
    return lib_path
