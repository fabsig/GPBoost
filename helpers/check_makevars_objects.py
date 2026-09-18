# coding: utf-8
"""Helper script for checking the object lists of the R package.

CMake globs 'src/GPBoost/', the Makevars of the R package do not: a source file that is missing
from their OBJECTS lists is only noticed when the R package is built, which happens long after the
file has been added. This script compares the two.
"""
import os
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
SOURCE_DIR = os.path.join(REPO_DIR, "src", "GPBoost")
MAKEVARS = ["Makevars.in", "Makevars.win.in", "Makevars.win"]


def check_makevars_objects():
    """Check that every GPBoost source file is listed in the Makevars of the R package.

    Returns
    -------
    missing : list
        A list of (Makevars file, object file) pairs that are missing, empty if there are none.
    """
    objects = sorted(f[:-len(".cpp")] + ".o" for f in os.listdir(SOURCE_DIR) if f.endswith(".cpp"))
    missing = []
    for makevars in MAKEVARS:
        path = os.path.join(REPO_DIR, "R-package", "src", makevars)
        with open(path) as makevars_file:
            entries = {line.strip().rstrip("\\").strip() for line in makevars_file}
        for obj in objects:
            if obj not in entries:
                missing.append((makevars, obj))
    return missing


if __name__ == "__main__":
    missing_objects = check_makevars_objects()
    for makevars_file_name, object_file_name in missing_objects:
        print("'%s' is missing from the OBJECTS list of 'R-package/src/%s'"
              % (object_file_name, makevars_file_name))
    sys.exit(1 if missing_objects else 0)
