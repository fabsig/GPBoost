# coding: utf-8
"""Settings for the tests of the Python package."""
from gpboost.basic import _suppress_num_threads_message

# The message about the automatically selected number of threads is written once per process, when a
# model uses that number of threads for the first time. It would land in the output of whichever test
# happens to run first, so it is switched off for the whole test suite
_suppress_num_threads_message()
