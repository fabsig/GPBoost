Documentation
=============

The documentation for GPBoost is generated using `Sphinx <https://www.sphinx-doc.org/>`__
and `Breathe <https://breathe.readthedocs.io/>`__, which works on top of `Doxygen <https://www.doxygen.nl/index.html>`__ output.

The list of all tree-boosting related parameters and their description in `Parameters.rst <./Parameters.rst>`__
is generated automatically from comments in the `config file <https://github.com/fabsig/GPBoost/blob/master/include/LightGBM/config.h>`__
by `this script <https://github.com/fabsig/GPBoost/blob/master/helpers/parameter_generator.py>`__.

After each commit on ``master``, the documentation is updated and published to `Read the Docs <https://gpboost.readthedocs.io/>`__.

Build
-----

You can build the documentation locally. Just install Doxygen and run in ``docs`` folder

.. code:: sh

    pip install -r requirements.txt
    make html
