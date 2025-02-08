Comments on computational efficiency for GPBoost
================================================

Estimation/training of random effects models and Gaussian process (GP) models, in particular, can be computationally demanding for large data (not just in GPBoost). Below, we list some strategies for computationally efficient inference. 

* Note that GPBoost does automatic parallelization using OpenMP. By default, all available threads are used. However, for computers with many cores (e.g., computing clusters), it can be advantageous to limit the number of threads to, say, the number of physical cores or even less. This can be done using the ``num_parallel_threads`` argument of the ``GPModel()`` constructor.


Approximations for Gaussian processes
-------------------------------------

**In brief, try:**

* ``GPModel(..., gp_approx = "vecchia")`` for Gaussian process models

* ``GPModel(..., gp_approx = "vecchia", matrix_inversion_method = "iterative")`` for Gaussian processes with non-Gaussian likelihoods (e.g., classification)


**In more detail:**

* The GPBoost library implements Vecchia approximations for Gaussian processes. To activate this, set ``gp_approx = "vecchia"``. The parameter ``num_neighbors`` controls a trade-off between runtime and accuracy. The smaller the ``num_neighbors``, the faster the code will run. See `here <http://arxiv.org/abs/2004.02653>`__ for more information on the methodological background.

   * For non-Gaussian likelihoods (e.g., classification), iterative methods (instead of the Cholesky decomposition) can additionally speed-up computations with a Vecchia approximation: ``GPModel(..., gp_approx = "vecchia", matrix_inversion_method = "iterative")``.

   * Additional speed-up for ``matrix_inversion_method = "iterative"`` can sometimes be obtained by setting the ``cg_max_num_it`` and ``cg_max_num_it_tridiag`` parameters to lower values, say, 100 or 20 (default = 1000). This can be done by calling the ``set_optim_params`` function prior to running the GPBoost algorithm or by setting this in the ``params`` argument when training a GPModel. This option is particularly relevant for the GPBoost algorithm.

* The GPBoost library also imports other GP approximations; see `here <https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#model-specification-parameters>`__ (``gp_approx``) for a list of currently supported approximations


Adjusting the way hyperparameters (*of random effects / GP models*) are chosen
----------------------------------------------------------------------------------

* Setting the parameter ``train_gp_model_cov_pars`` to false in the function ``gpb.train`` can make training faster for the GPBoost algorithm as covariance parameters are not trained. In this case, you can consider them as tuning parameters that can be chosen using, e.g., cross-validation. As a middle ground, you can also increase the convergence tolerance for the hyperparameter estimation. This means that hyperparameters are estimated less accurately but faster. Examples:

   * ``gpb.train(..., train_gp_model_cov_pars=FALSE))`` (R) / ``gpb.train(..., train_gp_model_cov_pars=False)`` (Python)

   * ``gp_model$set_optim_params(params=list(delta_rel_conv=1e-3))`` (R) / ``gp_model.set_optim_params(params={"delta_rel_conv": 1e-3})`` (Python)

* To get a better understanding of the progress of the hyperparameter optimization, set the option ``trace`` to true in the ``gp_model``. Examples:

   * ``gp_model$set_optim_params(params=list(trace=TRUE))`` (R) / ``gp_model.set_optim_params(params={"trace": True})`` (Python)