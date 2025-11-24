Comments on computational efficiency
====================================

Estimation and prediction with random effects models and Gaussian process (GP) models can be computationally demanding for large data (not just in GPBoost). Below, we list some strategies for computational efficiency. 

* **Gaussian process approximations**: The `GPBoost library implements several scalable GP approximations <https://github.com/fabsig/GPBoost/blob/master/docs/Main_parameters.rst#model-specification-parameters>`__ which can be enabled via the ``gp_approx`` argument.

   * In general, we recommend **Vecchia approximations** (``gp_approx = "vecchia"``). The parameter ``num_neighbors`` controls a trade-off between runtime and accuracy (smaller = faster). See `here <http://arxiv.org/abs/2004.02653>`__ for more information on the methodological background.

   * For higher-dimensional inputs (say > 10), `VIF (Vecchia\-Inducing\-Points Full\-Scale) approximations <https://arxiv.org/abs/2507.05064>`__ (``gp_approx = "vif"``) can be more accurate. The parameters ``num_neighbors`` and ``num_ind_points`` control a trade-off between runtime and accuracy (smaller = faster).

* **Iterative methods** (instead of the Cholesky decomposition) can additionally speed-up computations. These are activated by default when possible. Additional speed-ups can be obtained by **setting the ``cg_max_num_it`` and ``cg_max_num_it_tridiag`` parameters to lower values**, say, 100 or 10 (default = 1000). This can be done by calling the ``set_optim_params`` function prior to running the GPBoost algorithm or by setting this in the ``params`` argument when calling the ``fit`` function of a GPModel. This option is particularly relevant for the GPBoost algorithm.

   * ``gp_model$set_optim_params(params=list(cg_max_num_it = 10, cg_max_num_it_tridiag = 10))`` (R)
   * ``gp_model.set_optim_params(params={"cg_max_num_it": 10, "cg_max_num_it_tridiag": 10})`` (Python)
   * Do some sensitivity checks (try multiple values) to make sure that this does not distort your results.

* GPBoost does automatic parallelization using OpenMP. By default, all available threads are used. However, **for computers with many cores (e.g., computing clusters), it can be advantageous to limit the number of OpenMP threads** to, say, the number of physical cores or even less. This can be done using the ``num_parallel_threads`` argument of the ``GPModel()`` constructor.

* **Disabling hyperparameter parameter estimation in the GPBoost algorithm** can make training faster. In this case, you should consider them as tuning parameters that are chosen using, e.g., cross-validation. 

   * ``gpb.train(..., train_gp_model_cov_pars=FALSE)`` (R)
   * ``gpb.train(..., train_gp_model_cov_pars=False)`` (Python)

* You can also **increase the convergence tolerance for the hyperparameter estimation**. This means that hyperparameters are estimated less accurately but faster.

   * ``gp_model$set_optim_params(params=list(delta_rel_conv=1e-3))`` (R)
   * ``gp_model.set_optim_params(params={"delta_rel_conv": 1e-3})`` (Python)
   * Do some sensitivity checks (try multiple values) to make sure that this does not distort your results.

* To get a better understanding of the progress of the hyperparameter optimization, set the option ``trace`` to true in the ``gp_model``.

   * ``gp_model$set_optim_params(params=list(trace=TRUE))`` (R)
   * ``gp_model.set_optim_params(params={"trace": True})`` (Python)