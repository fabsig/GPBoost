Comments on computational efficiency for GPBoost
================================================

Estimation/training of random effects models and Gaussian process (GP) models, in particular, can be computationally demanding for large data (not just in GPBoost). Below, we list some strategies for computationally efficient inference.

Adjusting the way hyperparameters (*of random effects / GP models*) are chosen
----------------------------------------------------------------------------------

* It can be faster to set ``optimizer_cov`` to ``nelder_mead`` when having large data as this can lead to reduced computational cost and memory usage. For small data, however, setting ``optimizer_cov`` to ``gradient_descent`` or ``fisher_scoring`` (*the latter for Gaussian likelihoods only*) can lead to faster calculations compared to the option ``nelder_mead``. Examples:

   * ``gp_model$set_optim_params(params=list(optimizer_cov="nelder_mead"))`` (R) / ``gp_model.set_optim_params(params={"optimizer_cov": "nelder_mead"})`` (Python)

* Setting the parameter ``train_gp_model_cov_pars`` to false in the function ``gpb.train`` can also make training faster for the GPBoost algorithm as covariance parameters are not trained. In this case, you can consider them as tuning parameters that can be chosen using, e.g., cross-validation. As a middle ground, you can also increase the convergence tolerance for the hyperparameter estimation. This means that hyperparameters are estimated less accurately but faster. Examples:

   * ``gpb.train(..., train_gp_model_cov_pars=FALSE))`` (R) / ``gpb.train(..., train_gp_model_cov_pars=False)`` (Python)

   * ``gp_model$set_optim_params(params=list(delta_rel_conv=1e-3))`` (R) / ``gp_model.set_optim_params(params={"delta_rel_conv": 1e-3})`` (Python)

* To get a better understanding of the progress of the hyperparameter optimization, set the option ``trace`` to true in the ``gp_model``. Examples:

   * ``gp_model$set_optim_params(params=list(trace=TRUE))`` (R) / ``gp_model.set_optim_params(params={"trace": True})`` (Python)

Approximations for Gaussian processes
-------------------------------------

**In brief, try:**

* ``GPModel(..., gp_approx = "vecchia")`` or ``GPModel(..., gp_approx = "tapering")`` for Gaussian process models

* For ``GPModel(..., gp_approx = "vecchia", matrix_inversion_method = "iterative")`` for Gaussian processes with non-Gaussian likelihoods (e.g., classification)


**In more detail:**

* The GPBoost library implements Vecchia approximations for Gaussian processes. To activate this, set ``gp_approx = "vecchia"``. The parameter ``num_neighbors`` controls a trade-off between runtime and accuracy. The smaller the ``num_neighbors``, the faster the code will run.  See `here <http://arxiv.org/abs/2004.02653>`__ for more information on the methodological background.

   * For non-Gaussian likelihoods (e.g., classification), iterative methods (instead of the Cholesky decomposition) can additionally speed up computations with a Vecchia approximation: ``GPModel(..., gp_approx = "vecchia", matrix_inversion_method = "iterative")``

* Alternatively, covariance tapering (``gp_approx = "tapering"``) and compactly supported covariance functions (``cov_function = "wendland"``) can speed up computations and reduce memory usage. The taper range parameter ``cov_fct_taper_range`` controls the amount of sparsity in the covariance matrices and thus the computational cost and memory usage. See `here <https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-2/Estimation-and-prediction-using-generalized-Wendland-covariance-functions-under-fixed/10.1214/17-AOS1652.short>`__ for more information on the methodological background.

* Both the Vecchia approximation and covariance tapering are techniques that work well for low-dimensional input features (e.g., spatial data). For higher-dimensional input features, it is less clear how well they perform. Work on this is in progress.

* *Note: There exist various other possible strategies for GPs (low-rank approximations, basis function expansions, spectral methods, full-scale approximations, etc.) such that computations scale well to large data that are currently not implemented. Work on this is in progress. Contributions and suggestions are welcome.* 
