Comments on computational efficiency for GPBoost
================================================

Training of random effects models and Gaussian process models, in particular, is computational demanding, and this issue translates to the GPBoost algorithm. In the following, a few strategies for computationally efficient inference are presented.

* In general, it **is recommended to set** ``optimizer_cov`` **to "nelder_mead" when having large data** as this often leads to reduced computational cost and memory usage
* For small data, however, setting ``optimizer_cov`` to "gradient_descent" or "fisher_scoring" (Gaussian data only) can lead to faster calculations compared to the option "nelder_mead".
* For **Gaussian process models, the compactly supported covariance functions** (``cov_function``) "wendland" and "exponential_tapered" can speed up computations and reduce memory usage. The taper range parameter ``cov_fct_taper_range`` controls the amount of sparsity in the covariance matrices and thus the computational cost and memory usage. See `here <https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-2/Estimation-and-prediction-using-generalized-Wendland-covariance-functions-under-fixed/10.1214/17-AOS1652.short>`__ for more information on the methodological background.
* The GPBoost library also implements a **Vecchia approximation for Gaussian processes**. However, this option is only recommended for Gaussian data. To activate this, set ``vecchiac_approx`` to true and choose the ``num_neighbors`` parameter. The smaller the ``num_neighbors``, the better the computational efficiency. Depending on the amount of data and the chosen number of neighbors, the current implementation can require a lot of memory and your computer might start to do memory swapping (which is not necessarily an issue). However, this could be changed as this is mostly a software engineering issue (-> open a GitHub issue). See `here <http://arxiv.org/abs/2004.02653>`__ for more information on the methodological background.
* Setting the parameter ``train_gp_model_cov_pars`` to false can also make training faster as covariance parameters are not trained. As an alternative to training them, you can consider them as tuning parameters which can be chosen using, e.g., cross-validation.
* *For Gaussian processes, there exist various other possible strategies (low-rank approximations, basis function expansions, spectral methods, full scale approximations, etc.) such that computations scale well to large data which are currently not implemented. Contributions and suggestions are welcome.* 



