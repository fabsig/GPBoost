.. role:: raw-html(raw)
    :format: html

Main parameters for GPBoost
===========================

This page contains a summary of the most important parameters. We distinguish between the following two types of parameters:

- (Tuning-) parameters for the tree-boosting part 
- Model specification and optimization parameters for the Gaussian process and random effects part

**Currently, the GPBoost library supports the following likelihoods**: ``gaussian``, ``bernoulli_probit`` (= ``binary``), ``bernoulli_logit``, ``poisson``, ``gamma``. This distribution of the data can be specified through the ``likelihood`` parameter for the random effects model and the ``objective`` paramter for the tree-boosting part.

.. contents:: **Contents**
    :depth: 2
    :local:
    :backlinks: none

Important tree-boosting parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a list of important parameters for the tree-boosting part. `A comprehensive list of all tree-bosting related parameters can be found here <https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst>`_.

-  ``num_iterations`` :raw-html:`<a id="num_iterations" title="Permalink to this parameter" href="#num_iterations">&#x1F517;&#xFE0E;</a>`, default = ``100``, type = int, aliases: ``num_iteration``, ``n_iter``, ``num_tree``, ``num_trees``, ``num_round``, ``num_rounds``, ``num_boost_round``, ``n_estimators``, constraints: ``num_iterations >= 0``

   -  number of boosting iterations

   -  this is arguably the most important tuning parameter, in particular for regession settings

-  ``learning_rate`` :raw-html:`<a id="learning_rate" title="Permalink to this parameter" href="#learning_rate">&#x1F517;&#xFE0E;</a>`, default = ``0.1``, type = double, aliases: ``shrinkage_rate``, ``eta``, constraints: ``learning_rate > 0.0``

   -  shrinkage rate or damping parameter

   -  smaller values lead to higher predictive accuracy but require more computational time since more boosting iterations are needed

-  ``max_depth`` :raw-html:`<a id="max_depth" title="Permalink to this parameter" href="#max_depth">&#x1F517;&#xFE0E;</a>`, default = ``-1``, type = int

   -  maximal depth of a tree

   -  ``<= 0`` means no limit

-  ``min_data_in_leaf`` :raw-html:`<a id="min_data_in_leaf" title="Permalink to this parameter" href="#min_data_in_leaf">&#x1F517;&#xFE0E;</a>`, default = ``20``, type = int, aliases: ``min_data_per_leaf``, ``min_data``, ``min_child_samples``, constraints: ``min_data_in_leaf >= 0``

   -  minimal number of samples in a leaf

-  ``num_leaves`` :raw-html:`<a id="num_leaves" title="Permalink to this parameter" href="#num_leaves">&#x1F517;&#xFE0E;</a>`, default = ``31``, type = int, aliases: ``num_leaf``, ``max_leaves`` ``max_leaf``, constraints: ``1 < num_leaves <= 131072``

   -  maximal number of leaves of a tree

-  ``train_gp_model_cov_pars`` :raw-html:`<a id="train_gp_model_cov_pars" title="Permalink to this parameter" href="#train_gp_model_cov_pars">&#x1F517;&#xFE0E;</a>`, default = ``true``, type = bool

   -  if ``true``, the covariance parameters of the Gaussian process / random effects model are trained (estimated) in every boosting iteration of the GPBoost algorithm, otherwise not

-  ``use_gp_model_for_validation`` :raw-html:`<a id="use_gp_model_for_validation" title="Permalink to this parameter" href="#use_gp_model_for_validation">&#x1F517;&#xFE0E;</a>`, default = ``true``, type = bool

   -  set this to ``true`` to also use the Gaussian process / random effects model (in addition to the tree model) for calculating predictions on the validation data when using the GPBoost algorithm

-  ``leaves_newton_update`` :raw-html:`<a id="leaves_newton_update" title="Permalink to this parameter" href="#leaves_newton_update">&#x1F517;&#xFE0E;</a>`, default = ``false``, type = bool

   -  if ``true``, a Newton update step is done for the tree leaves after the gradient step

   -  applies only to the GPBoost algorithm for Gaussian data and cannot be used for non-Gaussian data

-  ``lambda_l1`` :raw-html:`<a id="lambda_l1" title="Permalink to this parameter" href="#lambda_l1">&#x1F517;&#xFE0E;</a>`, default = ``0.0``, type = double, aliases: ``reg_alpha``, constraints: ``lambda_l1 >= 0.0``

   -  L1 regularization

-  ``lambda_l2`` :raw-html:`<a id="lambda_l2" title="Permalink to this parameter" href="#lambda_l2">&#x1F517;&#xFE0E;</a>`, default = ``0.0``, type = double, aliases: ``reg_lambda``, ``lambda``, constraints: ``lambda_l2 >= 0.0``

   -  L2 regularization

-  ``max_bin`` :raw-html:`<a id="max_bin" title="Permalink to this parameter" href="#max_bin">&#x1F517;&#xFE0E;</a>`, default = ``255``, type = int, constraints: ``max_bin > 1``

   -  maximal number of bins that feature values will be bucketed in

   -  GPBoost uses histogram-based algorithms `[1, 2, 3] <#references>`__, which bucket continuous feature (covariate) values into discrete bins. A small number speeds up training and reduces memory usage but may reduce the accuracy of the model

-  ``min_gain_to_split`` :raw-html:`<a id="min_gain_to_split" title="Permalink to this parameter" href="#min_gain_to_split">&#x1F517;&#xFE0E;</a>`, default = ``0.0``, type = double, aliases: ``min_split_gain``, constraints: ``min_gain_to_split >= 0.0``

   -  the minimal gain to perform split

-  ``objective`` :raw-html:`<a id="objective" title="Permalink to this parameter" href="#objective">&#x1F517;&#xFE0E;</a>`, default = ``regression_l2``, type = enum, options: ``regression_l2``, ``binary``, ``poisson``, ``gamma``, aliases: ``objective_type``, ``app``, ``application``

   -  ``regression_l2``, Gaussian likelihood, aliases: ``regression``, ``l2``

   -  ``binary``, Bernoulli likelihood with a logit link function

   -  ``poisson``, Poisson likelihood with log link function

   -  ``gamma``, Gamma likelihood with log link function

   - When including random effects and Gaussian processes (i.e., a ``GPModel``), both the ``likelihood`` parameter of the ``GPModel`` and the ``objective`` parameter determine the likelihood. They should thus match. It is possible to provide only one of these two parameters, i.e., either ``objective`` or ``likelihood``. *It is, arguably, a potential disadvantage that two parameters determine the likelihood. But this is the case because both tree-boosting and (linear) random effects and GP models can also be used in stand-alone mode*

-  ``boosting`` :raw-html:`<a id="boosting" title="Permalink to this parameter" href="#boosting">&#x1F517;&#xFE0E;</a>`, default = ``gbdt``, type = enum, options: ``gbdt``, aliases: ``boosting_type``, ``boost``

   -  ``gbdt``, traditional tree-boosting, aliases: ``gbrt``

   - Only the option ``gbdt`` is currently supported when including random effects and Gaussian processes

*Note on ``max_depth`` and ``num_leaves`` parameters*: GPBoost uses the LightGBM tree growing algorithm which grows trees using a leaf-wise strategy. I.e., trees are grown by splitting leaf nodes first that maximize the information gain until the maximal number of leaves ``num_leaves`` or the maximal depth of a tree ``max_depth`` is attained, even when this leads to unbalanced trees. This in contrast to a depth-wise growth strategy of other boosting implementations which builds more balanced trees. For shallow trees (=small ``max_depth``), there is likely no difference between these two tree growing strategies. If you only want to tune the maximal depth of a tree ``max_depth`` parameter and not the ``num_leaves`` parameter, it is recommended that you set the ``num_leaves`` parameter to a large value.

..
    Categorical features
    --------------------

    The tree building algorithm of GPBoost (i.e. the LightGBM tree building algorithm) can use categorical features directly (without one-hot encoding). It is common to represent categorical features with one-hot encoding, but this approach is suboptimal for tree learners. Particularly for high-cardinality categorical features, a tree built on one-hot features tends to be unbalanced and needs to grow very deep to achieve good accuracy.

    Instead of one-hot encoding, the optimal solution is to split on a categorical feature by partitioning its categories into 2 subsets. If the feature has ``k`` categories, there are ``2^(k-1) - 1`` possible partitions.
    But there is an efficient solution for regression trees `Fisher (1958) <http://www.csiss.org/SPACE/workshops/2004/SAC/files/fisher.pdf>`_. It needs about ``O(k * log(k))`` to find the optimal partition.
    The basic idea is to sort the categories according to the training objective at each split.

    For further details on using categorical features, please refer to the ``categorical_feature`` `parameter <./Parameters.rst#categorical_feature>`__.


Gaussian process and random effects parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a list of parameters for specifying ``GPModel`` objects for modeling Gaussian processes and grouped random effects
and for specifying how these models are trained. These parameters are documented in a generic manner in the form they are
used in the R and Python package. The C API works slightly different.

Model specification parameters
------------------------------

-  ``likelihood`` : string, (default = ``gaussian``)

   -  Likelihood function of the response variable = distribution of the label variable

   -  Currently supported values: ``gaussian``, ``bernoulli_probit`` (= ``binary``), ``bernoulli_logit``, ``poisson``, ``gamma``

-  ``group_data`` : two dimensional array / matrix of doubles or strings, optional (default = None)

   -  Labels of group levels for grouped random effects

-  ``group_rand_coef_data`` : two dimensional array / matrix of doubles or None, optional (default = None)

   -  Covariate data for grouped random coefficients

-  ``ind_effect_group_rand_coef`` : integer vector / array of integers or None, optional (default = None)

   -  Indices that relate every random coefficients to a "base" intercept grouped random effect. Counting starts at 1.

-  ``gp_coords`` : two dimensional array / matrix of doubles or None, optional (default = None)

   -  Coordinates (features) for Gaussian process

-  ``gp_rand_coef_data`` : two dimensional array / matrix of doubles or None, optional (default = None)

   -  Covariate data for Gaussian process random coefficients

-  ``cov_function`` : string, (default = ``exponential``)

   -  Covariance function for the Gaussian process. Available options: ``exponential``, ``gaussian``, ``matern``, ``powered_exponential``, ``wendland`` 

      - For ``exponential``, ``gaussian``, and ``powered_exponential``, we use the parametrization of Diggle and Ribeiro (2007) 

      - For ``matern``, we use the parametrization of Rasmussen and Williams (2006) 

      - For ``wendland``, we use the parametrization of Bevilacqua et al. (2019, AOS)

-  ``cov_fct_shape`` : double, (default = 0.5)

   -  Shape parameter of the covariance function (=smoothness parameter for Matern covariance). This parameter is irrelevant for some covariance functions such as the exponential or Gaussian.

-  ``gp_approx`` : string, (default = ``none``)

   -  Specifies the use of a large data approximation for Gaussian processes. Available options:

      - ``none`` : No approximation

      - ``vecchia`` : A Vecchia approximation; see Sigrist (2022, JMLR for more details)

      - ``tapering`` : The covariance function is multiplied by a compactly supported Wendland correlation function

-  ``cov_fct_taper_range`` : double, (default = 1.)

   -  Range parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)

-  ``cov_fct_taper_shape`` : double, (default = 0.)

   -  Shape parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)

-  ``num_neighbors`` : integer, (default = 20)

   -  Number of neighbors for the Vecchia approximation

-  ``vecchia_ordering`` : string, (default = ``random``)

   -  Ordering used in the Vecchia approximation. Available options: 

      - ``none``: the default ordering in the data is used

      - ``random``: a random ordering

-  ``vecchia_pred_type`` : string, (default = Null)

   -  Type of Vecchia approximation used for making predictions

   - Default value if ``vecchia_pred_type`` = Null : ``order_obs_first_cond_obs_only``

   - Available options:

      -  ``order_obs_first_cond_obs_only`` : observed data is ordered first and the neighbors are only observed points

      - ``order_obs_first_cond_all`` : observed data is ordered first and the neighbors are selected among all points (observed + predicted)

      - ``latent_order_obs_first_cond_obs_only`` : Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points

      - ``latent_order_obs_first_cond_all`` : Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points

      - ``order_pred_first`` : predicted data is ordered first for making predictions. This option is only available for Gaussian likelihoods

-  ``num_neighbors_pred`` : integer, (default = Null)

   - Number of neighbors for the Vecchia approximation for making predictions. 

   - Default value if ``num_neighbors_pred`` = Null: ``num_neighbors_pred`` = 2 * ``num_neighbors``

-  ``num_ind_points`` : integer, (default = 500)

   -  Number of inducing points / knots for, e.g., a predictive process approximation

-  ``matrix_inversion_method`` : string, (default = ``cholesky``)

   -  Method used for inverting covariance matrices. Available options:

      -  ``cholesky`` : Cholesky factorization

-  ``seed`` : integer, (default = 0)

   -  The seed used for model creation (e.g., random ordering in Vecchia approximation)

-  ``cluster_ids`` : one dimensional numpy array (vector) with integer data or Null, (default = Null)

   -  IDs / labels indicating independent realizations of random effects / Gaussian processes (same values = same process realization)


Optimization parameters
-----------------------

The following list shows options for the optimization of the variance and covariance parameters of ``gp_model`` objects which contain Gaussian process and/or grouped random effects models. These parameters are passed to either the ``fit`` function of a ``gp_model`` object in Python and R or to the ``set_optim_params`` function prior to running the GPBoost algorithm.

-  ``optimizer_cov`` : string, optional (default = ``gradient_descent``)

   -  Optimizer used for estimating covariance parameters

   -  Options: ``gradient_descent``, ``fisher_scoring``, ``nelder_mead``, ``bfgs``, ``adam``

-  ``optimizer_coef`` : string, optional (default = ``wls`` for Gaussian data and ``gradient_descent`` for other likelihoods)

   -  Optimizer used for estimating linear regression coefficients, if there are any (for the GPBoost algorithm there are usually none)

   -  Options: ``gradient_descent``, ``wls``, ``nelder_mead``, ``bfgs``, ``adam``. Gradient descent steps are done simultaneously with gradient descent steps for the covariance paramters. ``wls`` refers to doing coordinate descent for the regression coefficients using weighted least squares

   -  If ``optimizer_cov`` is set to ``nelder_mead``, ``bfgs``, or ``adam``, ``optimizer_coef`` is automatically also set to the same value

-  ``maxit`` : integer, optional (default = 1000)

   -  Maximal number of iterations for optimization algorithm

-  ``delta_rel_conv`` : double, optional (default = 1e-6 except for ``nelder_mead`` for which the default is 1e-8)

   -  Convergence tolerance. The algorithm stops if the relative change in eiher the (approximate) log-likelihood or the parameters is below this value. For ``bfgs`` and ``adam``, the L2 norm of the gradient is used instead of the relative change in the log-likelihood

   -  If < 0, internal default values are used (= 1e-6 except for ``nelder_mead`` for which the default is 1e-8)

-  ``convergence_criterion`` : string, optional (default = ``relative_change_in_log_likelihood``)

   -  The convergence criterion used for terminating the optimization algorithm. Options: ``relative_change_in_log_likelihood`` or ``relative_change_in_parameters``

-  ``init_cov_pars`` : numeric vector / array of doubles, optional (default = Null)

   -  Initial values for covariance parameters of Gaussian process and random effects (can be Null)

-  ``init_coef`` : numeric vector / array of doubles, optional (default = Null)

   -  Initial values for the regression coefficients (if there are any, can be Null)

-  ``lr_cov`` : double, optional (default = 0.1 for ``gradient_descent`` and 1. for ``fisher_scoring``)

   -  Learning rate for covariance parameters
   
   -  If < 0, internal default values are used (0.1 for ``gradient_descent`` and 1. for ``fisher_scoring``)

-  ``lr_coef`` : double, optional (default = 0.1)

   -  Learning rate for fixed effect regression coefficients

-  ``use_nesterov_acc`` : bool, optional (default = True)

   -  If True Nesterov acceleration is used (only for gradient descent)

-  ``acc_rate_cov`` : double, optional (default = 0.5)

   -  Acceleration rate for covariance parameters for Nesterov acceleration

-  ``acc_rate_coef`` : double, optional (default = 0.5)

   -  Acceleration rate for coefficients for Nesterov acceleration

-  ``momentum_offset`` : integer, optional (default = 2)

   -  Number of iterations for which no mometum is applied in the beginning

-  ``trace`` : bool, optional (default = False)

   -  If True, information on the progress of the parameter optimization is printed.

-  ``std_dev`` : bool, optional (default = False)

   -  If True, (asymptotic) standard deviations are calculated for the covariance parameters