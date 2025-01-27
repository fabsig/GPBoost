#############################################################
# Illustration of the usage of gpboost in comparison to lme4
#   The ChickWeight data is used to show the handling of
#   grouped random effects in gpboost. The following topics are covered:
#   - Categorical variables in fixed effects
#   - Nested random effects
#   - Random coefficients / slopes
# 
# Author: Fabio Sigrist
#############################################################

library(gpboost)
package_to_load <- "lme4" # load required package (non-standard way of loading to avoid CRAN warnings)
do.call(require,list(package_to_load, character.only=TRUE))

#################################
# Handling of categorical fixed effects variables
#################################

# Model with Diet and Time as categorical fixed effects variables and Chick as random effect

# lme4:
mod_lme4 <- lmer(weight ~ Diet + as.factor(Time) + (1 | Chick), data = ChickWeight, REML = FALSE)
summary(mod_lme4)

# For gpbooost, we first need to prepare the fixed effects design matrix, i.e., 
#   create dummy variables for categorical variables
fixed_effects_matrix <- model.matrix(weight ~ Diet + as.factor(Time), data = ChickWeight)
mod_gpb <- fitGPModel(X = fixed_effects_matrix, 
                      group_data = ChickWeight$Chick, 
                      y = ChickWeight$weight, params = list(std_dev = TRUE))
summary(mod_gpb)
# Alternative way:
mod_gpb <- GPModel(group_data = ChickWeight$Chick)
fit(mod_gpb, X = fixed_effects_matrix, y = ChickWeight$weight, params = list(std_dev = TRUE))


#################################
# Nested random effects
#################################

# A model with Time as categorical fixed effects variables and Diet and Chick
#   as random effects, where Chick is nested in Diet

# lme4:
mod_lme4 <-  lmer(weight ~ as.factor(Time) + (1 | Diet/Chick), data = ChickWeight, REML = FALSE)
summary(mod_lme4)

# For gpboost, we first need to create nested random effects variable "manually" 
# Note: you need gpboost version 0.7.9 or later to use the function 'get_nested_categories'
chick_nested_diet <- get_nested_categories(ChickWeight$Diet, ChickWeight$Chick)
fixed_effects_matrix <- model.matrix(weight ~ as.factor(Time), data = ChickWeight)
mod_gpb <- fitGPModel(X = fixed_effects_matrix, 
                      group_data = cbind(diet=ChickWeight$Diet, chick_nested_diet), 
                      y = ChickWeight$weight, params = list(std_dev = TRUE))
summary(mod_gpb)


#################################
# Random slopes / random coefficients
#################################

# Model with Diet and Time categorical fixed effects variables,  
#   Chick as random effect, and random slopes for Time

# lme4:
mod_lme4 <- lmer(weight ~ Diet + as.factor(Time) + (Time | Chick), data = ChickWeight, REML = FALSE)
summary(mod_lme4)

# gpboost:
fixed_effects_matrix <- model.matrix(weight ~ Diet + as.factor(Time), data = ChickWeight)
mod_gpb <- fitGPModel(X = fixed_effects_matrix, 
                      group_data = ChickWeight$Chick, 
                      group_rand_coef_data = ChickWeight$Time, ind_effect_group_rand_coef = c(1),
                      y = ChickWeight$weight, params = list(std_dev = TRUE))
summary(mod_gpb)
