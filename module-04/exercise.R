# ============================================================================
# Module 4 Exercise: Tree-Based Methods for Surge Pricing
# ============================================================================
#
# You'll build and compare decision tree, random forest, and XGBoost models
# for predicting surge multipliers, then audit them for neighborhood bias
# using SHAP values.
#
# Instructions: work through each section, running the code and answering
# the questions in comments. Fill in the blanks marked with _____.

library(tidyverse)
library(tidymodels)
library(rpart)
library(rpart.plot)
library(ranger)
library(xgboost)
set.seed(42)

# --- 1. Simulate data -------------------------------------------------------
# Richer version of the slides data: 5 neighborhoods with explicit
# demographic correlations (income, % minority — not used as features,
# but available for auditing).

n <- 3000
neighborhoods <- tibble(
  neighborhood = c("Downtown", "Midtown", "Uptown", "Southside", "Westend"),
  base_demand  = c(1.8, 1.4, 1.1, 1.6, 0.9),
  base_drivers = c(12, 8, 8, 4, 5),
  median_income = c(85000, 72000, 60000, 38000, 45000),
  pct_minority  = c(0.25, 0.30, 0.35, 0.70, 0.55)
)

surge <- tibble(
  hour = sample(0:23, n, replace = TRUE),
  day_of_week = sample(1:7, n, replace = TRUE),
  neighborhood = sample(neighborhoods$neighborhood, n, replace = TRUE)
) |>
  left_join(neighborhoods, by = "neighborhood") |>
  mutate(
    demand_ratio = rnorm(n, mean = base_demand, sd = 0.3),
    drivers_nearby = rpois(n, lambda = base_drivers),
    is_weekend = day_of_week %in% c(6, 7),
    is_rush = hour %in% c(7:9, 17:19),
    surge_mult = 1 + 0.4 * demand_ratio - 0.05 * drivers_nearby +
      0.3 * is_rush + 0.15 * is_weekend + rnorm(n, sd = 0.15),
    surge_mult = pmax(surge_mult, 1.0)
  )

# Quick look at the data
glimpse(surge)

# Q1: What is the mean surge multiplier by neighborhood? Which neighborhood
#     pays the most? Does that correlate with income or % minority?
surge |>
  group_by(neighborhood) |>
  summarise(
    mean_surge = mean(surge_mult),
    mean_income = first(median_income),
    mean_minority = first(pct_minority)
  ) |>
  arrange(desc(mean_surge))

# --- 2. Train/test split ----------------------------------------------------

surge_split <- initial_split(surge, prop = 0.8, strata = neighborhood)
surge_train <- training(surge_split)
surge_test  <- testing(surge_split)

# Features for modeling (exclude demographic vars and neighborhood metadata)
model_features <- c("hour", "demand_ratio", "drivers_nearby",
                     "neighborhood", "is_weekend", "is_rush")

# --- 3. Decision tree --------------------------------------------------------

# Fit a single tree with maxdepth = 4
tree_fit <- rpart(
  surge_mult ~ hour + demand_ratio + drivers_nearby +
    neighborhood + is_weekend + is_rush,
  data = surge_train,
  control = rpart.control(maxdepth = 4)
)

# Visualize
rpart.plot(tree_fit, roundint = FALSE, digits = 3)

# Q2: What is the first split? Why does that make sense given the data
#     generating process?

# Predict on test set
tree_preds <- predict(tree_fit, newdata = surge_test)
tree_rmse <- sqrt(mean((surge_test$surge_mult - tree_preds)^2))
cat("Decision tree RMSE:", round(tree_rmse, 4), "\n")

# --- 4. Random forest --------------------------------------------------------

# Q3: Fill in the blanks to fit a random forest with 500 trees and mtry = 2
rf_fit <- ranger(
  surge_mult ~ hour + demand_ratio + drivers_nearby +
    neighborhood + is_weekend + is_rush,
  data = surge_train,
  num.trees = _____,
  mtry = _____,
  min.node.size = 5,
  importance = "impurity"
)

rf_preds <- predict(rf_fit, data = surge_test)$predictions
rf_rmse <- sqrt(mean((surge_test$surge_mult - rf_preds)^2))
cat("Random Forest RMSE:", round(rf_rmse, 4), "\n")

# Q4: What is the variable importance ranking? Does it match the true
#     data generating process?
tibble(
  variable = names(rf_fit$variable.importance),
  importance = rf_fit$variable.importance
) |>
  arrange(desc(importance))

# --- 5. Tune random forest with tidymodels -----------------------------------

# Recipe
rf_recipe <- recipe(
  surge_mult ~ hour + demand_ratio + drivers_nearby +
    neighborhood + is_weekend + is_rush,
  data = surge_train
) |>
  step_dummy(all_nominal_predictors())

# Model spec with tunable hyperparameters
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) |>
  set_engine("ranger", importance = "impurity") |>
  set_mode("regression")

rf_wf <- workflow() |>
  add_recipe(rf_recipe) |>
  add_model(rf_spec)

# Cross-validation folds
folds <- vfold_cv(surge_train, v = 5)

# Tune grid
rf_grid <- grid_regular(
  mtry(range = c(2, 6)),
  min_n(range = c(5, 25)),
  levels = 4
)

# Q5: How many total models will be fit? (grid size x folds)
cat("Grid size:", nrow(rf_grid), "x 5 folds =", nrow(rf_grid) * 5, "fits\n")

rf_tune_res <- tune_grid(rf_wf, resamples = folds, grid = rf_grid)

# Best hyperparameters
show_best(rf_tune_res, metric = "rmse", n = 5)
rf_best <- select_best(rf_tune_res, metric = "rmse")
cat("Best mtry:", rf_best$mtry, "  Best min_n:", rf_best$min_n, "\n")

# --- 6. XGBoost --------------------------------------------------------------

# Prepare numeric matrix (xgboost doesn't handle factors)
xgb_recipe <- recipe(
  surge_mult ~ hour + demand_ratio + drivers_nearby +
    neighborhood + is_weekend + is_rush,
  data = surge_train
) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune()
) |>
  set_engine("xgboost", verbosity = 0) |>
  set_mode("regression")

xgb_wf <- workflow() |>
  add_recipe(xgb_recipe) |>
  add_model(xgb_spec)

# Q6: Fill in the grid ranges. Start with:
#     trees: 100 to 500, tree_depth: 1 to 6, learn_rate: 0.01 to 0.3
xgb_grid <- grid_regular(
  trees(range = c(_____, _____)),
  tree_depth(range = c(_____, _____)),
  learn_rate(range = c(-2, log10(0.3)), trans = log10_trans()),
  levels = 3
)

xgb_tune_res <- tune_grid(xgb_wf, resamples = folds, grid = xgb_grid)

show_best(xgb_tune_res, metric = "rmse", n = 5)
xgb_best <- select_best(xgb_tune_res, metric = "rmse")

# --- 7. Compare models -------------------------------------------------------

# Finalize the best XGBoost model and fit on full training data
xgb_final_wf <- finalize_workflow(xgb_wf, xgb_best)
xgb_final_fit <- fit(xgb_final_wf, data = surge_train)

# Q7: Compare RMSE across all three models on the test set
cat("Decision Tree RMSE:", round(tree_rmse, 4), "\n")
cat("Random Forest RMSE:", round(rf_rmse, 4), "\n")

xgb_test_preds <- predict(xgb_final_fit, new_data = surge_test)$.pred
xgb_rmse <- sqrt(mean((surge_test$surge_mult - xgb_test_preds)^2))
cat("XGBoost RMSE:", round(xgb_rmse, 4), "\n")

# Q8: Which model wins? By how much? Is the improvement worth the
#     additional complexity?

# --- 8. SHAP audit -----------------------------------------------------------

# Extract the raw xgboost model object from the tidymodels workflow
xgb_raw <- extract_fit_parsnip(xgb_final_fit)$fit

# Prepare test data as a matrix
xgb_test_baked <- bake(extract_recipe(xgb_final_fit, estimated = TRUE),
                        new_data = surge_test) |>
  select(-surge_mult) |>
  as.matrix()

# Compute SHAP values
shap_values <- predict(xgb_raw, newdata = xgb_test_baked, predcontrib = TRUE)
shap_df <- as_tibble(shap_values[, -ncol(shap_values)]) |>
  set_names(colnames(xgb_test_baked))

# Mean absolute SHAP per feature
shap_importance <- shap_df |>
  summarise(across(everything(), ~ mean(abs(.)))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "mean_abs_shap") |>
  arrange(desc(mean_abs_shap))

print(shap_importance, n = 15)

# Q9: Do any neighborhood dummy variables appear in the top SHAP features?
#     If so, that means the model is using neighborhood *beyond* what
#     demand_ratio and drivers_nearby already capture.

# Plot SHAP for Southside specifically
southside_idx <- which(surge_test$neighborhood == "Southside")
southside_shap <- shap_df[southside_idx, ] |>
  summarise(across(everything(), mean)) |>
  pivot_longer(everything(), names_to = "feature", values_to = "mean_shap") |>
  filter(abs(mean_shap) > 0.001) |>
  mutate(feature = fct_reorder(feature, mean_shap))

ggplot(southside_shap, aes(mean_shap, feature,
                            fill = mean_shap > 0)) +
  geom_col(show.legend = FALSE) +
  scale_fill_manual(values = c("TRUE" = "firebrick", "FALSE" = "steelblue")) +
  labs(title = "Mean SHAP Values for Southside Rides",
       subtitle = "Positive = pushes surge UP for this neighborhood",
       x = "Mean SHAP value", y = NULL)

# --- 9. Audit summary -------------------------------------------------------

# Q10: Write a 3-sentence audit summary:
# 1. Which model performs best and by how much?
# 2. Is neighborhood driving predictions beyond supply/demand factors?
# 3. What would you recommend: keep neighborhood as a feature, drop it,
#    or something else?

# Your answer:
# 1. _____
# 2. _____
# 3. _____
