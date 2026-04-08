# =============================================================================
# Module 2: Linear Models
# Application: Driver Acceptance Modeling and Proxy Discrimination
# =============================================================================
#
# We build a logistic regression model to predict whether a driver accepts a
# ride request. The data includes pickup neighborhood, time of day, trip
# distance, and rider rating. Race is NOT in the model — but it lurks in the
# neighborhood feature. Watch the disparate impact emerge.

library(tidyverse)
library(glmnet)
library(broom)

# =============================================================================
# PART 1: Generate Synthetic Driver-Acceptance Data
# =============================================================================

set.seed(2024)
n <- 5000

neighborhoods <- tibble(
  neighborhood   = c("Downtown", "Midtown", "Eastside", "Southside"),
  base_accept    = c(0.85, 0.78, 0.55, 0.40),  # true acceptance probability
  pct_minority   = c(0.20, 0.35, 0.70, 0.85),
  median_income  = c(95000, 72000, 38000, 29000)
)

requests <- tibble(
  request_id = 1:n,
  neighborhood = sample(
    neighborhoods$neighborhood, n, replace = TRUE,
    prob = c(0.35, 0.30, 0.20, 0.15)
  )
) |>
  left_join(neighborhoods, by = "neighborhood") |>
  mutate(
    hour = sample(0:23, n, replace = TRUE),
    is_night = as.integer(hour >= 22 | hour <= 5),
    trip_distance_mi = pmax(rlnorm(n, meanlog = 1.2, sdlog = 0.6), 0.3),
    rider_rating = pmin(pmax(rnorm(n, mean = 4.7, sd = 0.4), 1), 5),
    # True acceptance probability depends on:
    #   - neighborhood (the source of social bias)
    #   - time of night (drivers don't like night rides)
    #   - distance (drivers don't like very short or very long trips)
    #   - rating (drivers prefer high-rated riders)
    logit_p = qlogis(base_accept) +
              -0.6 * is_night +
              -0.15 * abs(trip_distance_mi - 4) +
              0.5 * (rider_rating - 4.7),
    accept_prob = plogis(logit_p),
    accepted = rbinom(n, size = 1, prob = accept_prob),
    # Racial group of the rider — correlated with neighborhood, but not used
    # as a model feature. We use it to AUDIT the model.
    is_minority = rbinom(n, size = 1, prob = pct_minority)
  ) |>
  select(-base_accept, -median_income, -pct_minority, -logit_p, -accept_prob)

# Quick look
cat("Acceptance rate by neighborhood:\n")
requests |>
  group_by(neighborhood) |>
  summarise(n = n(), accept_rate = mean(accepted), .groups = "drop") |>
  print()

cat("\nAcceptance rate by demographic group:\n")
requests |>
  group_by(is_minority) |>
  summarise(n = n(), accept_rate = mean(accepted), .groups = "drop") |>
  print()


# =============================================================================
# PART 2: Plain Logistic Regression
# =============================================================================

# We fit the model WITHOUT using is_minority as a feature.
# This is the "race-blind" approach that many platforms claim.

set.seed(123)
train_idx <- sample(1:n, 0.7 * n)
train <- requests[train_idx, ]
test  <- requests[-train_idx, ]

fit_logit <- glm(
  accepted ~ neighborhood + is_night + trip_distance_mi + rider_rating,
  data = train, family = binomial
)

cat("\n--- Plain Logistic Regression Coefficients ---\n")
print(tidy(fit_logit))

# Note the neighborhood coefficients: Southside and Eastside have large
# negative log-odds compared to Downtown (the reference). This is the model
# encoding "this area = lower acceptance" — a learned proxy for race.


# =============================================================================
# PART 3: Audit the "Race-Blind" Model
# =============================================================================

# The model never saw is_minority. Did it discriminate anyway?

test$pred_prob <- predict(fit_logit, test, type = "response")
test$pred_class <- as.integer(test$pred_prob >= 0.5)

cat("\n--- Predicted Acceptance Rate by Demographic Group ---\n")
test |>
  group_by(is_minority) |>
  summarise(
    n = n(),
    actual_accept_rate    = mean(accepted),
    predicted_accept_rate = mean(pred_prob),
    .groups = "drop"
  ) |>
  print()

# Even though race is not in the model, the predicted acceptance rate for
# minority riders is much lower than for non-minority riders. The neighborhood
# feature did the discriminating for us.


# =============================================================================
# PART 4: Regularization with Lasso
# =============================================================================

# Lasso will shrink some coefficients exactly to zero — automatic feature
# selection. Will it drop the neighborhood features? Probably not, because
# they're the strongest predictors.

# glmnet needs a model matrix
x_train <- model.matrix(
  accepted ~ neighborhood + is_night + trip_distance_mi + rider_rating,
  data = train
)[, -1]  # drop intercept; glmnet adds its own
y_train <- train$accepted

x_test <- model.matrix(
  accepted ~ neighborhood + is_night + trip_distance_mi + rider_rating,
  data = test
)[, -1]

set.seed(456)
cv_lasso <- cv.glmnet(
  x_train, y_train, family = "binomial", alpha = 1, nfolds = 10
)

cat("\n--- Lasso CV ---\n")
cat("lambda.min:", cv_lasso$lambda.min, "\n")
cat("lambda.1se:", cv_lasso$lambda.1se, "\n")

# Coefficients at lambda.1se
cat("\n--- Lasso Coefficients (lambda.1se) ---\n")
print(coef(cv_lasso, s = "lambda.1se"))

# Plot the coefficient path
plot(cv_lasso$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(cv_lasso$lambda.1se), lty = 2)
title("Lasso Coefficient Paths", line = 2.5)


# =============================================================================
# PART 5: Ridge and Elastic Net for Comparison
# =============================================================================

set.seed(456)
cv_ridge <- cv.glmnet(
  x_train, y_train, family = "binomial", alpha = 0, nfolds = 10
)

set.seed(456)
cv_enet <- cv.glmnet(
  x_train, y_train, family = "binomial", alpha = 0.5, nfolds = 10
)

cat("\n--- Ridge Coefficients (lambda.1se) ---\n")
print(coef(cv_ridge, s = "lambda.1se"))

cat("\n--- Elastic Net Coefficients (alpha=0.5, lambda.1se) ---\n")
print(coef(cv_enet, s = "lambda.1se"))


# =============================================================================
# PART 6: Compare Models on the Audit
# =============================================================================

# Generate predictions from each model
test$pred_lasso <- as.numeric(predict(
  cv_lasso, x_test, s = "lambda.1se", type = "response"
))
test$pred_ridge <- as.numeric(predict(
  cv_ridge, x_test, s = "lambda.1se", type = "response"
))
test$pred_enet <- as.numeric(predict(
  cv_enet, x_test, s = "lambda.1se", type = "response"
))

audit <- test |>
  group_by(is_minority) |>
  summarise(
    n            = n(),
    actual       = mean(accepted),
    plain_logit  = mean(pred_prob),
    lasso        = mean(pred_lasso),
    ridge        = mean(pred_ridge),
    elastic_net  = mean(pred_enet),
    .groups = "drop"
  )

cat("\n--- Audit: Predicted Acceptance Rate by Group, Across Models ---\n")
print(audit)

# Compute disparate impact ratio for each model
# (predicted acceptance rate of minority / non-minority — closer to 1 = fairer)
di_ratios <- audit |>
  pivot_longer(c(plain_logit, lasso, ridge, elastic_net),
               names_to = "model", values_to = "rate") |>
  pivot_wider(id_cols = model, names_from = is_minority,
              values_from = rate, names_prefix = "group_") |>
  mutate(disparate_impact_ratio = group_1 / group_0) |>
  arrange(desc(disparate_impact_ratio))

cat("\n--- Disparate Impact Ratios (1.0 = parity) ---\n")
print(di_ratios)


# =============================================================================
# PART 7: What If We Drop Neighborhood?
# =============================================================================

# Surely removing the obvious proxy will help, right? Let's see.

fit_no_nbhd <- glm(
  accepted ~ is_night + trip_distance_mi + rider_rating,
  data = train, family = binomial
)

test$pred_no_nbhd <- predict(fit_no_nbhd, test, type = "response")

cat("\n--- Model WITHOUT Neighborhood Feature ---\n")
test |>
  group_by(is_minority) |>
  summarise(
    n = n(),
    actual = mean(accepted),
    predicted = mean(pred_no_nbhd),
    .groups = "drop"
  ) |>
  print()

# The disparate impact shrinks dramatically! But so does the model's accuracy.
# This is the accuracy-fairness tradeoff in action.

cat("\n--- Test Accuracy by Model ---\n")
cat("With neighborhood:    ",
    round(mean((test$pred_prob >= 0.5) == test$accepted), 3), "\n")
cat("Without neighborhood: ",
    round(mean((test$pred_no_nbhd >= 0.5) == test$accepted), 3), "\n")


# =============================================================================
# PART 8: Visualizing the Tradeoff
# =============================================================================

ggplot(audit |>
       pivot_longer(c(plain_logit, lasso, ridge, elastic_net),
                    names_to = "model", values_to = "predicted_rate"),
       aes(x = model, y = predicted_rate, fill = factor(is_minority))) +
  geom_col(position = "dodge") +
  scale_fill_manual(
    values = c("0" = "steelblue", "1" = "firebrick"),
    labels = c("Non-minority", "Minority"),
    name = "Group"
  ) +
  labs(
    title = "Predicted Acceptance Rate by Group, Across Models",
    subtitle = "Regularization doesn't fix discrimination — the proxies are still doing the work",
    x = "Model", y = "Predicted Acceptance Rate"
  ) +
  theme_minimal()


cat("\n")
cat("Key takeaways:\n")
cat("1. Removing 'race' from the features doesn't make the model fair —\n")
cat("   neighborhood acts as a proxy.\n")
cat("2. Lasso, Ridge, and Elastic Net all keep neighborhood (it's predictive).\n")
cat("3. Regularization is about overfitting, NOT about fairness.\n")
cat("4. Dropping neighborhood reduces disparate impact but also accuracy.\n")
cat("5. Fairness requires explicit fairness criteria (Module 11), not just\n")
cat("   feature engineering.\n")
