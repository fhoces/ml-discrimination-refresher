# =============================================================================
# Module 1: The Learning Problem
# Application: Disparate Wait Times in Ride-Sharing
# =============================================================================

library(tidyverse)

# =============================================================================
# PART 1: Generate Synthetic Ride-Sharing Data
# =============================================================================

# We simulate ride requests across 4 neighborhoods that differ in:
# - driver supply (fewer drivers in underserved areas)
# - demand patterns
# - demographics (which we'll use to check for disparate impact)

set.seed(42)

n <- 2000

neighborhoods <- tibble(
  neighborhood = c("Downtown", "Midtown", "Eastside", "Southside"),
  base_wait    = c(3, 4, 7, 9),        # true avg wait (minutes)
  driver_supply = c(50, 35, 15, 10),    # relative driver density
  pct_minority = c(0.20, 0.35, 0.70, 0.85),  # demographic composition
  median_income = c(95000, 72000, 38000, 29000)
)

rides <- tibble(
  ride_id = 1:n,
  neighborhood = sample(
    neighborhoods$neighborhood, n, replace = TRUE,
    prob = c(0.35, 0.30, 0.20, 0.15)  # more rides requested downtown
  )
) |>
  left_join(neighborhoods, by = "neighborhood") |>
  mutate(
    hour = sample(0:23, n, replace = TRUE),
    # Wait time depends on neighborhood base + time of day + noise
    # Peak hours (7-9am, 5-7pm) increase wait, especially in underserved areas
    is_peak = as.integer(hour %in% c(7, 8, 9, 17, 18, 19)),
    peak_penalty = is_peak * (10 - driver_supply / 10),
    # True wait time (what we're trying to predict)
    wait_time = base_wait + peak_penalty + rnorm(n, 0, 1.5),
    wait_time = pmax(wait_time, 0.5)  # minimum 30 seconds
  )

# Quick look at the data
rides |>
  group_by(neighborhood) |>
  summarise(
    n_rides = n(),
    avg_wait = mean(wait_time),
    pct_minority = first(pct_minority),
    median_income = first(median_income),
    .groups = "drop"
  ) |>
  arrange(avg_wait) |>
  print()

# Visualize: wait time distributions by neighborhood
ggplot(rides, aes(x = wait_time, fill = neighborhood)) +
  geom_density(alpha = 0.5) +
  labs(
    title = "Wait Time Distributions by Neighborhood",
    x = "Wait Time (minutes)", y = "Density"
  ) +
  theme_minimal()


# =============================================================================
# PART 2: Bias-Variance Tradeoff
# =============================================================================

# We'll predict wait_time from hour using polynomial regression of
# increasing degree. This demonstrates the classic ISLR bias-variance idea.

# Use a single numeric feature (hour) for clear visualization
# Split into train / validation / test
set.seed(123)
train_idx <- sample(1:n, 0.6 * n)
val_idx   <- sample(setdiff(1:n, train_idx), 0.2 * n)
test_idx  <- setdiff(1:n, c(train_idx, val_idx))

train <- rides[train_idx, ]
val   <- rides[val_idx, ]
test  <- rides[test_idx, ]

# Fit polynomials of degree 1 through 15
degrees <- 1:15

results <- map_dfr(degrees, function(d) {
  fit <- lm(wait_time ~ poly(hour, d), data = train)
  tibble(
    degree    = d,
    train_mse = mean((train$wait_time - predict(fit, train))^2),
    val_mse   = mean((val$wait_time - predict(fit, val))^2)
  )
})

# Classic bias-variance tradeoff plot
results |>
  pivot_longer(cols = c(train_mse, val_mse), names_to = "set", values_to = "mse") |>
  mutate(set = ifelse(set == "train_mse", "Training", "Validation")) |>
  ggplot(aes(x = degree, y = mse, color = set)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  labs(
    title = "Bias-Variance Tradeoff: Polynomial Degree vs. MSE",
    subtitle = "Training error always decreases; validation error has a sweet spot",
    x = "Polynomial Degree", y = "Mean Squared Error", color = "Dataset"
  ) +
  theme_minimal()

best_degree <- results$degree[which.min(results$val_mse)]
cat("\nBest polynomial degree (by validation MSE):", best_degree, "\n")


# =============================================================================
# PART 3: The Punchline — Statistical Bias vs. Social Bias
# =============================================================================

# Now fit the "best" model and check: does low overall MSE mean fair outcomes?

best_fit <- lm(wait_time ~ poly(hour, best_degree), data = train)

# Overall test MSE — looks good!
overall_mse <- mean((test$wait_time - predict(best_fit, test))^2)
cat("\nOverall test MSE:", round(overall_mse, 2), "\n")

# But what happens when we disaggregate by neighborhood?
test_with_preds <- test |>
  mutate(predicted = predict(best_fit, test))

group_metrics <- test_with_preds |>
  group_by(neighborhood) |>
  summarise(
    n = n(),
    avg_actual    = mean(wait_time),
    avg_predicted = mean(predicted),
    mse           = mean((wait_time - predicted)^2),
    bias          = mean(predicted - wait_time),  # systematic over/under
    pct_minority  = first(pct_minority),
    .groups = "drop"
  ) |>
  arrange(pct_minority)

cat("\n--- Disaggregated Model Performance ---\n")
print(group_metrics)

# Visualize: prediction error by neighborhood
ggplot(group_metrics, aes(x = reorder(neighborhood, pct_minority), y = mse, fill = pct_minority)) +
  geom_col() +
  scale_fill_gradient(low = "steelblue", high = "firebrick", labels = scales::percent) +
  labs(
    title = "Model Error is NOT Equal Across Neighborhoods",
    subtitle = "Same model, same overall MSE — but error concentrates in minority areas",
    x = "Neighborhood (ordered by % minority)", y = "MSE",
    fill = "% Minority"
  ) +
  theme_minimal()

# Scatter: predicted vs actual, colored by neighborhood
ggplot(test_with_preds, aes(x = wait_time, y = predicted, color = neighborhood)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "Predicted vs. Actual Wait Time",
    subtitle = "A 'good' model overall can systematically underperform for specific groups",
    x = "Actual Wait (min)", y = "Predicted Wait (min)"
  ) +
  theme_minimal()


# =============================================================================
# PART 4: A Better Model — But Does "Better" Mean "Fairer"?
# =============================================================================

# Now include neighborhood as a feature — MSE will improve everywhere.
# But think about what this means: the model *learns* that Southside gets
# longer waits and *encodes* that as a prediction. Accurate? Yes.
# Fair? That's the question.

better_fit <- lm(wait_time ~ poly(hour, best_degree) + neighborhood, data = train)

test_better <- test |>
  mutate(predicted_v2 = predict(better_fit, test))

group_metrics_v2 <- test_better |>
  group_by(neighborhood) |>
  summarise(
    n = n(),
    avg_actual    = mean(wait_time),
    avg_predicted = mean(predicted_v2),
    mse           = mean((wait_time - predicted_v2)^2),
    bias          = mean(predicted_v2 - wait_time),
    pct_minority  = first(pct_minority),
    .groups = "drop"
  ) |>
  arrange(pct_minority)

cat("\n--- Model v2: With Neighborhood Feature ---\n")
print(group_metrics_v2)

cat("\nOverall test MSE v1 (hour only):", round(overall_mse, 2), "\n")
cat("Overall test MSE v2 (hour + neighborhood):",
    round(mean((test_better$wait_time - test_better$predicted_v2)^2), 2), "\n")

cat("\n")
cat("Key insight: Model v2 is *more accurate* — lower MSE everywhere.\n")
cat("But it has *learned* the disparity: it predicts longer waits for\n")
cat("Southside/Eastside because that's what the data shows.\n")
cat("\n")
cat("Questions to consider:\n")
cat("1. Should a model predict what IS or what SHOULD BE?\n")
cat("2. If we use these predictions for driver dispatch, do we perpetuate\n")
cat("   the very inequality we measured?\n")
cat("3. Is a model that ignores neighborhood (v1) 'fairer' even though\n")
cat("   it's less accurate?\n")
