# =============================================================================
# Module 11: Fairness Frameworks & Metrics
# Application: Three Fairness Criteria Walk Into a Dispatch System
# =============================================================================
#
# We reuse the driver-acceptance scenario from Modules 2 and 3. The model is
# fit without using race; the audit from Module 3 already showed the model
# discriminates anyway. This module asks: which formal definition of "fair"
# should we adopt — and what does each one cost?

library(tidyverse)

# =============================================================================
# PART 1: Rebuild the data and model from Modules 2-3
# =============================================================================

set.seed(2024)
n <- 5000

neighborhoods <- tibble(
  neighborhood   = c("Downtown", "Midtown", "Eastside", "Southside"),
  base_accept    = c(0.85, 0.78, 0.55, 0.40),
  pct_minority   = c(0.20, 0.35, 0.70, 0.85)
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
    logit_p = qlogis(base_accept) - 0.6 * is_night -
              0.15 * abs(trip_distance_mi - 4) +
              0.5 * (rider_rating - 4.7),
    accepted = rbinom(n, 1, plogis(logit_p)),
    is_minority = rbinom(n, 1, pct_minority)
  )

set.seed(123)
train_idx <- sample(1:n, 0.7 * n)
train <- requests[train_idx, ]
test  <- requests[-train_idx, ]

# Race-blind logistic regression
fit <- glm(accepted ~ neighborhood + is_night + trip_distance_mi + rider_rating,
           data = train, family = binomial)

test$pred_prob <- predict(fit, test, type = "response")
test$pred_class <- as.integer(test$pred_prob >= 0.5)


# =============================================================================
# PART 2: Compute the three fairness criteria
# =============================================================================

# A small helper that returns all the metrics we need per group
group_metrics <- function(df, threshold = 0.5) {
  df |>
    mutate(pred = as.integer(pred_prob >= threshold)) |>
    group_by(is_minority) |>
    summarise(
      n           = n(),
      base_rate   = mean(accepted),                              # P(Y=1 | A)
      pos_rate    = mean(pred),                                  # P(Yhat=1 | A)
      tpr         = sum(pred == 1 & accepted == 1) / sum(accepted == 1),
      fpr         = sum(pred == 1 & accepted == 0) / sum(accepted == 0),
      precision   = sum(pred == 1 & accepted == 1) / max(sum(pred == 1), 1),
      .groups = "drop"
    )
}

cat("\n--- Per-group metrics at threshold 0.5 ---\n")
m <- group_metrics(test, 0.5)
print(m)

# Demographic parity:  pos_rate equal across groups
# Equalized odds:      tpr AND fpr equal across groups
# Predictive parity:   precision equal across groups

cat("\n--- Fairness gaps at threshold 0.5 ---\n")
cat(sprintf("  Demographic parity gap (pos_rate):  %.3f\n",
            diff(m$pos_rate)))
cat(sprintf("  Equalized odds gap (TPR):           %.3f\n",
            diff(m$tpr)))
cat(sprintf("  Equalized odds gap (FPR):           %.3f\n",
            diff(m$fpr)))
cat(sprintf("  Predictive parity gap (precision):  %.3f\n",
            diff(m$precision)))


# =============================================================================
# PART 3: The impossibility result, empirically
# =============================================================================
#
# Sweep thresholds, compute every metric, and look at the gaps. We will see
# that no single threshold zeroes all three gaps simultaneously.

thresholds <- seq(0.05, 0.95, by = 0.02)

sweep <- map_dfr(thresholds, function(t) {
  g <- group_metrics(test, t)
  tibble(
    threshold        = t,
    dp_gap   = abs(diff(g$pos_rate)),
    tpr_gap  = abs(diff(g$tpr)),
    fpr_gap  = abs(diff(g$fpr)),
    prec_gap = abs(diff(g$precision)),
    accuracy = mean((test$pred_prob >= t) == test$accepted)
  )
})

cat("\n--- Best (closest-to-zero gap) thresholds for each criterion ---\n")
cat(sprintf("  Demographic parity (min |dp_gap|):    threshold = %.2f\n",
            sweep$threshold[which.min(sweep$dp_gap)]))
cat(sprintf("  Equalized TPR     (min |tpr_gap|):    threshold = %.2f\n",
            sweep$threshold[which.min(sweep$tpr_gap)]))
cat(sprintf("  Predictive parity (min |prec_gap|):   threshold = %.2f\n",
            sweep$threshold[which.min(sweep$prec_gap)]))

# Plot all the gaps as a function of threshold
sweep |>
  pivot_longer(c(dp_gap, tpr_gap, fpr_gap, prec_gap),
               names_to = "criterion", values_to = "gap") |>
  ggplot(aes(threshold, gap, color = criterion)) +
  geom_line(linewidth = 1.1) +
  scale_color_manual(values = c(
    dp_gap   = "firebrick",
    tpr_gap  = "darkgreen",
    fpr_gap  = "steelblue",
    prec_gap = "orange"
  ),
  labels = c(
    dp_gap   = "Demographic parity",
    fpr_gap  = "FPR (eq. odds)",
    prec_gap = "Predictive parity",
    tpr_gap  = "TPR (eq. opportunity)"
  )) +
  labs(title = "No single threshold zeros every gap",
       subtitle = "Each curve = absolute difference between minority and non-minority",
       x = "Decision threshold",
       y = "|gap| between groups",
       color = NULL) +
  theme_minimal()


# =============================================================================
# PART 4: Post-processing fix — group-specific thresholds for demographic parity
# =============================================================================
#
# Pick a different threshold for each group so the predicted positive rate
# matches across groups. Then measure what happens to accuracy and to the
# OTHER fairness criteria.

target_pos_rate <- mean(test$pred_class)  # the model's overall positive rate

# For each group, find the threshold that yields the target positive rate
threshold_for_group <- function(group_df, target) {
  ts <- seq(0, 1, by = 0.001)
  pos_rates <- sapply(ts, \(t) mean(group_df$pred_prob >= t))
  ts[which.min(abs(pos_rates - target))]
}

t_majority <- threshold_for_group(filter(test, is_minority == 0), target_pos_rate)
t_minority <- threshold_for_group(filter(test, is_minority == 1), target_pos_rate)

cat("\n--- Post-processing: per-group thresholds for demographic parity ---\n")
cat(sprintf("  Majority threshold: %.3f\n", t_majority))
cat(sprintf("  Minority threshold: %.3f\n", t_minority))

test$pred_postproc <- ifelse(test$is_minority == 1,
                              as.integer(test$pred_prob >= t_minority),
                              as.integer(test$pred_prob >= t_majority))

cat("\n--- After post-processing fix ---\n")
post_metrics <- test |>
  group_by(is_minority) |>
  summarise(
    n         = n(),
    pos_rate  = mean(pred_postproc),
    tpr       = sum(pred_postproc == 1 & accepted == 1) / sum(accepted == 1),
    fpr       = sum(pred_postproc == 1 & accepted == 0) / sum(accepted == 0),
    precision = sum(pred_postproc == 1 & accepted == 1) / max(sum(pred_postproc == 1), 1),
    .groups   = "drop"
  )
print(post_metrics)

cat(sprintf("\n  DP gap after fix:    %.3f (was %.3f)\n",
            diff(post_metrics$pos_rate),  diff(m$pos_rate)))
cat(sprintf("  TPR gap after fix:   %.3f (was %.3f)\n",
            diff(post_metrics$tpr),       diff(m$tpr)))
cat(sprintf("  Precision gap after: %.3f (was %.3f)\n",
            diff(post_metrics$precision), diff(m$precision)))

cat(sprintf("  Accuracy before:     %.3f\n",
            mean(test$pred_class == test$accepted)))
cat(sprintf("  Accuracy after fix:  %.3f\n",
            mean(test$pred_postproc == test$accepted)))


# =============================================================================
# PART 5: The accuracy–fairness frontier
# =============================================================================
#
# Sweep the *minority* threshold while holding the majority threshold fixed
# at 0.5; trace accuracy vs the demographic-parity gap. Each point on the
# curve is one possible operating policy.

t_min_grid <- seq(0.05, 0.95, by = 0.02)

frontier <- map_dfr(t_min_grid, function(tm) {
  pred <- ifelse(test$is_minority == 1,
                 as.integer(test$pred_prob >= tm),
                 as.integer(test$pred_prob >= 0.5))
  pos_rates <- tapply(pred, test$is_minority, mean)
  tibble(
    minority_threshold = tm,
    accuracy = mean(pred == test$accepted),
    dp_gap   = abs(pos_rates[1] - pos_rates[2])
  )
})

ggplot(frontier, aes(dp_gap, accuracy)) +
  geom_line(linewidth = 1, color = "firebrick") +
  geom_point(size = 1.4, color = "firebrick") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
  labs(
    title = "Accuracy vs Fairness Frontier",
    subtitle = "Each point = one minority-group threshold; majority fixed at 0.5",
    x = "Demographic parity gap (|minority pos_rate - majority pos_rate|)",
    y = "Accuracy"
  ) +
  theme_minimal()


cat("\n")
cat("Key takeaways:\n")
cat("1. The same model has DIFFERENT 'fair' answers depending on which\n")
cat("   criterion you adopt. The criteria are mutually incompatible.\n")
cat("2. Closing the demographic parity gap forces the model to use HIGHER\n")
cat("   thresholds for one group, which hurts that group's TPR.\n")
cat("3. There is no statistical fix to the impossibility theorem. The\n")
cat("   choice of fairness criterion is a value judgment.\n")
cat("4. The accuracy-fairness frontier is what you actually have to pick\n")
cat("   a point on. The math shows the cost; humans pick the point.\n")
