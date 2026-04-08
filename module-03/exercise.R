# =============================================================================
# Module 3: Model Evaluation & Selection
# Application: Fraud Detection with Unequal Error Rates
# =============================================================================
#
# We build a fraud-detection classifier for ride requests and audit it for
# unequal error rates across demographic groups. The point: a model can have
# equal AUC across groups and STILL produce wildly different false positive
# rates — which is the central tension formalized in Module 11.

library(tidyverse)
library(yardstick)

# =============================================================================
# PART 1: Simulate Fraud-Detection Data
# =============================================================================

set.seed(2025)
n <- 8000

# Two demographic groups, with different background fraud rates AND different
# feature distributions. The "fraud signal" is real but mixed with proxies.
requests <- tibble(
  request_id = 1:n,
  is_minority = rbinom(n, 1, 0.4),
  pickup_neighborhood = ifelse(is_minority == 1,
                               sample(c("Eastside","Southside","Midtown"), n, TRUE,
                                      prob = c(0.5, 0.4, 0.1)),
                               sample(c("Downtown","Midtown","Eastside"), n, TRUE,
                                      prob = c(0.55, 0.35, 0.1))),
  trip_distance_mi = pmax(rlnorm(n, meanlog = 1.2, sdlog = 0.6), 0.3),
  rider_account_age_days = pmax(rnorm(n, mean = 400, sd = 200), 1),
  num_recent_cancels = rpois(n, lambda = 0.4),
  payment_method = sample(c("card", "wallet", "cash"), n, replace = TRUE,
                          prob = c(0.7, 0.2, 0.1))
) |>
  mutate(
    # True fraud probability — driven by genuine signals + a small bias
    # baked into the neighborhood feature (the model will exploit this)
    logit_fraud = -3.5 +
      0.6 * num_recent_cancels +
      -0.003 * rider_account_age_days +
      0.8 * (payment_method == "cash") +
      0.4 * (pickup_neighborhood %in% c("Eastside", "Southside")),
    fraud_prob = plogis(logit_fraud),
    is_fraud = rbinom(n, 1, fraud_prob)
  )

cat("Overall fraud rate:", round(mean(requests$is_fraud), 3), "\n")
cat("Fraud rate by group:\n")
requests |>
  group_by(is_minority) |>
  summarise(rate = round(mean(is_fraud), 3), n = n(), .groups = "drop") |>
  print()


# =============================================================================
# PART 2: Train / Test Split and Fit a Model
# =============================================================================

set.seed(123)
train_idx <- sample(1:n, 0.7 * n)
train <- requests[train_idx, ]
test  <- requests[-train_idx, ]

# Race-blind logistic regression (is_minority is NOT in the formula)
fit <- glm(
  is_fraud ~ pickup_neighborhood + trip_distance_mi +
             rider_account_age_days + num_recent_cancels + payment_method,
  data = train, family = binomial
)

test$pred_prob <- predict(fit, test, type = "response")
test$pred_class <- factor(as.integer(test$pred_prob >= 0.5),
                          levels = c(0, 1))
test$truth <- factor(test$is_fraud, levels = c(0, 1))


# =============================================================================
# PART 3: Overall Metrics
# =============================================================================

cat("\n--- Overall Performance ---\n")

# Confusion matrix
cm <- conf_mat(test, truth = truth, estimate = pred_class)
print(cm)

# A bunch of metrics in one go
metrics_overall <- bind_rows(
  accuracy(test, truth, pred_class),
  precision(test, truth, pred_class, event_level = "second"),
  recall(test, truth, pred_class, event_level = "second"),
  f_meas(test, truth, pred_class, event_level = "second"),
  roc_auc(test, truth, pred_prob, event_level = "second")
)
print(metrics_overall)


# =============================================================================
# PART 4: The Audit — Disaggregate by Group
# =============================================================================

cat("\n--- Per-Group Metrics ---\n")

per_group <- test |>
  group_by(is_minority) |>
  summarise(
    n         = n(),
    actual_fraud_rate = mean(is_fraud),
    flagged_rate      = mean(pred_class == 1),
    accuracy  = accuracy_vec(truth, pred_class),
    precision = precision_vec(truth, pred_class, event_level = "second"),
    recall    = recall_vec(truth, pred_class, event_level = "second"),
    fpr       = {
      tn <- sum(truth == 0 & pred_class == 0)
      fp <- sum(truth == 0 & pred_class == 1)
      fp / (fp + tn)
    },
    auc       = roc_auc_vec(truth, pred_prob, event_level = "second"),
    .groups   = "drop"
  )
print(per_group)

# Compute the gap
fpr_gap <- diff(per_group$fpr)
cat("\nFalse positive rate gap (minority - non-minority):",
    round(fpr_gap, 3), "\n")
cat("This is the percentage of innocent minority riders flagged MORE\n",
    "than innocent non-minority riders.\n")


# =============================================================================
# PART 5: Per-Group ROC Curves
# =============================================================================

roc_data <- test |>
  group_by(is_minority) |>
  roc_curve(truth, pred_prob, event_level = "second") |>
  mutate(group = ifelse(is_minority == 1, "Minority", "Non-minority"))

ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity, color = group)) +
  geom_line(linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  scale_color_manual(values = c("Non-minority" = "steelblue",
                                 "Minority" = "firebrick")) +
  labs(
    title = "Per-Group ROC Curves",
    subtitle = "Even when AUC is similar, operating points (e.g. threshold = 0.5) can differ",
    x = "False Positive Rate", y = "True Positive Rate", color = "Group"
  ) +
  theme_minimal()


# =============================================================================
# PART 6: Per-Group Calibration
# =============================================================================

calib <- test |>
  mutate(bin = cut(pred_prob, breaks = seq(0, 1, by = 0.1),
                   include.lowest = TRUE)) |>
  group_by(is_minority, bin) |>
  summarise(
    avg_predicted = mean(pred_prob),
    actual_rate   = mean(is_fraud),
    n = n(),
    .groups = "drop"
  ) |>
  mutate(group = ifelse(is_minority == 1, "Minority", "Non-minority"))

ggplot(calib, aes(x = avg_predicted, y = actual_rate, color = group)) +
  geom_line(linewidth = 1) +
  geom_point(aes(size = n)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  scale_color_manual(values = c("Non-minority" = "steelblue",
                                 "Minority" = "firebrick")) +
  labs(
    title = "Per-Group Calibration",
    subtitle = "Above the diagonal = model under-predicts fraud rate",
    x = "Predicted Probability", y = "Actual Fraud Rate", color = "Group", size = "n"
  ) +
  theme_minimal()


# =============================================================================
# PART 7: Threshold Sweep — Disparate Impact at Every Cutoff
# =============================================================================

thresholds <- seq(0.05, 0.95, by = 0.05)

threshold_audit <- map_dfr(thresholds, function(t) {
  test |>
    mutate(flag = as.integer(pred_prob >= t)) |>
    group_by(is_minority) |>
    summarise(flag_rate = mean(flag), .groups = "drop") |>
    pivot_wider(names_from = is_minority, values_from = flag_rate,
                names_prefix = "g") |>
    mutate(threshold = t,
           di_ratio = g1 / g0)
})

ggplot(threshold_audit, aes(x = threshold, y = di_ratio)) +
  geom_line(linewidth = 1.2, color = "firebrick") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey") +
  labs(
    title = "Disparate Impact Ratio Across Thresholds",
    subtitle = "Ratio = minority flag rate / non-minority flag rate. 1.0 = parity.",
    x = "Decision Threshold", y = "Disparate Impact Ratio"
  ) +
  theme_minimal()


cat("\n")
cat("Key takeaways:\n")
cat("1. Overall AUC and accuracy hide group-level disparities.\n")
cat("2. The same model has DIFFERENT false positive rates across groups.\n")
cat("3. Calibration can look fine even when error rates differ.\n")
cat("4. Picking a threshold is itself a fairness decision.\n")
cat("5. Auditing = compute every metric per group. No new math required.\n")
