# Module 3: Model Evaluation & Selection

## Quick Refresher

You remember this from ISLR Chapter 5. Here's the fast version.

### Why You Need More Than One Number

Module 1's "MSE" and Module 2's "accuracy" are single scalars. They hide
*where* and *for whom* the model is wrong. For classification, you need to
look at the **confusion matrix**.

|              | Predicted = 1 | Predicted = 0 |
|--------------|---------------|---------------|
| Actual = 1   | TP            | FN            |
| Actual = 0   | FP            | TN            |

From this you derive:

- **Accuracy** = `(TP + TN) / total` — overall fraction correct
- **Precision** = `TP / (TP + FP)` — of the things you flagged, how many were real?
- **Recall** (= sensitivity, true positive rate) = `TP / (TP + FN)` — of the
  real positives, how many did you catch?
- **Specificity** (true negative rate) = `TN / (TN + FP)`
- **False positive rate** = `FP / (FP + TN) = 1 - specificity`
- **F1 score** = harmonic mean of precision and recall

### Which Metric When?

Accuracy is misleading for **imbalanced** problems. If 1% of rides are
fraudulent, predicting "no fraud" always gets 99% accuracy. Useless.

- **Fraud detection / rare-event problems** → recall (don't miss the bad
  ones) and precision (don't waste investigator time)
- **Calibrated probabilities matter** → cross-entropy / log loss
- **Want to compare models across all thresholds** → AUC-ROC

### ROC and AUC

A binary classifier outputs a probability. You pick a **threshold** to turn
that into a 0/1 prediction. Different thresholds give different (FPR, TPR)
points.

- **ROC curve** = TPR vs FPR as the threshold sweeps from 0 to 1
- **AUC** = area under the ROC curve
- AUC = 0.5 → random; AUC = 1.0 → perfect ranking
- Interpretation: AUC = probability that a randomly chosen positive scores
  higher than a randomly chosen negative

### k-Fold Cross-Validation

Split the training data into `k` folds. For each fold:
1. Train on the other `k-1` folds
2. Predict on the held-out fold
3. Record the metric

Average across folds. Common choices: `k = 5` or `k = 10`. Lower variance
than a single train/val split. Use it when:
- Your dataset is small (a single split throws away too much)
- You need to tune hyperparameters
- You want a stable estimate of generalization error

### Calibration

A classifier is **well-calibrated** if, among requests it scored 0.8, about
80% are actually positive. You can check this with a **reliability diagram**
(predicted probability vs observed frequency, binned).

A high-AUC model can still be poorly calibrated — it ranks things correctly
but its probabilities are off. Calibration matters when:
- You're using the probability directly (e.g., expected value calculations)
- You're applying a fixed threshold (a miscalibrated 0.5 might be the wrong
  cutoff)

You can fix calibration after the fact with **Platt scaling** (logistic
regression on the scores) or **isotonic regression**.

---

## The Discrimination Angle: Unequal Error Rates

Here's where Module 1's "look at the loss per group" gets formalized.

### The same model, different errors

A classifier can have great overall accuracy and *still* be terrible for a
specific group, in two distinct ways:

1. **Different precision** → the same flag (e.g. "this ride is suspicious")
   means different things for different groups
2. **Different recall** → the model catches positives for one group but
   misses them for another

For ride-sharing fraud detection, this looks like:

- **Disparate FPR**: the model flags innocent rides from low-income zip codes
  more often than from wealthy ones (false positive rate higher for one group)
- **Disparate TPR**: it misses real fraud in one group more than another
- **Disparate calibration**: a "score of 0.8" actually means 60% fraud risk
  for group A and 90% for group B

### The famous example: COMPAS

The 2016 ProPublica investigation of the COMPAS recidivism algorithm found
that it had similar overall AUC for Black and white defendants, **but**:

- Among defendants who did **not** re-offend, Black defendants were
  classified as high-risk **twice as often** (higher FPR)
- Among defendants who **did** re-offend, white defendants were
  classified as low-risk more often (lower TPR for whites means model
  catches Black recidivism more aggressively)

The COMPAS makers responded that the model was **calibrated** equally across
groups (a score of 7 meant the same recidivism rate for both). Both sides
were right — and that's the **impossibility result** we'll see in Module 11:
you can't satisfy all fairness criteria at once.

### What "auditing" actually means

A real audit of an ML system is just: compute every metric you care about,
**broken down by demographic group**, and look at the differences.

You don't need new statistics — you need the discipline to slice the
existing ones. The `yardstick` package in tidymodels makes this trivial:
`group_by(group) |> metrics(truth, prediction)`.

### How this plays out at Uber/Lyft

1. **Fraud detection** flags certain ride patterns. If pickup neighborhood
   is in the model, the false positive rate will be higher for low-income
   areas → drivers' trips get cancelled, riders get banned more often
2. **Driver dishonesty detection** (e.g. cancelling rides at the last
   minute). If the model uses any geographic feature, certain drivers in
   certain areas get flagged disproportionately
3. **"Quality" scores** for both drivers and riders are typically classifiers
   trained on past behavior. Past behavior is biased → future flags are
   biased → people lose access to the platform

### What to compute on every model

- Per-group **precision, recall, FPR, F1**
- Per-group **ROC curves** overlaid on one plot
- Per-group **calibration plots**
- The **disparate impact ratio** for any decision threshold you might use

This module is the practical bridge to fairness metrics in Module 11.

---

## Exercise Preview

In the R exercise, you will:
1. Simulate a fraud-detection dataset with a hidden demographic group
2. Train a logistic regression and an xgboost-ish booster (we'll use a
   simple gradient boost or another logistic) — anything to compare
3. Use `yardstick` to compute confusion matrices, precision, recall, ROC,
   and AUC for the whole test set
4. Then disaggregate every metric by demographic group and look at the
   disparities
5. Plot per-group ROC curves and calibration curves
6. Discuss why "equal AUC" is not the same as "equal treatment"
