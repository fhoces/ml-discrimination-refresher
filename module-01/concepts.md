# Module 1: The Learning Problem

## Quick Refresher

You remember this from ISLR Chapter 2. Here's the fast version.

### The Setup

We observe data `(X, Y)` and want to learn a function `f` such that `Y ≈ f(X)`.
The irreducible error `ε` means no model is perfect:

```
Y = f(X) + ε
```

We choose `f̂` from some model class by minimizing a **loss function** on
training data, then evaluate on held-out data.

### Bias-Variance Tradeoff

For any estimate `f̂`, the expected prediction error at a point `x₀` decomposes:

```
E[(Y - f̂(x₀))²] = Var(f̂(x₀)) + [Bias(f̂(x₀))]² + Var(ε)
                     \_________/   \________________/   \_____/
                      variance        bias²            irreducible
```

- **Bias** — error from wrong assumptions (e.g., fitting a line to a curve)
- **Variance** — sensitivity to the specific training sample
- **Simple models** → high bias, low variance
- **Complex models** → low bias, high variance
- The sweet spot minimizes their sum → this is model selection

### Train / Validation / Test

- **Training set** — fit the model
- **Validation set** — choose hyperparameters, compare models
- **Test set** — final unbiased estimate of performance, used once

ISLR used cross-validation to approximate the validation step. Same idea.

### Loss Functions

| Task | Loss | Formula |
|------|------|---------|
| Regression | MSE | `(1/n) Σ(yᵢ - ŷᵢ)²` |
| Classification | 0-1 loss | `(1/n) Σ I(yᵢ ≠ ŷᵢ)` |
| Classification | Cross-entropy | `-Σ yᵢ log(p̂ᵢ)` |

MSE penalizes large errors more. Cross-entropy is preferred for probabilistic
classifiers because it rewards well-calibrated probabilities.

---

## The Discrimination Angle: Statistical Bias ≠ Social Bias

Here's where it gets interesting for ride-sharing.

**Statistical bias** means your model's average prediction is off — it
systematically over- or under-estimates. You fix it by using a more flexible
model.

**Social bias** (discriminatory impact) means your model treats groups
differently in ways that cause harm — even if the model is statistically
unbiased overall.

### A concrete example

Imagine a wait-time prediction model for a ride-sharing app:

- The model predicts average wait = 5 min across all neighborhoods (low bias!)
- But it's 3 min in wealthy/white neighborhoods and 8 min in low-income/Black
  neighborhoods
- The *model* isn't wrong — it accurately reflects the *system* (fewer drivers
  dispatched to certain areas)
- The system itself encodes discrimination, and the model faithfully learns it

This is the core tension: **a statistically good model can be a socially
harmful model.** Optimizing for overall MSE can hide disparities that only
appear when you disaggregate by group.

### How this plays out at Uber/Lyft

1. **Dispatch optimization** minimizes total wait time → sends more drivers
   where demand is dense → underserves sparse/low-income areas
2. **Surge pricing** responds to supply-demand imbalance → areas with fewer
   drivers get higher prices → low-income riders pay more
3. **ETA prediction** trained on historical data → if historical service was
   worse in certain neighborhoods, the model learns to predict longer waits
   there → this becomes a self-fulfilling prophecy (drivers avoid areas with
   long predicted ETAs)

### What to watch for

When evaluating a model, always ask:
- What is the loss **overall**?
- What is the loss **per group**?
- Are error rates equal across groups, or does the model work better for some?

This is the foundation for everything in Modules 11–12 (fairness metrics).

---

## Exercise Preview

In the R exercise, you will:
1. Generate synthetic ride data with neighborhood demographics
2. Fit polynomial models of increasing complexity
3. Show the classic bias-variance tradeoff curves
4. Then disaggregate by neighborhood to reveal hidden social bias
