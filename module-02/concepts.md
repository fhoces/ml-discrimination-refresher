# Module 2: Linear Models

## Quick Refresher

You remember this from ISLR Chapters 3, 4, and 6. Here's the fast version.

### Linear Regression

Predict a continuous outcome `Y` as a linear combination of features:

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
```

Coefficients are estimated by minimizing **residual sum of squares**:

```
β̂ = argmin Σᵢ (yᵢ - β₀ - Σⱼ βⱼxᵢⱼ)²
```

Closed form: `β̂ = (XᵀX)⁻¹ XᵀY`. No iterative optimization needed.

### Logistic Regression

For binary outcomes `Y ∈ {0, 1}`, model the log-odds as linear:

```
log(p / (1-p)) = β₀ + β₁X₁ + ... + βₚXₚ
```

Equivalently, `p = σ(β₀ + Σ βⱼXⱼ)` where `σ` is the sigmoid function. Fit by
**maximum likelihood** (no closed form — use iteratively reweighted least squares
or gradient methods).

Coefficients are interpreted as **log-odds ratios**: a one-unit change in `Xⱼ`
multiplies the odds of `Y=1` by `exp(βⱼ)`.

### Why Regularize?

Plain linear/logistic regression has problems when:
- `p` is large relative to `n` (overfitting, instability)
- Features are correlated (multicollinearity → huge variance in coefficients)
- You want **automatic feature selection**

Regularization adds a penalty on the size of coefficients to the loss function.

### Ridge (L2)

```
β̂ridge = argmin [ RSS + λ Σⱼ βⱼ² ]
```

- Shrinks coefficients toward zero, but **never exactly zero**
- Handles multicollinearity well
- All features stay in the model

### Lasso (L1)

```
β̂lasso = argmin [ RSS + λ Σⱼ |βⱼ| ]
```

- Shrinks **and** sets some coefficients exactly to zero
- Performs **feature selection** automatically
- Picks one of a group of correlated features arbitrarily

### Elastic Net

Mixes L1 and L2:

```
β̂elastic = argmin [ RSS + λ (α Σⱼ |βⱼ| + (1-α) Σⱼ βⱼ²) ]
```

- `α = 1` → Lasso, `α = 0` → Ridge
- Best of both: groups correlated features together (Ridge) while still
  selecting features (Lasso)

### Choosing λ

Use **cross-validation**. `glmnet::cv.glmnet()` does this for you and reports:
- `lambda.min` — minimizes CV error
- `lambda.1se` — largest λ within 1 standard error of the minimum (sparser
  model, often preferred)

### A Note on Scaling

Regularization penalties depend on the scale of features. **Always standardize**
predictors (mean 0, sd 1) before fitting Ridge/Lasso/Elastic Net. `glmnet`
does this by default.

---

## The Discrimination Angle: Proxy Variables

Linear models are simple, interpretable, and **dangerous** when it comes to
discrimination — precisely because they're so interpretable that you might
think you've solved the problem when you haven't.

### Removing the protected attribute is not enough

Suppose you build a driver-acceptance model and decide not to use **race** as
a feature. Problem solved? No.

The model will happily learn from **proxy variables** — features that correlate
with race but aren't race itself:

- **Pickup neighborhood** — strongly correlated with racial demographics
- **Zip code** — same problem
- **First name** (if used) — has racial signal
- **Phone area code** — geographic, hence demographic
- **Trip distance + pickup time** — can encode "this person lives in a poor
  neighborhood and works late shifts"

A linear model will assign coefficients to these proxies that effectively
recreate racial discrimination, even though "race" never appears in the data.
This is **disparate impact**: a facially neutral model produces racially
disparate outcomes.

### Why Lasso makes it worse (or better?)

Lasso selects features. If neighborhood is the strongest predictor and gets
selected, you have a sparse, interpretable model that explicitly bases its
decisions on neighborhood — which is racially correlated. The **sparsity is
honest** about what the model is doing, but the **interpretability** can lull
you into thinking the model is fair.

Conversely, Ridge keeps all features but shrinks them. The discrimination
gets spread across many proxies, making it **harder to detect** by inspection.

### Coefficient interpretation ≠ causation

When you look at a logistic regression coefficient for "pickup in Southside =
-0.8 log-odds of acceptance," you might say "the model penalizes Southside
pickups." But that's only the *correlation* in your training data — it could
reflect:
- Genuinely different driver behavior in that area
- Historical underservicing creating a feedback loop
- Sample selection (which trips ended up in your dataset)

The model can't tell you which. You have to.

### How this plays out at Uber/Lyft

1. **Driver acceptance models** learn that requests from certain neighborhoods
   have lower acceptance rates → algorithm shows fewer requests from those
   areas to drivers → drivers learn to associate the area with bad rides →
   acceptance drops further → loop continues
2. **Surge pricing** based on linear models with location features will surge
   more in low-supply areas, which correlate with race
3. **Insurance and credit scoring** for gig workers use similar linear models
   with location proxies → systematic disadvantage

### What to watch for

- **Feature audit**: For every feature, ask "could this be a proxy?"
- **Coefficient magnitudes by group**: Even with race excluded, compute the
  model's predictions per demographic group
- **Counterfactual test**: If you change only the neighborhood (holding all
  else equal), does the prediction change a lot? That's a proxy in action
- **Sparsity ≠ fairness**: A simpler model is not automatically a fairer model

This is the foundation for everything in Modules 11–12 (fairness frameworks
will give you formal definitions of what "fair" can mean here).

---

## Exercise Preview

In the R exercise, you will:
1. Simulate a driver-acceptance dataset with neighborhood, time, distance,
   rating features — and a hidden racial demographic per pickup
2. Fit plain logistic regression and inspect coefficients
3. Fit Lasso with `glmnet::cv.glmnet()` and watch which features survive
4. Fit Ridge and Elastic Net for comparison
5. Plot coefficient paths to see how regularization shrinks features
6. Compute acceptance rates per demographic group — even without using race
   as a feature, watch the disparate impact emerge
