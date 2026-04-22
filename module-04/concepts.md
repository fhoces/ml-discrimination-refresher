# Module 4: Tree-Based Methods

## Quick Refresher

You remember decision trees from ISLR Chapter 8. Here's the fast version.

### Decision Trees

A decision tree splits the data recursively by asking yes/no questions about
features. At each internal node, it picks the feature and threshold that best
separates the outcome classes (or reduces variance for regression).

**How splits are chosen** — for classification, the standard criterion is
**Gini impurity**:

$$G = 1 - \sum_k p_k^2$$

where $p_k$ is the fraction of class $k$ in the node. A pure node (all one
class) has $G = 0$. A 50/50 split has $G = 0.5$. The tree picks the split
that produces the largest decrease in weighted Gini across the two child
nodes.

**Overfitting**: unconstrained trees grow until every leaf is pure — they
memorize the training data. You control complexity via:
- `max_depth` — how deep the tree can grow
- `min_n` — minimum observations in a leaf
- Pruning — grow a big tree, then cut back branches using cross-validation

### Random Forests

A single tree is high-variance (small changes in data → very different tree).
Random forests fix this by averaging many trees:

1. **Bagging** (bootstrap aggregating): draw B bootstrap samples from the
   data, fit one tree to each, average the predictions.
2. **Feature randomization**: at each split, only consider a random subset of
   `mtry` features (out of p total). This **decorrelates** the trees —
   without it, every tree would split on the same strong predictor first,
   and averaging correlated predictions doesn't reduce variance much.

**Key hyperparameters:**
- `trees` (B) — number of trees. More is better up to diminishing returns
  (~500 is usually sufficient). Unlike boosting, random forests don't overfit
  as you add more trees.
- `mtry` — number of features considered at each split. Default: $\sqrt{p}$
  for classification, $p/3$ for regression. Lower values → more decorrelation
  but weaker individual trees.
- `min_n` — minimum leaf size. Larger values → simpler trees.

**Variable importance**: for each tree, measure how much each feature
decreases Gini impurity across all splits that use it. Average across all
trees. Features that appear in many trees and create large impurity decreases
are "important."

### Gradient Boosting (GBM / XGBoost)

While random forests build trees **independently** (in parallel), boosting
builds them **sequentially** — each tree corrects the mistakes of the
ensemble so far.

1. Start with a simple prediction (e.g., the overall class proportion).
2. Compute the **residuals** (how wrong the current ensemble is).
3. Fit a small ("weak") tree to predict those residuals.
4. Add that tree to the ensemble, scaled by a **learning rate** $\eta$.
5. Repeat for `trees` iterations.

**Key hyperparameters:**
- `trees` — number of boosting rounds. Too many → overfitting (unlike RF).
- `tree_depth` — depth of each individual tree. Typically 1–6. Depth 1
  ("stumps") captures only main effects; depth 6 captures up to 6-way
  interactions.
- `learn_rate` ($\eta$) — shrinkage factor. Smaller values are more
  conservative (need more trees but generalize better). Typical range:
  0.01–0.3.

**Interaction between hyperparameters**: low learning rate + many trees ≈
high learning rate + few trees, but the low-rate version usually generalizes
better (at the cost of computation time).

### SHAP Values

SHAP (SHapley Additive exPlanations) decomposes each individual prediction
into feature contributions. For observation $i$:

$$\hat{f}(x_i) = \phi_0 + \sum_j \phi_j(x_i)$$

where $\phi_0$ is the average prediction and $\phi_j(x_i)$ is how much
feature $j$ pushed this prediction above or below average.

SHAP values come from cooperative game theory (Shapley values). They are
the *only* feature attribution method that satisfies:
- **Efficiency**: contributions sum to the prediction
- **Symmetry**: features that contribute equally get equal SHAP values
- **Dummy**: features the model doesn't use get SHAP = 0

For tree-based models, exact SHAP values can be computed efficiently using
the TreeSHAP algorithm (polynomial time vs. exponential for general Shapley).

### Comparing the Three

| Property | Decision Tree | Random Forest | GBM |
|----------|--------------|---------------|-----|
| Bias | High (simple) | Medium | Low |
| Variance | Very high | Low (averaging) | Low (sequential correction) |
| Interpretability | High (one tree) | Medium (importance) | Medium (SHAP) |
| Overfits with more trees? | N/A | No | Yes |
| Handles interactions? | Naturally | Naturally | Naturally |
| Training speed | Fast | Parallelizable | Sequential (slower) |

### Key Papers & References
- Breiman (2001) — "Random Forests" (the original paper)
- Friedman (2001) — "Greedy Function Approximation: A Gradient Boosting
  Machine"
- Chen & Guestrin (2016) — "XGBoost: A Scalable Tree Boosting System"
- Lundberg & Lee (2017) — "A Unified Approach to Interpreting Model
  Predictions" (SHAP)
