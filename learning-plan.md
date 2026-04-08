# ML Refresher: Discrimination in Ride-Sharing

Refresh ML fundamentals through the lens of algorithmic discrimination in
ride-sharing platforms (Uber, Lyft). Each module teaches an ML concept and
applies it to a real discrimination scenario — all exercises in R.

**Time budget: ~10 hours**

| Block | Modules | Time | Focus |
|-------|---------|------|-------|
| **A** | 1–3 | 2.5h | Foundations + evaluation |
| **B** | 4 + 5 | 2h | Trees/boosting + clustering |
| **C** | 6 | 1.5h | Neural net basics |
| **D** | 7–8 | 2.5h | Fairness frameworks + auditing |
| **Buffer** | — | 1.5h | Deep dives on what interests you most |

---

## Phase 1: Core Foundations (2.5h)

### 1. The Learning Problem
- Bias-variance tradeoff, train/val/test splits, loss functions
- **Application — Disparate wait times:** Uber's pricing and dispatch algorithms
  optimize for efficiency, but efficiency metrics can encode geographic and
  racial bias. What does "bias" mean statistically vs. socially?
- **Exercise (R):** Generate synthetic ride request data across neighborhoods with
  different demographics. Fit models of varying complexity to predict wait time.
  Show how a model can have low statistical bias but high social bias.
- **R tools:** `tidyverse`, `ggplot2`, `caret::createDataPartition()`

### 2. Linear Models
- Linear & logistic regression, regularization (Ridge, Lasso, Elastic Net)
- **Application — Driver acceptance modeling:** Build a logistic regression
  to predict whether a driver accepts a ride request. Features include pickup
  neighborhood, time of day, rider rating, estimated trip distance. Investigate
  which features act as proxies for race/income.
- **Exercise (R):** Simulate a driver acceptance dataset. Fit regularized logistic
  regression with `glmnet`. Examine how Lasso selects or drops neighborhood
  features and what that means for disparate treatment. Plot coefficient paths.
- **R tools:** `glmnet`, `tidymodels`, `broom`

### 3. Model Evaluation & Selection
- k-fold CV, precision/recall/F1, AUC-ROC, calibration
- **Application — Unequal error rates:** A fraud detection model that flags
  "suspicious" ride patterns may have different false positive rates across
  demographic groups (e.g., flagging riders from low-income zip codes more
  often).
- **Exercise (R):** Train a classifier to detect fraudulent rides. Compute
  confusion matrices and ROC curves separately per demographic group using
  `yardstick`. Quantify the disparity.
- **R tools:** `rsample`, `yardstick`, `pROC`, `ggplot2`

---

## Phase 2: Classical ML Methods (2h)

### 4. Tree-Based Methods
- Decision trees, Random Forests, Gradient Boosting (XGBoost)
- **Application — Surge pricing and neighborhood effects:** Surge pricing
  models (often tree-based) can systematically charge more in neighborhoods
  that correlate with race. Explore how tree splits on geographic features
  create de facto redlining.
- **Exercise (R):** Build a gradient-boosted surge pricing model with `xgboost`.
  Use `SHAPforxgboost` to identify whether neighborhood/zip code is driving
  predictions. Compare a model trained with vs. without geographic features.
- **R tools:** `rpart`, `rpart.plot`, `ranger`, `xgboost`, `SHAPforxgboost`

### 5. Unsupervised Learning
- PCA, K-means, hierarchical clustering, UMAP
- **Application — Rider and driver segmentation:** Platforms cluster users
  for marketing and service tiers. These clusters can mirror demographic
  lines without explicitly using protected attributes.
- **Exercise (R):** Cluster riders by trip patterns, ratings, and location.
  Use `uwot::umap()` to visualize. Overlay demographic labels to check if
  "behavioral" clusters are actually demographic clusters.
- **R tools:** `stats::prcomp()`, `stats::kmeans()`, `uwot`, `factoextra`

---

## Phase 3: Neural Networks (1.5h)

### 6. Neural Networks Foundations
- Perceptrons, multi-layer networks, backpropagation, SGD/Adam
- **Application — Dynamic pricing with deep models:** Neural net pricing
  models are harder to interpret than linear ones. When a deep model sets
  prices, how do you audit it for discrimination?
- **Exercise (R):** Build a small pricing neural network with `torch` for R.
  Compare its pricing predictions across neighborhoods to the linear model
  from Module 2. Attempt interpretation with `luz` callbacks.
- **R tools:** `torch`, `luz`, `ggplot2`

---

## Phase 4: Fairness & Auditing (2.5h)

### 7. Fairness Frameworks & Metrics
- Demographic parity, equalized odds, predictive parity, individual fairness
- Impossibility theorem: you can't satisfy all fairness criteria simultaneously
- Pre-processing (reweighting), in-processing (constraints), post-processing
  (threshold adjustment)
- **Application — Fair dispatch algorithms:** Can a dispatch system minimize
  wait times while ensuring equal service across neighborhoods?
- **Exercise (R):** Implement a ride dispatch simulator. Train models with
  and without fairness constraints. Compute fairness metrics manually and
  with the `fairness` package. Measure the accuracy-fairness tradeoff.
- **R tools:** `fairness`, `mlr3fairness`, `ggplot2`

### 8. Auditing & Interpretability
- SHAP, LIME, partial dependence plots, counterfactual explanations
- Audit methodologies: correspondence studies, outcome tests
- **Application — Auditing Uber's algorithm:** Replicate (in simplified form)
  the methodology from studies that sent matched ride requests from different
  neighborhoods to detect differential treatment.
- **Exercise (R):** Perform a full audit of a trained ride-sharing model:
  compute SHAP values with `shapviz`, generate LIME explanations with `lime`,
  create partial dependence plots with `pdp`, and write a short audit report.
- **R tools:** `shapviz`, `lime`, `pdp`, `iml`, `DALEXtra`

---

## Key Papers & References
- Ge, Knittel, MacKenzie & Zoepf (2016) — "Racial and Gender Discrimination
  in Transportation Network Companies" (field experiment on Uber/Lyft)
- Pandey & Caliskan (2021) — "Disparate Impact of Artificial Intelligence
  Bias in Ridehailing Economy's Price Discrimination Algorithms"
- Chicago rideshare trip data (publicly available via city open data portal)
- NYC TLC trip record data (publicly available)

## R Stack
- **tidyverse** — data wrangling and visualization
- **tidymodels** (rsample, recipes, parsnip, yardstick, tune, workflows) — modeling framework
- **glmnet** — regularized regression
- **rpart / ranger / xgboost** — tree-based methods
- **torch / luz** — neural networks
- **SHAPforxgboost / shapviz / lime / pdp / iml / DALEXtra** — interpretability
- **fairness / mlr3fairness** — fairness metrics
- **uwot / factoextra** — unsupervised learning visualization
- **pROC** — ROC analysis

## How to Use This Plan
1. Read the concept review for each module (I can generate these).
2. Work through the exercise in an R script or RMarkdown notebook.
3. Ask me to review your code or explain anything.

Say **"start module 1"** to begin.
