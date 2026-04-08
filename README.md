# ML Refresher: Discrimination in Ride-Sharing

A short, hands-on refresher on machine learning fundamentals through the lens
of algorithmic discrimination in ride-sharing platforms (Uber, Lyft). Each
module pairs a classical ML concept with a real discrimination scenario, and
every exercise is in R.

> **Live slides:** https://fhoces.github.io/ml-discrimination-refresher/

## Why this exists

ML textbooks teach you to optimize a loss function. They rarely teach you that
a model with low loss can still be socially harmful — and they almost never
teach you to *check* for that systematically. This refresher closes that gap
by walking through the standard ML pipeline (linear models → trees → neural
nets) while keeping a single applied question in view: **how does this kind
of model end up discriminating, and how would you detect it?**

The ride-sharing setting is concrete enough to ground every formula and
broad enough to carry across modules. The same synthetic dataset (driver
acceptance, neighborhoods correlated with demographics) gets reused, so you
can see how each new technique changes — or fails to change — the same
fairness story.

## How to use this repo

Each module folder contains three files:

- **`concepts.md`** — written refresher of the ML ideas (the "what")
- **`slides.Rmd`** — `xaringan` slide deck rendered to `slides.html` (the "show")
- **`exercise.R`** — runnable R script with the worked example (the "do")

To rebuild the slides locally:

```r
rmarkdown::render("module-01/slides.Rmd")
```

The slides depend on `tidyverse`, `glmnet`, `broom`, `yardstick`, `patchwork`,
and `ggrepel`. Install with `install.packages(...)` as needed.

## Modules

| # | Module | Concepts | Application |
|---|--------|----------|-------------|
| **1** | [The Learning Problem](module-01/) ([slides](https://fhoces.github.io/ml-discrimination-refresher/module-01/slides.html)) | Bias-variance, train/val/test, loss functions | Disparate wait times — statistical bias vs social bias |
| **2** | [Linear Models](module-02/) ([slides](https://fhoces.github.io/ml-discrimination-refresher/module-02/slides.html)) | Linear & logistic regression, Ridge / Lasso / Elastic Net | Driver acceptance & proxy discrimination |
| **3** | [Model Evaluation & Selection](module-03/) ([slides](https://fhoces.github.io/ml-discrimination-refresher/module-03/slides.html)) | Confusion matrix, precision / recall / FPR, ROC-AUC, calibration, k-fold CV | Auditing the acceptance model — equal AUC ≠ equal treatment |
| 4 | Tree-Based Methods | Decision trees, Random Forest, XGBoost, SHAP | Surge pricing & geographic redlining |
| 5 | Unsupervised Learning | PCA, K-means, hierarchical clustering, UMAP | Rider segmentation that mirrors demographics |
| 6 | Neural Networks | Perceptrons, backprop, SGD/Adam | Deep pricing models — auditing the un-interpretable |
| **7** | [Fairness Frameworks & Metrics](module-07/) ([slides](https://fhoces.github.io/ml-discrimination-refresher/module-07/slides.html)) | Demographic parity, equalized odds, predictive parity, impossibility theorem | Fair dispatch under accuracy/fairness tradeoff |
| 8 | Auditing & Interpretability | SHAP, LIME, partial dependence, counterfactuals | A full audit report for a ride-sharing model |

## Key papers & references

- Ge, Knittel, MacKenzie & Zoepf (2016) — *Racial and Gender Discrimination
  in Transportation Network Companies* (Uber/Lyft field experiment)
- Pandey & Caliskan (2021) — *Disparate Impact of Artificial Intelligence
  Bias in Ridehailing Economy's Price Discrimination Algorithms*
- Angwin, Larson, Mattu & Kirchner (2016) — *Machine Bias* (ProPublica's
  COMPAS investigation, used as the canonical example in Module 3)

## License

The exercises and notes are released under the MIT License so you can reuse
or adapt them in your own teaching.
