# Module 11: Fairness Frameworks & Metrics

## Quick Refresher

Module 3 taught you to compute the same metric per group and look at the
gap. This module tells you what the gaps **mean** and shows that you
**can't close all of them at once**.

### The setup

For a binary classifier:
- $Y \in \{0, 1\}$ — the actual label (e.g. driver accepted the ride)
- $\hat{Y} \in \{0, 1\}$ — the model's prediction
- $A$ — a protected attribute (e.g. minority status), used only for *auditing*

A "fairness criterion" is some equality condition between groups. There are
many of them, and most of the famous ones are mutually incompatible.

### The Three Big Fairness Criteria

#### 1. Demographic parity (a.k.a. statistical parity, group fairness)

The model predicts "positive" at the same rate for both groups:

$$P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)$$

For driver acceptance: minority and non-minority riders are predicted to be
accepted at the same rate. **Ignores the actual labels** — it's about the
model's outputs alone.

When it makes sense: hiring screens, loans, dispatch, anything where you
want equal exposure to a positive decision regardless of base rate.

When it breaks: if base rates genuinely differ, demographic parity forces
the model to misclassify on purpose to hit equal positive rates.

#### 2. Equalized odds (a.k.a. error rate parity)

True positive rate **and** false positive rate are equal across groups:

$$P(\hat{Y} = 1 \mid Y = 1, A = 0) = P(\hat{Y} = 1 \mid Y = 1, A = 1)$$
$$P(\hat{Y} = 1 \mid Y = 0, A = 0) = P(\hat{Y} = 1 \mid Y = 0, A = 1)$$

The model is allowed to have different positive rates *if and only if* the
underlying truth justifies it. Among the actually-accepted requests, the
model catches the same fraction in each group; among the actually-rejected
ones, it falsely flags at the same rate.

A common weaker version, **equal opportunity**, only requires the TPR
condition (the FPR is allowed to differ).

When it makes sense: classification with a real outcome you care about
(medical diagnosis, recidivism, fraud), where missing real positives or
flagging real negatives has direct human consequences.

#### 3. Predictive parity (a.k.a. calibration within groups)

Among instances the model says "positive", the actual positive rate is the
same in both groups (and likewise for negatives):

$$P(Y = 1 \mid \hat{Y} = 1, A = 0) = P(Y = 1 \mid \hat{Y} = 1, A = 1)$$

This is *precision* equality across groups. For probabilistic models, the
stronger version is **calibration within groups**: at every probability
score $s$, the actual positive rate equals $s$ in both groups.

When it makes sense: when downstream users interpret a score the same way
regardless of group ("a 0.8 means 0.8"), like risk scores in lending or
recidivism.

### The Impossibility Theorem

**Chouldechova (2017) / Kleinberg, Mullainathan & Raghavan (2017):** if
the base rates differ between groups (i.e. $P(Y=1 \mid A=0) \neq P(Y=1 \mid
A=1)$) and the classifier is not perfect, then **at most one of**:

1. Demographic parity
2. Equalized odds
3. Predictive parity

can hold. There is no model that satisfies all three.

This is not a quirk of any particular model. It is a **mathematical fact
about the relationships between the criteria**. If you try to enforce two,
the third breaks.

So fairness requires *picking* which criterion you care about — and
defending the choice. The COMPAS debate from Module 3 was exactly this:
ProPublica argued for equalized odds (FPRs were unequal), Northpointe
argued for predictive parity (calibration was equal), and **both were
right** — the system can't satisfy both at once.

### Individual Fairness

A different paradigm entirely: similar individuals should receive similar
predictions. Usually formalized as a Lipschitz condition on the scoring
function:

$$|f(x_1) - f(x_2)| \leq L \cdot d(x_1, x_2)$$

for some distance $d$ on the input space. Sounds nice; the catch is that
you have to *pick a distance* and that choice itself encodes social judgments.

### How to Enforce a Criterion

There are three injection points:

#### Pre-processing
Massage the **training data** before training. Reweight examples,
resample, or transform features to remove dependence on the protected
attribute. Pros: model-agnostic. Cons: throws away signal; the model can
relearn proxies.

#### In-processing
Modify the **training objective** to include a fairness penalty. Examples:
adversarial debiasing, fairness-constrained optimization. Pros: principled.
Cons: needs custom training code; harder to interpret.

#### Post-processing
Take the trained model's scores and **adjust the decision rule** per
group. Easiest version: pick a different threshold per group so the chosen
fairness criterion holds. Pros: simplest, model-agnostic, immediate. Cons:
explicitly uses the protected attribute at decision time, which is illegal
in many settings.

### The Accuracy–Fairness Tradeoff

Every fairness intervention costs accuracy when base rates differ.
The more aggressively you enforce parity, the more the model has to deviate
from its loss-optimal predictions. The result is a Pareto frontier:

- Fully optimized for accuracy → biggest disparities
- Fully fair (under whatever criterion) → biggest accuracy hit
- Most production systems live somewhere on the curve, with the operating
  point chosen by policy rather than by math

---

## The Discrimination Angle: Picking a Definition Is a Political Choice

Fairness is **not** a mathematical problem with a unique correct answer.
It's a value judgment about *which equality you want to protect*, and the
math just tells you that you can't have all of them.

That means engineers, product managers, and lawyers — not statisticians —
have to decide. The job of the data scientist is:

1. **Show the disparities** (Module 3).
2. **Translate them into the available fairness criteria** (this module).
3. **Show what each fix costs** in accuracy and in the *other* fairness
   criteria.
4. **Make the tradeoff explicit** so the people with authority can pick.

A common failure mode: a team picks "demographic parity" because it's
easy, ships it, and discovers six months later that the model is now
*systematically miscalibrated* for one group — the price of demographic
parity was predictive parity. Both are real things you can mean by
"fair," and you have to choose.

### How this plays out at Uber/Lyft

- **Dispatch algorithms** that maximize total throughput will underserve
  low-density (often low-income, often non-white) neighborhoods. Enforcing
  geographic demographic parity raises wait times overall.
- **Surge pricing** that's calibrated equally across neighborhoods will
  produce different surge frequencies, because demand patterns differ —
  and that's the *price of calibration*.
- **Driver acceptance models** can be made to predict "accept" at equal
  rates across demographics (demographic parity), but the FPR will then
  differ by group (equalized odds breaks).

There is no neutral default. Every choice is a choice.

---

## Exercise Preview

In the R exercise, you will:

1. Reuse the driver-acceptance dataset from Modules 2–3
2. Compute demographic parity, equalized odds, and predictive parity for
   the race-blind logistic model
3. Verify the impossibility theorem empirically: show that you can satisfy
   any two of the three but not all three at once
4. Apply a **post-processing fix**: pick group-specific thresholds that
   enforce demographic parity, then measure what happens to accuracy and
   to the other fairness metrics
5. Plot the **accuracy–fairness frontier** by sweeping thresholds and
   visualize where the tradeoffs live
