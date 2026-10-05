# OPIM 348 — Homework 2 Report

**Group ID:** ____  **Members:** ____

**Software:** Python 3 with pandas, numpy, scipy, scikit-learn, statsmodels (`ExponentialSmoothing`, `initialization_method="estimated"`) and matplotlib. All code is in `HW2_solution.ipynb`, which runs top to bottom from the two GitHub URLs.
**Rounding:** statistics are shown to 2 decimals. OR bookings are rounded to the nearest whole minute (floored at 0). Q, R and safety stock are rounded **up** to whole units.

---

## Question 1 — Operating room booking

**(a)** D = Wheels Out − Wheels In (minutes). All **2,172** encounters are usable (no missing timestamps, all D > 0).
**Mean D = 79.70 min, SD = 31.82 min.** In the plot of D against Booked Time, points sit in vertical bands because bookings come in 15/30-minute blocks, and 57% of cases overrun their booking.

**(b)** C(Q, D) = 100·max(Q − D, 0) + 50·max(D − Q, 0).
Cₒ = 100, Cᵤ = 50, so the **critical ratio = 50 / (50 + 100) = 1/3**. The optimal booking is the 33rd percentile of D, which is **below** the predicted mean (for a symmetric predictive distribution the mean equals the median, and 1/3 < 1/2). Idle booked time costs twice as much as an overrun, so the cost-optimal booking deliberately runs short.

**(c)** 80/20 split (random_state = 42): 1,737 training and 435 test encounters. One-hot encoding of CPT Code, OR Suite and weekday is fit on training data only. The test set contains no CPT codes unseen in training.
**Test MAE = 4.89 min, Test RMSE = 7.47 min.**

**(d)** Training residual SD (df-adjusted) **σ̂ = 7.42 min**. The cost-based booking is
**Qⱼ = round(μ̂ⱼ + z₁/₃·σ̂) = round(μ̂ⱼ − 0.4307 × 7.42) ≈ μ̂ⱼ − 3.2 min**, rounded to the nearest minute.

| Policy (test set, n = 435) | Total cost | Unused min | Overrun min | % overrun |
|---|---:|---:|---:|---:|
| Historical booking | $336,400 | 1,755 | 3,218 | 61.4% |
| Predicted mean | $149,650 | 873 | 1,247 | 42.8% |
| Cost-based, Normal (required) | $149,700 | 421 | 2,152 | 79.1% |
| *Sensitivity: empirical 1/3 quantile of training residuals* | *$140,300* | *522* | *1,762* | *64.1%* |

Both model-based policies cut cost by about 56%. The Normal cost-based booking ties the predicted mean: it moves time from unused minutes into overruns as intended, but it overshoots, giving a 79% overrun rate against the 67% it targets. The training residuals are right-skewed (skew 0.76) and heavy-tailed (excess kurtosis 4.6), so the Normal 1/3 quantile (−3.2 min) sits too far below the mean compared with the empirical one (−2.0 min). The sensitivity row is estimated from training data only.

**(e) Recommendation.** Replace historical bookings with model-based bookings built from the predicted duration for each case's CPT code, OR suite and weekday. This step alone cuts test mismatch cost by more than half. Because an unused minute costs twice an overrun minute, book slightly below the predicted mean, at the 1/3 quantile. Estimate that quantile from the empirical training residuals (about 2 minutes below the mean) rather than from a Normal approximation. Refit the model regularly, add a fallback for unseen CPT codes, and revisit the cost weights, since frequent overruns carry overtime and patient-delay costs this model leaves out.

---

## Question 2 — Two-day sales forecasting

Blocks: training = 675 days (1 Jan 2018 – 6 Nov 2019), validation = 225 (7 Nov 2019 – 18 Jun 2020), test = 226 (19 Jun 2020 – 30 Jan 2021).

**(a)** Daily sales are right-skewed: median 11, mean 17.75, SD 31.9, max 343. There are **61 zero-sales days (5.4%)**. A one-off **surge in March 2019** (≈120–340 units/day for the whole month) accounts for all 29 days above mean + 3 SD, and it falls in the training block. After it, the level drops: the training block averages 25.1/day, validation 6.7 and test 6.9. There is a mild weekly pattern with Friday–Sunday higher (Saturday mean 23.2 vs 14.8–15.5 Tue–Thu).

**(b–c)** Expanding-window rolling validation: there are 224 origins, from the end of training to the last origin whose two target days both fall inside validation. Each ETS model is refit at every origin on all data through that day.

| Model (validation, two-day totals) | MAE | RMSE | Mean error (actual − pred) |
|---|---:|---:|---:|
| **MA7** | **4.92** | **6.57** | −0.14 |
| ETS additive seasonal (sp = 7), no trend | 9.83 | 12.05 | −0.10 |
| ETS additive trend + additive seasonal | 9.79 | 12.03 | −0.07 |

**(d)** **MA7 is selected** for its lowest validation MAE, and it also has the lowest RMSE. All three models have a near-zero, slightly negative bias, so there is no trade-off to weigh. The ETS models' additive weekly amplitude is learned largely from the high-volume, spike-affected training period, so they project weekly swings of about ±7 units on a level of only about 7/day. Adding a trend doesn't fix this. MA7 adapts because it uses only the last week. The notebook plots the final 30 validation origins.

**(e)** With MA7 frozen and rolled through the test block (225 origins):

| MA7 | MAE | RMSE | Mean error |
|---|---:|---:|---:|
| Validation | 4.92 | 6.57 | −0.14 |
| Test | 4.85 | 6.14 | −0.12 |

Test performance is slightly better. Test demand has a similar mean to validation (6.9 vs 6.7) but less day-to-day volatility (SD 4.0 vs 4.4, max 18 vs 29), so two-day totals are a little easier to predict. Bias stays near zero. No model was chosen or tuned using test data.

---

## Question 3 — Continuous review (Q, R)

**(a)** First 80% (900 days): daily mean **20.48**, sample SD **35.10**. **D = 20.48 × 365 = 7,475.6 units/yr**, **H = 0.30 × 1.80 = $0.54/unit/yr**, **EOQ = √(2·10·7,475.6 / 0.54) = 526.19 → Q = 527.**

**(b)** **CSL = 1 − HQ/(pD) = 1 − 0.54·527 / (0.20·7,475.6) = 0.8097.** CSL is the probability that a replenishment cycle ends without a stockout, i.e., that demand during the lead time does not exceed R. It is not the fill rate (the share of demand served).

**(c)** Under i.i.d. Normal demand: μ_L = 2 × 20.48 = **40.96**, σ_L = √2 × 35.10 = **49.64**, z = Φ⁻¹(0.8097) = 0.8766, **SS = 43.52**, **R = ⌈84.48⌉ = 85.** (The March 2019 spike inflates the SD: excluding that month, the daily SD would be 10.7.)

**(d)** Two-day validation errors (actual − predicted) from the no-trend seasonal ETS: n = 224, nearest rank k = ⌈0.8097 × 224⌉ = 182, e₍₁₈₂₎ = 10.33, so **forecast-based SS = 11 units.**
A two-day error is needed because the lead time is two days: once an order is placed, stock must cover the next two days of demand, so the relevant risk is the error in the two-day total. One-day errors understate this. The variance roughly doubles over two days, and correlated consecutive errors mean you can't simply scale by √2, whereas realised two-day errors capture both effects.

**(e)** For each test day t, set the reorder point at the start of the day (forecast origin = end of day t − 1, using only sales observed so far). Refit the no-trend weekly-seasonal ETS and forecast days t and t + 1. Then set
**R_t = ⌈F̂_t + F̂_{t+1} + 11⌉**.
Whenever inventory position falls to R_t or below during day t, order Q = 527 units. Across the 226 test days, R_t ranges from 11 to 45 (mean 25.3, median 24), well below the fixed Normal R of 85. The fixed R is driven by the stale, spike-inflated full-history mean and SD. The full day-by-day table is in the notebook and in `q3e_reorder_points.csv`.
