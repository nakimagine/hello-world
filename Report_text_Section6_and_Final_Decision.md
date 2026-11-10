# Report text: Sections 4–7

_Paste into the report and edit to fit. Lecture weeks are in brackets so you can cite them. "(see output)" means take the number from your latest Colab run. Numbers already filled in come from Sections 1–4, which won't change when you rerun._

## 4. Competitive landscape: opportunity map

_Section 4's table uses all 9,220 listings. The final decision re-computes the same clusters' statistics on the 80% training set only, so no test demand feeds the decision. The two tables show the same pattern._

| Cluster | Listings | Total demand | Demand / listing | Median price | A-class share | Reading |
|---|---|---|---|---|---|---|
| 0 | 1,806 | 163,900 | 91 | $76 | 9.5% | Large but weak (median demand 0): **unattractive** |
| 1 | 933 | 170,400 | 183 | $150 | 27.0% | Premium, demand concentrated in a few A-class winners |
| 2 | 1,676 | 232,350 | 139 | $60 | 15.4% | Large, below-average demand per listing |
| 3 | 753 | 230,000 | **305** | $45 | 18.6% | **Smallest cluster, highest demand per listing: the opportunity** |
| 4 | 2,092 | 492,200 | 235 | $35 | 14.6% | Strongest total demand but the most listings: **crowded** |
| 5 | 1,960 | 425,950 | 217 | $32 | 11.7% | Strong demand, also crowded |

**Where a proposed design should sit.** The brief's map has three cases:
- A large cluster with strong demand is crowded (clusters 4 and 5).
- A cluster with weak demand is unattractive (cluster 0).
- A cluster with meaningful demand and fewer close substitutes is worth testing.

Cluster 3 is the clearest case of the last one: it has the fewest listings and the highest demand per listing. _(Describe what cluster 3 looks like from the Section 4 sample images. The final-decision shortlist suggests calculators and label makers.)_

**How to differentiate.** Inside cluster 3, a new design should avoid duplicating its nearest neighbours. The final decision lists those neighbours. Ways to stand apart include a different price point from those neighbours, a bundle (for example, a calculator with a case and batteries), or a feature they lack. Week 3's caution applies: a cluster shows association, not causation, so the cluster is a launch signal, not a guarantee.

## 5. Predict demand and bestseller status

**Features (known before launch only).** Each product is described by:
- its DINOv2 image embedding
- its MiniLM title embedding
- its planned price

Reviews, stars, the bestseller flag and demand are never inputs. Both embeddings are L2-normalized, so direction rather than length drives similarity (Week 4). Price is log-transformed because it is skewed (Week 3), then z-scored (Weeks 3–4). All preprocessing sits inside a pipeline, so in every CV fold it is fitted on that fold's training rows only, which avoids leakage (Week 4).

**Model.** K-nearest neighbours (Week 4): a new design's demand is the average demand of its K most similar existing listings. This is the course's cold-start logic.

**Validation (Week 4).**
1. The 80/20 split is made once, stratified on the rare bestseller flag (Section 3).
2. On the 80%, 5-fold CV (also stratified) chooses K (5–100) and the weight of price in the distance (0 to 2). MAE is the selection metric because demand has many zeros (Week 7).
3. The chosen model is refitted on the full 80% and scored once on the untouched 20%.

**Bestseller classifier.** KNN probability = the share of the K neighbours that are bestsellers. Bestsellers are about 0.4% of listings, so a 0.5 cutoff never predicts one. Following Week 4's point that the cutoff should reflect the cost of each error, CV chooses K and the cutoff to maximize balanced accuracy.

**Revenue proxy.** Predicted revenue = predicted demand × planned price.

**Results.**

| Outcome | 5-fold CV (80%) | Untouched test (20%) |
|---|---|---|
| Demand: chosen settings | K = 10 neighbours, price weight 0.5 | |
| Demand MAE | 215.9 units | 217.5 units |
| Demand RMSE | 463.1 units | 456.0 units |
| Benchmark: predict the training mean | | MAE 235.1 units |
| Revenue proxy MAE (demand × price) | | $13,812 |
| Bestseller: chosen settings | K = 10; flag a design if ≥ 1 of 10 neighbours is a bestseller (cutoff 0.10) | |
| Bestseller balanced accuracy | 0.574 (recall 0.179) | 0.485 |
| Bestseller confusion matrix | | TN 1,781 · FP 56 · FN 7 · TP 0 |
| Bestseller accuracy / precision / recall / F1 | | 0.966 / 0 / 0 / 0 |

**What the results mean:**
- **Test error matches CV error** (217.5 vs 215.9), so the model was not overfitted to the training set.
- **KNN beats the predict-the-mean benchmark by about 7.5%.** The gain is real but modest: design similarity explains only part of demand, consistent with Week 3's finding that visual clusters explain a small share of sales variation.
- **The bestseller classifier does not work on the test set.** There are only 7 bestsellers among 1,844 test listings, it caught none, and balanced accuracy is below 0.5. The 0.966 accuracy is misleading because it comes almost entirely from predicting "not a bestseller" (Week 4: accuracy hides imbalance). Bestseller status depends on things a seller can't see before launch, such as reviews and ranking history.

**Limits.** The test set checks predictions on unseen *existing* listings. It does not validate future launches or remove snapshot selection bias.

## 6. Recommend initial inventory

**Demand scenarios.** The newsvendor needs a distribution, not a point forecast (Week 5).
1. Using the same 5 folds, every training product gets an out-of-fold prediction from a model that never saw it. The residual is e = actual − forecast (Week 7).
2. Error size grows with the forecast. Training products are split into 10 bands by predicted demand, and each band keeps its own residuals. This is a local uncertainty estimate, like Week 5's KNN neighbour spread.
3. Each test product's scenarios are max(0, forecast + each residual in its band): about 740 equally likely 30-day demands, all nonnegative as the brief requires.
4. Calibration check (Week 10). Compare the average scenario mean with the average point forecast and the training average (see output). The scenarios add the model's historical errors, so they correct any systematic over-forecasting.

**Cost assumptions (brief defaults).**

| Symbol | Meaning | Value |
|---|---|---|
| p | planned selling price | listing price |
| c | unit procurement cost | 0.45p |
| s | recovery value of an unsold unit | 0.10c = 0.045p |
| Cu = p − c | underage cost | 0.55p |
| Co = c − s | overage cost | 0.405p |
| CR = Cu/(Cu + Co) | critical ratio | 0.576 |

**Order rule (Weeks 5, 10).** Q* is the smallest scenario demand whose cumulative probability reaches 0.576, rounded up. Because c and s scale with p, CR is the same for every product.

**Profit.** Expected profit = p·E[min(Q,D)] + s·E[(Q−D)+] − cQ = (p−c)E[D] − Co·E[(Q−D)+] − Cu·E[(D−Q)+] (Week 5). Procurement spend cQ is the budget. The loss on unsold units is already in the overage term, so cQ is not added to the mismatch cost. The notebook prints a check that both forms give the same total.

**Other assumptions.** One order with no reorder within 30 days, no holding or FBA fees beyond c, and lost sales are not backlogged. `boughtInLastMonth` is rounded down, so Q is probably conservative.

**Calibration.** The average point forecast is 231.9 units, but the average scenario mean is 199.9. That is closer to the training average of 183.6. The CV residuals correct the KNN model's tendency to over-forecast.

**Results.** Q and cQ for every test product are in `inventory_plan_test_products.csv`. Category totals under the base-case costs:

| Item | Value |
|---|---|
| Test products stocked (Q > 0) | 1,702 of 1,844 |
| Total units ordered | 191,895 |
| **Procurement budget (Σ cQ)** | **$5,127,007** |
| Expected revenue | $7,342,308 |
| Expected salvage recovery | $182,297 |
| Expected overage cost (Co × unsold) | $1,640,672 |
| Expected shortage cost (Cu × lost sales) | $9,194,733 |
| **Expected profit** | **$2,397,598** |

The check (p − c)·E[D] − mismatch cost = $2,397,598 matches expected profit, confirming nothing is double-counted.

Expected shortage cost is much larger than overage cost, for two reasons:
- CR is 0.576, so each product deliberately accepts about a 42% chance of stocking out.
- Demand has a long right tail: a few listings sell thousands of units.

The 142 products with Q = 0 are those where more than 57.6% of scenarios are zero demand.

**Sensitivity.**

| Assumption | CR | Budget | Expected profit |
|---|---|---|---|
| c = 0.35p | 0.674 | $5.91M | $3.75M |
| **c = 0.45p (base)** | **0.576** | **$5.13M** | **$2.40M** |
| c = 0.55p | 0.476 | $4.48M | $1.48M |
| c = 0.65p | 0.374 | $3.62M | $0.82M |
| s = 0 | 0.550 | $4.46M | $2.23M |
| s = 0.30c | 0.636 | $6.48M | $2.85M |
| s = 0.50c | 0.710 | $9.53M | $3.55M |

Unit cost has the biggest effect. Raising c from 0.45p to 0.65p lowers CR, cuts the budget by about 30% and cuts expected profit by about two-thirds. A higher recovery value s raises CR, so we stock more: a better salvage deal, such as a supplier buyback, justifies a larger first order (Week 5).

**Two scenario-method checks** pick Q a different way, then judge it against the same banded scenarios:
- **Pooled residuals:** one shared residual pool with no bands.
- **Week 5's Normal shortcut:** Q = μ + 0.19 × 461, one σ for every product.

| Method for choosing Q | Units | Budget | Expected profit |
|---|---|---|---|
| **Banded residuals (our method)** | **191,895** | **$5.13M** | **$2.40M** |
| Pooled residuals, no bands | 369,535 | $9.30M | $1.34M |
| Normal shortcut, Q = μ + 0.19 × 461 | 591,751 | $20.32M | **−$4.07M** |

Both alternatives do worse.
- **Pooled residuals** use the big errors of high-demand listings for every product, so low-demand products are overstocked. That needs about 80% more capital for about 45% less profit.
- **The Normal shortcut** applies one σ (461 units) to every product. It adds about 89 units even to the many products with near-zero demand, which more than doubles expected overage cost to $12.3M and turns profit negative.

This is why we estimate uncertainty locally, by forecast band, rather than with one number for the whole category (Week 5's local-uncertainty idea).

**Back-test (evaluation only; actual test demand was not used to set Q).**

| Policy | Units | Spend | Realized profit | Fill rate | Products stocking out |
|---|---|---|---|---|---|
| **Newsvendor Q** | 191,895 | $5.13M | **$1.08M** | 31.6% | 39.0% |
| Order the point forecast | 427,635 | $11.73M | **−$0.88M** | 53.7% | 24.5% |

The newsvendor policy turns an $880k loss into a $1.08M profit, using less than half the capital. Actual demand fell at or below Q for 61.0% of products, close to the 57.6% target, so the scenario distribution is reasonably calibrated.

Two honest caveats:
- **Realized profit ($1.08M) is below expected profit ($2.40M).** Expected profit assumes the scenarios are exactly right, and the forecasts are noisy.
- **The fill rate is low (31.6%).** A few very high-demand listings dominate total units, and the policy deliberately accepts stockouts on them rather than risk leftovers.

## 7. Final decision

**Selection rule.** Each step maps to a lecture or to the brief.
1. **Positive expected profit** under the newsvendor plan (Week 5).
2. **Launch threshold (Week 4):** at least a 50% chance of selling the training median demand (50 units) or more.
3. **Opportunity cluster (Section 4 / brief):** demand per listing at or above the median cluster *and* no more listings than the median cluster. This is "meaningful demand and fewer close substitutes". On the training map, only **cluster 3** qualifies.
4. **Not crowded locally:** the design is not among the 25% of designs whose KNN neighbours are closest.
5. Among the designs left, choose the **highest expected profit** (Week 5).

**Screening funnel:** 1,844 test designs → 1,702 with positive expected profit → 987 likely to sell ≥ 50 units → 87 in the opportunity cluster (3) → **78** not in the most crowded quarter.

**Recommended design: Casio FX-260 Solar II Scientific Calculator (ASIN B000Q5XTBQ)**

| Item | Value |
|---|---|
| Cluster | 3: the opportunity cluster (621 training listings, 293 units per listing, the highest of all six) |
| Planned price p | $23.48 |
| Predicted 30-day demand proxy | 1,825 units (scenarios: P10 898 / mean 1,575 / P90 2,265) |
| Predicted bestseller probability | 0.00 (training base rate 0.004; the classifier is uninformative, see Section 5) |
| Initial stock Q | **1,385 units** |
| Unit cost c / procurement spend cQ | $10.57 / **$14,634** |
| Expected 30-day profit | **$14,130** (expected profit / spend 0.97) |
| Expected overage / shortage cost | $1,593 / $4,621 |
| P(zero demand) | 0.4% |

**Nearest competing listings.** These are the 10 training listings the KNN model averaged. They are mostly Casio and Texas Instruments scientific calculators priced $9–$25, and most are A-class (for example the Casio fx-115ES PLUS at 8,000 a month and the fx-300ES PLUS at 2,000–3,000 a month). The neighbourhood has proven demand, and it is a niche of its own rather than a slice of the large ink-and-toner clusters.

**Risk to flag: price.** One neighbour is the same FX-260 model listed at **$9.49**, selling 2,000 a month. At $23.48 our listing must justify the premium, for example with a bundle (protective case, spare battery, exam-ready pack) or a back-to-school timing push. Otherwise it should be priced closer to the market. This is the "how could it be differentiated" answer for Section 4.

**Runner-up.** The Auto-Cut Label Maker PS-100E ($79.99, cluster 3) has almost the same expected profit ($14,127). Its neighbours' median demand is only 100 a month (versus 1,000 for the Casio), and its profit per dollar of stock is lower (0.68 versus 0.97). The calculator is the safer choice: its neighbourhood shows proven demand, and it needs less capital per dollar of expected profit.

### Why not just copy the most popular observed item?

The most popular training item is the Scantron 882-E answer sheet: 9,000 units a month at $8.00, an A-class bestseller. Its closest test look-alike is the ScanRite 815-E Scantron-compatible quiz sheet ($9.45).

1. **Its demand doesn't transfer to a copy.** Its 20 closest training look-alikes have median demand of only 200 a month, about 2% of the leader's 9,000. Brand, reviews and ranking history drive the leader's sales, and a new listing has none of them. Week 3 says the same: similar-looking products can sell very differently.
2. **Copying lands in a crowded cluster.** The top seller's test look-alike sits in cluster 5, one of the two largest clusters (1,960 listings). The recommended design comes from cluster 3, the smallest cluster with the highest demand per listing.
3. **We optimize risk-adjusted profit, not popularity.** Newsvendor expected profit already prices in leftovers (0.405p per unit) and lost sales (0.55p per unit). The copycat's forecast is high (1,115 units), but at $9.45 its expected profit is only **$2,111** (on $2,870 of stock), against **$14,130** for the Casio. The flaw of averages (Week 10) applies: the leader's demand is one extreme outcome, not a new copy's expected outcome.
4. **No look-ahead.** "Copy the top seller" relies on observed demand. Our rule uses only pre-launch information: forecast, CV error distribution and training-set market summaries.

**Caveats.** The data is a single snapshot and the demand proxy is coarse. The bestseller model is weak, so its probability is reported but not used to rank. We recommend a test order of Q units, then reordering from the first weeks of real sales. This is the test-and-repeat logic of Weeks 3–4 (SHEIN).
