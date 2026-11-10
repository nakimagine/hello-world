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

**Results (see output):**
- chosen K and price weight
- CV MAE versus test MAE and RMSE
- test MAE compared with the predict-the-training-mean benchmark
- revenue MAE
- bestseller confusion matrix, accuracy, balanced accuracy, precision, recall and F1, reported at both CV and test

Be candid that the bestseller classifier is weak. There are only 7 bestsellers in the test set, and balanced accuracy is close to 0.5.

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

**Results (see output):** Q and cQ for every test product (`inventory_plan_test_products.csv`); category totals (procurement budget, expected overage cost, expected shortage cost, expected profit).

**Sensitivity (see output):**
- c/p from 0.35 to 0.65 (CR from 0.67 to 0.37)
- s/c from 0 to 0.5
- one residual pool with no bands
- Week 5's Normal shortcut, Q = μ + z·σ with one σ for all products

**Back-test (evaluation only).** On the test set, compare the newsvendor Q with ordering the point forecast: realized profit, fill rate and stockouts (Weeks 5 and 7). Also check that actual demand ≤ Q for roughly the CR share of products. Actual test demand is never used to choose Q.

## 7. Final decision

**Selection rule.** Each step maps to a lecture or to the brief.
1. **Positive expected profit** under the newsvendor plan (Week 5).
2. **Launch threshold (Week 4):** at least a 50% chance of selling the training median demand (50 units) or more.
3. **Opportunity cluster (Section 4 / brief):** demand per listing at or above the median cluster *and* no more listings than the median cluster. This is "meaningful demand and fewer close substitutes". On the training map, only **cluster 3** qualifies.
4. **Not crowded locally:** the design is not among the 25% of designs whose KNN neighbours are closest.
5. Among the designs left, choose the **highest expected profit** (Week 5).

**Recommended design (see output):** image, ASIN, title, cluster, planned price, predicted demand proxy (P10/mean/P90), predicted bestseller probability, initial stock Q, procurement spend cQ, expected profit, and the nearest competing listings. The competitors are the KNN neighbours the demand model averaged for this design (Week 4).

### Why not just copy the most popular observed item?

The most popular training item is the Scantron 882-E answer sheet: 9,000 units a month at $8.00, an A-class bestseller. _(Check the rest against output 7.5 after your rerun.)_

1. **Its demand doesn't transfer to a copy.** Its 20 closest training look-alikes have median demand of only 200 a month, about 2% of the leader's 9,000. Brand, reviews and ranking history drive the leader's sales, and a new listing has none of them. Week 3 says the same: similar-looking products can sell very differently.
2. **Copying lands in a crowded cluster.** The top seller's test look-alike sits in cluster 5, one of the two largest clusters (1,960 listings). The recommended design comes from cluster 3, the smallest cluster with the highest demand per listing.
3. **We optimize risk-adjusted profit, not popularity.** Newsvendor expected profit already prices in leftovers (0.405p per unit) and lost sales (0.55p per unit). The copycat is cheap ($9.45), so even at a high forecast it earns little expected profit. The flaw of averages (Week 10) applies: the leader's demand is one extreme outcome, not a new copy's expected outcome.
4. **No look-ahead.** "Copy the top seller" relies on observed demand. Our rule uses only pre-launch information: forecast, CV error distribution and training-set market summaries.

**Caveats.** The data is a single snapshot and the demand proxy is coarse. The bestseller model is weak, so its probability is reported but not used to rank. We recommend a test order of Q units, then reordering from the first weeks of real sales. This is the test-and-repeat logic of Weeks 3–4 (SHEIN).
