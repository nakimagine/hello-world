# Report text: Sections 5–6 and Final Decision

_Paste into the report and edit to fit. Wherever it says "(see output)", fill in the number from your Colab run. Lecture weeks are given in brackets so you can cite them._

## 5. Predict demand and bestseller status

**Features (known before launch only).** Each product is described by its DINOv2 image embedding, its MiniLM title embedding and its planned price. Reviews, star rating, bestseller flag and demand are never used as inputs. Both embeddings are L2-normalized, so the direction of the vector drives similarity and its length does not (Week 4). Price is log-transformed because it is heavily skewed (Week 3), then z-score standardized (Weeks 3–4). All of this preprocessing sits inside a pipeline, so in every cross-validation fold it is fitted on that fold's training rows only. This avoids the leakage warned about in Week 4.

**Model choice.** We use K-nearest neighbours (Week 4). A new design's demand is predicted as the average demand of the K most similar existing listings. This is the course's cold-start logic: visual (and title) similarity identifies analogues, and their sales become evidence for the new design.

**Validation workflow (Week 4).** The 80/20 split was made once, stratified on the rare bestseller flag. On the 80% development set, 5-fold cross-validation (also stratified on bestseller) chooses K (5–100) and how much weight price gets in the distance (0, 0.25 or 0.5). MAE is the selection metric because demand has many zeros (Week 7). The chosen model is refitted on the full 80% and scored once on the untouched 20% test set.

**Bestseller classifier.** A KNN classifier gives each design a probability equal to the share of its K neighbours that are bestsellers (Week 4). Bestsellers are only about 0.4% of listings, so the default 0.5 cutoff never predicts one. Following Week 4's point that the cutoff should reflect the cost of each error, K and the cutoff are chosen by 5-fold CV to maximize balanced accuracy.

**Revenue proxy.** Predicted revenue = predicted demand × planned price.

**Results (see output):** chosen K and price weight; test MAE and RMSE compared with a predict-the-mean benchmark; the predicted-versus-observed plot; revenue MAE; bestseller confusion matrix, accuracy, balanced accuracy, precision, recall and F1.

**Limits.** The test set checks predictions on unseen *existing* listings. It does not validate future launches, and the snapshot data carries selection bias.

## 6. Recommend initial inventory

**Demand scenarios.** A point forecast cannot describe the risk of leftovers or shortages. The newsvendor needs a distribution (Week 5).
1. Using the same 5 folds, every training product gets an out-of-fold prediction from a KNN model that never saw it. The residual is e = actual − forecast (Week 7).
2. Error size grows with the forecast. So we split training products into 10 bands by predicted demand and keep each band's residuals. This is a local uncertainty estimate, in the spirit of Week 5's KNN neighbour spread.
3. Each test product's scenarios are max(0, its forecast + each residual in its band), about 740 equally likely 30-day demands. Setting negatives to 0 keeps demand nonnegative, as the brief requires.
4. Calibration check (Week 10): we compare the average scenario mean with the average point forecast (see output).

**Cost assumptions (brief defaults).**

| Symbol | Meaning | Value |
|---|---|---|
| p | planned selling price | listing price |
| c | unit procurement cost | 0.45p |
| s | recovery value of an unsold unit | 0.10c = 0.045p |
| Cu = p − c | underage cost (margin lost per unit short) | 0.55p |
| Co = c − s | overage cost (loss per unsold unit) | 0.405p |
| CR = Cu/(Cu + Co) | critical ratio | 0.576 |

**Order rule (Weeks 5 and 10).** Q* is the smallest scenario demand whose cumulative probability reaches CR = 0.576, rounded up to a whole unit. c and s are proportional to p, so CR is the same for every product.

**Profit and costs.** Expected profit = p·E[min(Q,D)] + s·E[(Q−D)+] − cQ, which equals (p−c)E[D] − Co·E[(Q−D)+] − Cu·E[(D−Q)+] (Week 5). Procurement spend cQ is the budget. The loss on unsold units is already in the overage term, so cQ is not added to the mismatch cost (no double counting). The notebook prints a check that the two forms are equal.

**Other assumptions.** One order with no reorder within 30 days, no holding or FBA fees beyond c, and lost sales are not backlogged. `boughtInLastMonth` is rounded down ("50+"), so demand, and therefore Q, are probably understated.

**Results (see output):** Q and cQ for every test product (saved to `inventory_plan_test_products.csv`); category totals (procurement budget, expected overage cost, expected shortage cost, expected profit).

**Sensitivity (see output).**
- c from 0.35p to 0.65p: CR falls from 0.67 to 0.37, so Q and the budget shrink.
- s from 0 to 0.5c: higher recovery value raises CR and Q (Week 5).
- One residual pool with no bands.
- Week 5's Normal shortcut, Q = μ + z·σ with one σ for all products.

**Back-test (evaluation only).** On the test set we compare realized profit and fill rate of the newsvendor Q against simply ordering the point forecast, as Week 5 compares newsvendor booking with historical booking. We also check that actual demand ≤ Q for roughly 57.6% of products. Actual test demand is never used to choose Q.

## 7. Final decision

**How we choose.** Each step maps to a lecture or the brief.
1. **Positive expected profit** under the newsvendor plan (Week 5).
2. **Launch threshold (Week 4):** at least a 50% chance of selling 50 or more units in 30 days, where 50 is the training median demand proxy.
3. **Opportunity map (brief, Section 4):** the design's image cluster has at least the median demand per listing, meaning meaningful demand.
4. **Not crowded:** the design is not among the 25% of test designs whose KNN neighbours are closest, meaning fewer close substitutes.
5. Among the designs left, **pick the highest expected profit** (Week 5).

**Recommended design (see output):** image, ASIN, title, cluster, planned price, predicted demand proxy (with P10/mean/P90), predicted bestseller probability, initial stock Q, procurement spend cQ, expected profit, and its nearest competing listings. Those competitors are the KNN neighbours the demand model averaged for this design (Week 4).

### Why not just copy the most popular observed item?

_Check each claim against the F.5 output before submitting._

1. **A top seller's demand comes from things a copy does not get:** brand, thousands of reviews, ranking history and Buy Box position. F.5 shows this in the data. The training listings closest to the top seller sell far less than it does (see their median demand and the share with zero demand). Week 3 makes the same point: similar-looking products can sell very differently, and a cluster shows association, not causation.
2. **Copying puts us in the most crowded part of the map.** A new listing there competes on price against established sellers. Our design sits where similar listings show real demand but substitutes are fewer. That is the "meaningful demand and fewer close substitutes" area the brief describes.
3. **We optimize risk-adjusted profit, not popularity.** The newsvendor expected profit already prices in leftovers (0.405p per unit) and lost sales (0.55p per unit) (Week 5). The flaw of averages (Week 10) applies: the top seller's demand is one extreme outcome, not the expected outcome for a new copy.
4. **No look-ahead.** "Copy the top seller" relies on observed demand. Our choice uses only pre-launch information: forecast, CV error distribution and training-set market summaries.

**Caveats.** The data is a single snapshot, and the demand proxy is coarse. The bestseller classifier is weak because bestsellers are very rare, so its probability is only reported, not used to rank. We recommend treating the launch as a test order of Q units, then reordering from the first weeks of real sales. This is the test-and-repeat logic of Weeks 3–4 (SHEIN).
