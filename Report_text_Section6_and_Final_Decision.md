# Report text: Section 6 and Final Decision

_Moved out of the notebook. Paste into the report and edit to fit; check the claims against your Colab outputs._

## 6. Recommend initial inventory

**Demand scenarios.** The demand model's prediction is only a point estimate. To get a range of outcomes we:
1. Get **out-of-fold (5-fold CV) predictions** for every training product, using the same Random Forest settings as `demand_model`, and compute the residuals `actual − predicted`.
2. Residuals are not the same size everywhere: listings with a high predicted demand have much bigger errors. So we split the training predictions into **10 bins by predicted demand** and keep each bin's residuals separately.
3. For each test product, its scenarios are `max(0, prediction + r)` for every residual `r` in its bin, about 740 equally likely 30-day demand scenarios. Clipping at 0 keeps demand nonnegative, and the clipped values put a realistic amount of probability on "no sales".

**Cost assumptions (common brief assumptions)**

| Symbol | Meaning | Value |
|---|---|---|
| p | planned selling price | listing `price` |
| c | unit procurement cost | 0.45 p |
| s | recovery (salvage) value of an unsold unit | 0.10 c = 0.045 p |
| Cu = p − c | underage cost (margin lost per unit short) | 0.55 p |
| Co = c − s | overage cost (loss per unsold unit) | 0.405 p |
| CR = Cu / (Cu + Co) | critical ratio | 0.55 / 0.955 = **0.576** |

Q is the CR-quantile of the scenario distribution, rounded up to a whole unit. Because c and s are both proportional to p, the CR is the **same for every product**. Price changes the size of the money at stake but not the quantile chosen.

**How costs are counted (no double counting).** Expected profit = p·E[min(Q,D)] + s·E[(Q−D)+] − c·Q, which is the same as (p−c)·E[D] − Co·E[(Q−D)+] − Cu·E[(D−Q)+]. Procurement spend cQ is a **budget**, not an extra cost. The cost of unsold units is already in the overage term Co, so we do not add cQ to the mismatch cost.

Other assumptions: one order with no reorder in the 30 days, no holding cost or FBA fees beyond c, and lost sales are not backlogged. `boughtInLastMonth` is a coarse, rounded-down signal ("50+", "100+"), so these quantities are probably conservative.

## Final decision: which test-set design to launch

The test products stand in for **new designs whose demand we do not know yet**. We pick one using only information a seller has before launch:

* **Newsvendor economics (Section 6):** expected 30-day profit at the optimal Q, and return on the procurement budget (ROI = expected profit / cQ).
* **Model outputs (Section 5):** predicted demand proxy and predicted bestseller probability.
* **Competitive landscape (Section 4):** the design's image cluster and that cluster's **training-set** statistics (listings, demand per listing, A-class share).
* **Nearest competing listings:** the 10 most similar **training** listings by cosine similarity of the image and title embeddings (average of the two). Their median demand shows whether similar designs already sell. Their average similarity shows how crowded the niche is.

**Screening rules.** The design must have positive expected profit, at least a 50% chance of selling 50 or more units in its first 30 days, and must not be a near-copy of an existing listing (top-1 similarity < 0.95).

**Ranking.** Each remaining design gets an opportunity score, a weighted average of percentile ranks:
30% expected profit + 20% ROI + 15% bestseller probability + 15% neighbours' median demand + 10% low crowding + 10% cluster demand per listing.
The weights are a judgement call. The shortlist table lets you check that the top choice does not depend on them.

### Why not just copy the most popular observed item?

*(Check this text against the numbers printed in F.5 before you submit it.)*

1. **A top seller's demand comes from things a copy does not get.** The `boughtInLastMonth` of the best-selling training item reflects its brand, thousands of reviews, its ranking history and Buy Box position. None of these carry over to a new listing. F.5 shows this in the data: the training listings that *look* most like the top seller sell much less than it does (see their median demand and share with zero demand). The demand we see for the top seller is close to the most a design like it can achieve, not what a new listing should expect.
2. **Copying puts us in the most crowded part of the map.** The look-alike sits right next to the leader and many near-substitutes (high crowding score). New listings there compete on price against established sellers. The recommended design sits where similar listings already show real demand but the closest substitutes are fewer or less similar, which matches the "meaningful demand and fewer close substitutes" cell of the Section 4 opportunity map.
3. **We judge risk-adjusted profit, not popularity.** The ranking uses newsvendor expected profit, which already prices in the chance of unsold stock (Co = 0.405p per unit) and lost sales (Cu = 0.55p per unit). It also uses ROI on the procurement budget. A high-demand but uncertain copycat needs a large Q and ties up more capital for a similar or lower expected return.
4. **No look-ahead.** The "most popular item" rule uses *observed* demand. For a design that has not launched, all we have is a forecast and its error distribution (the CV residuals). Our choice uses only pre-launch information, so it is a decision we could actually have made.

**Caveats.** The data is a single snapshot, so the holdout test does not prove how a future launch will do. The demand proxy is coarse and rounded. The bestseller classifier is weak (balanced accuracy ≈ 0.5), so its probability is only a minor input. We recommend treating the launch as a **test order of Q units**, then reordering based on the first weeks of real sales.
