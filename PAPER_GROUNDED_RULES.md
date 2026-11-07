# ChainLens: Paper-Grounded Rule Analysis & Pivot Plan

> **Purpose:** Ground every ChainLens detection rule in published academic literature, identify gaps between current assumptions and published methods, and produce an actionable pivot plan with proper references.

---

## 1. Rule-by-Rule Literature Grounding

### R1: Circular Flow Detection (25 pts) — Wash Trading & Layering

#### Current Implementation
- Tarjan SCC → bounded cycle enumeration (3 ≤ k ≤ 5)
- Value tolerance: ±5% across all cycle edges
- Temporal window: 24 hours
- SCC size cap: 5,000 nodes

#### Grounding Papers

| # | Paper | Key Contribution | Year |
|---|-------|-----------------|------|
| **[P1]** | F. Victor & A.M. Weintraud, *"Detecting and Quantifying Wash Trading on Decentralized Cryptocurrency Exchanges"*, WWW '21, DOI: [10.1145/3442381.3449824](https://doi.org/10.1145/3442381.3449824) | **Defines** the SCC-based cycle detection method for wash trading on Ethereum DEXs. Identifies closed-loop trading patterns where the same entity controls multiple accounts and trades with itself. Uses Tarjan's SCC as the primary structural filter before cycle extraction. | 2021 |
| **[P2]** | M. Weber et al., *"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"*, KDD '19 Workshop, arXiv: [1908.02591](https://arxiv.org/abs/1908.02591) | Establishes the Elliptic Dataset benchmark. Demonstrates that graph topology (including cyclical flow patterns) is essential for AML detection beyond simple feature engineering. | 2019 |
| **[P3]** | Y. Hu et al., *"Characterizing and Detecting Money Laundering Activities on the Bitcoin Network"*, arXiv: [1912.12060](https://arxiv.org/abs/1912.12060) | Characterises money laundering as layered cyclic flows. Uses graph structural features including loops and PageRank to identify laundering intermediaries. | 2019 |

#### Alignment & Corrections

| Aspect | ChainLens (Current) | Literature (Victor & Weintraud [P1]) | Status |
|--------|---------------------|--------------------------------------|--------|
| SCC algorithm | Tarjan's via NetworkX | Tarjan's — same approach | ✅ Aligned |
| Cycle length bounds | 3–5 | Victor & Weintraud focus on **self-trade cycles of length 2–3** (two accounts trading back and forth). Longer cycles (4–5) are rarer in wash trading. | ⚠️ Consider adding length-2 cycles |
| Value tolerance | ±5% | Victor & Weintraud use **exact value matching** (same token, same amount minus gas fees). They also track volume fraction: what % of a DEX's total volume is wash-traded. | ⚠️ Tighten: use ±1–2% for ETH, exact match for ERC-20 |
| Temporal window | 24 hours | Victor & Weintraud don't use a fixed window; they analyse **all** historical trades per DEX. However, for AML layering, FATF guidance suggests **72-hour windows** for structuring. | ℹ️ 24h is conservative; consider configurable 24–72h |
| Metric reported | Binary: cycle member or not | Victor & Weintraud report **wash trade volume ratio** as a continuous metric per address. | 🔧 Enhance: report cycle volume as a proportion of total |

---

### R2: Outbound Fan-Out / Smurfing (20 pts)

#### Current Implementation
- Three-gate conjunction: `unique_counterparties ≥ 20` AND `tx_count_out ≥ 20` AND `fanout_ratio ≥ 0.5`

#### Grounding Papers

| # | Paper | Key Contribution | Year |
|---|-------|-----------------|------|
| **[P4]** | M. Starnini et al., *"Smurf-Based Anti-Money Laundering in Time-Evolving Transaction Networks"*, ECML PKDD 2021, DOI: [10.1007/978-3-030-86514-6_11](https://doi.org/10.1007/978-3-030-86514-6_11) | **The** definitive paper on smurf detection in transaction networks. Defines smurfing as **scatter-gather subgraph motifs** (fan-out → intermediary → fan-in). Introduces temporal database join-based detection that is orders of magnitude faster than subgraph isomorphism. | 2021 |
| **[P5]** | (Referenced in [P4]) General AML literature on structuring per FinCEN/FATF | Structuring = splitting transactions to avoid reporting thresholds (e.g., $10K USD for fiat; analogous thresholds for crypto compliance). | — |

#### Alignment & Corrections

| Aspect | ChainLens (Current) | Literature (Starnini et al. [P4]) | Status |
|--------|---------------------|-----------------------------------|--------|
| Detection model | Static three-gate threshold on features | Starnini defines smurfing as **temporal scatter-gather subgraph motifs**: (1) a source scatters funds to N intermediaries, (2) intermediaries forward to a collector, (3) all within a **short temporal window**. The full motif has **three layers**, not just fan-out. | ⚠️ Major gap: ChainLens only detects the scatter (fan-out) half, missing the gather phase |
| Thresholds | Hard-coded (≥20 counterparties, ≥20 txns) | Starnini uses **relative velocity thresholds**: the speed of fund dispersion matters more than absolute counts. Their method adapts to the transaction rate of the network. | ⚠️ Consider velocity-based thresholds instead of/alongside absolute counts |
| Temporal ordering | None (aggregated features) | Starnini's key insight: smurfing has **strict temporal ordering** (scatter happens before gather). Uses temporal joins. | 🔧 Add: timestamp ordering constraint |
| Weight | 20 pts | Not applicable (different scoring framework) | — |

> [!IMPORTANT]
> **Key finding:** The current R2 is a **simplified proxy** for smurfing. A literature-faithful implementation would detect the full scatter-gather motif with temporal ordering per Starnini et al. [P4]. However, the simplified fan-out gate is still a useful signal — it just catches *potential* scattering rather than *confirmed* smurfing.

---

### R3: Sanctioned Entity 1-Hop Proximity (30 pts)

#### Current Implementation
- Extract sanctioned addresses from training split labels
- Flag any address with a direct (1-hop) transaction to/from a sanctioned entity

#### Grounding Papers

| # | Paper | Key Contribution | Year |
|---|-------|-----------------|------|
| **[P6]** | E. Liu et al., *"Evasion Under Blockchain Sanctions"*, WWW '26, DOI: [10.1145/3774904.3792715](https://doi.org/10.1145/3774904.3792715), arXiv: [2507.11721](https://arxiv.org/abs/2507.11721) | Analyses OFAC sanctions on Tornado Cash across 6.79M Ethereum blocks. Finds sanctions reduced volume ~71% but attackers still used it in 78% of security incidents. Proposes **impurity-based quantitative scoring** (not binary blacklisting). | 2025 |
| **[P7]** | (Transaction Proximity approach), *"Transaction Proximity: A Graph-Based Approach to Blockchain Fraud Prevention"*, arXiv | Introduces "Transaction Proximity" metric measuring path distance to regulated entities. Demonstrates 1-hop structural proximity is a strong predictor of fraud exposure. | 2024 |
| **[P8]** | (Sanctions impact study), *"Assessing the Impact of Sanctions in the Crypto Ecosystem"*, arXiv | Analyses 1-hop and 2-hop address-transaction graphs of sanctioned actors. Shows sanctioned entities rely on predictable 1-hop pathways. | 2024 |

#### Alignment & Corrections

| Aspect | ChainLens (Current) | Literature ([P6], [P7], [P8]) | Status |
|--------|---------------------|-------------------------------|--------|
| Proximity metric | Binary: 1-hop yes/no | Liu et al. [P6] advocate **quantitative impurity scoring**: what fraction of a wallet's inflows originated from sanctioned sources, decayed by hop count. | ⚠️ Enhance: compute exposure fraction, not just binary |
| Seed source | Training-split sanctioned labels only | Should use a **maintained sanctions list** (OFAC SDN, EU lists, Chainalysis/Elliptic oracles) independent of training labels. | 🔧 Fix: decouple from training split |
| Hop depth | Strictly 1-hop | Literature supports 1-hop as the **strongest signal** [P7][P8], but recommends 2-hop with decay for comprehensive coverage [P6]. | ℹ️ 1-hop is well-supported; optional 2-hop extension |
| Dusting attack resilience | Not addressed | Liu et al. [P6] specifically warn about **dusting attacks**: adversaries send tiny amounts from sanctioned wallets to innocent targets. Need a **minimum value threshold** to avoid false positives. | 🔧 Add: minimum value gate (e.g., ≥0.01 ETH) |
| Weight | 30 pts (highest weight) | Appropriate — direct sanctioned exposure is the highest-severity signal per all papers. | ✅ Aligned |

---

### R4: Value-Weighted PageRank (15 pts) — Network Prominence

#### Current Implementation
- PageRank with α=0.85, value-weighted edges, tol=1e-6
- Flag top 5% (≥95th percentile)

#### Grounding Papers

| # | Paper | Key Contribution | Year |
|---|-------|-----------------|------|
| **[P9]** | D. Lin et al., *"Modeling and Understanding Ethereum Transaction Records via a Complex Network Approach"*, IEEE TCSII, DOI: [10.1109/TCSII.2020.2968376](https://doi.org/10.1109/TCSII.2020.2968376) | Applies PageRank to the Ethereum transaction network as a complex network. Identifies key hub addresses and demonstrates power-law distribution of PageRank scores. Validates that **value-weighted PageRank** captures flow-routing importance. | 2020 |
| **[P3]** | Y. Hu et al. (same as R1 reference) | Uses PageRank as one of the **curated graph features** for money laundering detection. Shows PageRank centrality distinguishes laundering intermediaries from normal nodes. | 2019 |
| **[P10]** | J. Lorenz et al., *"Machine Learning Methods to Detect Money Laundering in the Bitcoin Blockchain in the Presence of Label Scarcity"*, ICAIF '20, DOI: [10.1145/3383455.3422549](https://doi.org/10.1145/3383455.3422549) | Uses graph centrality features (including PageRank) in ensemble classifiers (Random Forest, Active Learning) for AML. Demonstrates centrality-based features **reduce false negatives** vs. transaction-only features. | 2020 |

#### Alignment & Corrections

| Aspect | ChainLens (Current) | Literature ([P9], [P3], [P10]) | Status |
|--------|---------------------|-------------------------------|--------|
| Algorithm | Standard PageRank with value weights | Lin et al. [P9] validate value-weighted PageRank on Ethereum. Same approach. | ✅ Aligned |
| Damping factor α | 0.85 | Standard across all cited papers (original Page & Brin, 1998). | ✅ Aligned |
| Threshold | Top 5% (95th percentile) | Hu et al. [P3] use PageRank as a continuous feature for ML classifiers, not a binary gate. Lorenz et al. [P10] similarly use it as a feature. **No paper uses a fixed 5% cutoff.** | ⚠️ Consider: use as a continuous feature contributing proportionally to score, rather than binary |
| Interpretation | "Hub identification in fund-routing rings" | Lin et al. [P9] show high-PageRank nodes include **exchanges, DeFi routers, and bridges** — legitimate entities that are high-volume hubs. Need **known-address filtering** before flagging. | ⚠️ Important: must combine with known-address exclusion (already done via downweighting, but could be stricter) |
| Weight | 15 pts | Appropriate as a supporting signal. Papers consistently show PageRank is useful but **not** a standalone AML indicator. | ✅ Aligned |

---

### R5: Apriori Association Rule Mining (20 pts) — Behavioural Profiling

#### Current Implementation
- Binarise 9 behavioural items per wallet → Apriori frequent itemsets (min_support=0.05) → association rules (confidence ≥ 0.80, lift ≥ 2.0)
- Mined on training-split positives only

#### Grounding Papers

| # | Paper | Key Contribution | Year |
|---|-------|-----------------|------|
| **[P11]** | R. Agrawal & R. Srikant, *"Fast Algorithms for Mining Association Rules"*, VLDB 1994 | **Original Apriori algorithm paper.** Establishes the min_support/min_confidence framework for frequent itemset mining. | 1994 |
| **[P2]** | M. Weber et al. (same as R1) | Feature engineering + ML pipeline for AML. The Elliptic benchmark demonstrates that **combining multiple behavioural signals** (analogous to association rule antecedents) improves detection. | 2019 |
| **[P10]** | J. Lorenz et al. (same as R4) | Shows that combining multiple weak signals via ensemble methods outperforms individual features for AML with label scarcity. Conceptually similar to association rules. | 2020 |

#### Alignment & Corrections

| Aspect | ChainLens (Current) | Literature | Status |
|--------|---------------------|-----------|--------|
| Algorithm | MLxtend Apriori + association_rules | Standard Apriori [P11]. Well-established in data mining. | ✅ Aligned |
| min_support | 0.05 | Reasonable for a small labelled set. Agrawal & Srikant [P11] recommend empirically tuning. | ✅ Reasonable |
| min_confidence | 0.80 | Strong threshold. Papers suggest 0.7–0.9 range for fraud detection to balance precision and coverage. | ✅ Aligned |
| min_lift | 2.0 | Good — ensures rules are at least 2x better than random. | ✅ Aligned |
| max_antecedent_size | 3 | Prevents overfitting to narrow patterns. | ✅ Aligned |
| Items used | 9 boolean features (high_fanout, high_tx_count, low_value, high_value, many_round, is_contract, recent_account, high_pagerank, dbscan_noise) | **No specific paper** uses this exact item set for blockchain association mining. This is ChainLens-specific feature engineering. Items are reasonable proxies for known suspicious behaviours documented across [P2][P3][P10]. | ℹ️ Novel contribution — well-motivated but not directly from a single paper |
| Training approach | Mined on labelled train-split positives | Standard supervised-to-unsupervised transfer learning. Sound methodology. | ✅ Aligned |

> [!NOTE]
> R5 is the most **novel** rule in ChainLens. While Apriori itself is well-established [P11], applying it to binarised blockchain behavioural profiles for AML is a creative adaptation without a single direct precedent paper. The approach is well-motivated by the ensemble/multi-signal philosophy demonstrated in [P2] and [P10].

---

### Unsupervised: DBSCAN Density Clustering

#### Grounding Papers

| # | Paper | Key Contribution | Year |
|---|-------|-----------------|------|
| **[P12]** | M. Ester et al., *"A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise"*, KDD '96 | **Original DBSCAN paper.** Defines core points, border points, and noise. | 1996 |
| **[P13]** | M. Ostapowicz & K. Żbikowski, *"Detecting Fraudulent Accounts on Blockchain: A Supervised Approach"*, WISE 2019, DOI: [10.1007/978-3-030-34223-4_2](https://doi.org/10.1007/978-3-030-34223-4_2) | Uses Ethereum account features for fraud detection. While this paper uses supervised methods (RF, SVM, XGBoost), the **feature engineering** (tx count, value, counterparties) directly informs the feature space used by ChainLens's DBSCAN. | 2019 |
| **[P14]** | J. Hirshman, Y. Huang, S. Macke, *"Unsupervised Approaches to Detecting Anomalous Behavior in the Bitcoin Transaction Network"*, Stanford CS229 Technical Report | Early application of unsupervised ML (k-means, Mahalanobis, unsupervised SVM) to Bitcoin transaction graph anomaly detection. Demonstrates that **density-based approaches isolate genuine outliers** in blockchain data. | 2013 |

#### Alignment
ChainLens's DBSCAN usage (noise label → Apriori item) is a **creative pipeline connection** not found in a single paper. The feature space (5 normalised features) and the noise-as-signal concept are well-supported by [P12][P14].

---

### Community Detection: Louvain Modularity

#### Grounding Papers

| # | Paper | Key Contribution | Year |
|---|-------|-----------------|------|
| **[P15]** | V.D. Blondel et al., *"Fast Unfolding of Communities in Large Networks"*, J. Statistical Mechanics, DOI: [10.1088/1742-5468/2008/10/P10008](https://doi.org/10.1088/1742-5468/2008/10/P10008) | **Original Louvain algorithm paper.** Modularity maximisation via greedy agglomeration. | 2008 |
| **[P16]** | S. Phetsouvanh, F. Oggier, A. Datta, *"EGRET: Extortion Graph Exploration Techniques in the Bitcoin Network"*, IEEE ICDMW 2018 | Applies graph exploration and community clustering to identify ransomware/extortion networks in Bitcoin. Validates community detection for isolating **colluding address clusters**. | 2018 |

#### Alignment
Louvain is used display-only (non-scoring) in ChainLens — this is appropriate. Papers [P15][P16] support its use for analyst-facing community identification.

---

## 2. Composite Scoring Methodology Grounding

The additive weighted scoring formula:

$$\text{score}(a) = \text{clamp}\Big(\sum_{i=1}^{5} w_i \cdot \mathbb{1}_{R_i(a)}\Big)$$

is consistent with the **multi-criteria weighted risk scoring** approach described across AML compliance literature and industry practice (FATF guidance, Chainalysis/Elliptic methodologies). No single paper prescribes the exact weights [25, 20, 30, 15, 20], but the **relative ordering** (sanctioned proximity > cycle detection > fan-out ≈ association rules > PageRank) is consistent with regulatory severity rankings where **direct illicit exposure is the highest-severity signal**.

---

## 3. Full Bibliography

| ID | Authors | Title | Venue | Year | DOI / Link |
|----|---------|-------|-------|------|------------|
| P1 | F. Victor, A.M. Weintraud | Detecting and Quantifying Wash Trading on Decentralized Cryptocurrency Exchanges | WWW '21 | 2021 | [10.1145/3442381.3449824](https://doi.org/10.1145/3442381.3449824) |
| P2 | M. Weber et al. | Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics | KDD '19 Workshop | 2019 | [arXiv:1908.02591](https://arxiv.org/abs/1908.02591) |
| P3 | Y. Hu et al. | Characterizing and Detecting Money Laundering Activities on the Bitcoin Network | arXiv preprint | 2019 | [arXiv:1912.12060](https://arxiv.org/abs/1912.12060) |
| P4 | M. Starnini et al. | Smurf-Based Anti-Money Laundering in Time-Evolving Transaction Networks | ECML PKDD 2021 | 2021 | [10.1007/978-3-030-86514-6_11](https://doi.org/10.1007/978-3-030-86514-6_11) |
| P5 | FATF | FATF Guidance on Virtual Assets and VASPs | FATF-GAFI | 2021 | [fatf-gafi.org](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Guidance-rba-virtual-assets-2021.html) |
| P6 | E. Liu et al. | Evasion Under Blockchain Sanctions | WWW '26 | 2025 | [10.1145/3774904.3792715](https://doi.org/10.1145/3774904.3792715) |
| P7 | — | Transaction Proximity: A Graph-Based Approach to Blockchain Fraud Prevention | arXiv preprint | 2024 | arXiv (see search results) |
| P8 | — | Assessing the Impact of Sanctions in the Crypto Ecosystem | arXiv preprint | 2024 | arXiv (see search results) |
| P9 | D. Lin et al. | Modeling and Understanding Ethereum Transaction Records via a Complex Network Approach | IEEE TCSII | 2020 | [10.1109/TCSII.2020.2968376](https://doi.org/10.1109/TCSII.2020.2968376) |
| P10 | J. Lorenz et al. | Machine Learning Methods to Detect Money Laundering in the Bitcoin Blockchain in the Presence of Label Scarcity | ICAIF '20 | 2020 | [10.1145/3383455.3422549](https://doi.org/10.1145/3383455.3422549) |
| P11 | R. Agrawal, R. Srikant | Fast Algorithms for Mining Association Rules | VLDB '94 | 1994 | — (seminal, pre-DOI) |
| P12 | M. Ester et al. | A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise | KDD '96 | 1996 | — (seminal, pre-DOI) |
| P13 | M. Ostapowicz, K. Żbikowski | Detecting Fraudulent Accounts on Blockchain: A Supervised Approach | WISE 2019 | 2019 | [10.1007/978-3-030-34223-4_2](https://doi.org/10.1007/978-3-030-34223-4_2) |
| P14 | J. Hirshman, Y. Huang, S. Macke | Unsupervised Approaches to Detecting Anomalous Behavior in the Bitcoin Transaction Network | Stanford CS229 Report | 2013 | — (technical report) |
| P15 | V.D. Blondel et al. | Fast Unfolding of Communities in Large Networks | J. Stat. Mech. | 2008 | [10.1088/1742-5468/2008/10/P10008](https://doi.org/10.1088/1742-5468/2008/10/P10008) |
| P16 | S. Phetsouvanh et al. | EGRET: Extortion Graph Exploration Techniques in the Bitcoin Network | IEEE ICDMW 2018 | 2018 | IEEE proceedings |

---

## 4. Recommended Rule Refinements for the Wallet Analyser

Based on the literature analysis, here are the paper-backed corrections to apply when building the wallet analyser:

### Priority Corrections

All eight priority corrections are implemented in this revision.

| Rule | Change | Paper Basis | Status | Implementation |
|------|--------|-------------|--------|----------------|
| **R1** | Add length-2 cycle detection (self-trade pairs) | [P1] Victor & Weintraud | ✅ Applied | `detect_cycles` default/min config length 2; canonical cycle de-duplication |
| **R1** | Tighten value tolerance to ±2% for ETH | [P1] | ✅ Applied | `config/rules.yaml` → `value_tolerance: 0.02` |
| **R2** | Add temporal ordering check (scatter must precede gather) | [P4] Starnini et al. | ✅ Applied | `_temporal_gather_evidence` checks full scatter→gather motif inside a 72 h window |
| **R3** | Add minimum value gate (≥0.01 ETH) to resist dusting attacks | [P6] Liu et al. | ✅ Applied | `detect_sanctioned_1hop(min_value_eth=0.01)` |
| **R3** | Decouple sanctions list from training split; use maintained OFAC/darklist | [P6][P8] | ✅ Applied | `config/sanctions.json` (frozen OFAC snapshot) unioned with labelled seeds; no split filter |
| **R3** | Compute exposure fraction (% of inflow from sanctioned sources) instead of binary | [P6] Liu et al. | ✅ Applied | `inflow_exposure_fraction` / `outflow_exposure_fraction` scale R3 strength, with a 0.10 direct-link floor |
| **R4** | Use PageRank as continuous input rather than binary top-5% gate | [P3][P10] Hu, Lorenz | ✅ Applied | Upper-tail strength `(percentile − 90) / 10`; R4 contribution = 15 × strength |
| **R4** | More aggressive known-address exclusion before PageRank flagging | [P9] Lin et al. | ✅ Applied | Known exchange/bridge/mixer/contract addresses are never R4-flagged |

### Optional Enhancements

| Enhancement | Paper Basis | Effort | Status |
|-------------|-------------|--------|--------|
| Full scatter-gather motif detection for R2 | [P4] Starnini | High | ✅ Implemented as the R2 ordering gate (scatter + gather + 72 h window) |
| 2-hop sanctioned proximity with decay for R3 | [P6][P8] | Medium | ⏳ Not implemented (1-hop remains the strongest signal) |
| SHAP-based explanation of contribution weights | General XAI literature | Medium | ⏳ Not implemented |

### Configuration added by the refinements

| Config key | Value | Purpose |
|---|---:|---|
| `R1.cycle_length_min` | `2` | Capture two-account wash-trading pairs |
| `R1.value_tolerance` | `0.02` | ETH transfer amounts match within gas-fee noise |
| `R2.require_temporal_ordering` | `true` | Require scatter→gather ordering |
| `R2.gather_window_hours` | `72` | Maximum scatter-to-gather delay |
| `R2.min_gather_fraction` | `0.10` | Fraction of scattering intermediaries that must forward |
| `R3.sanctions_path` | `config/sanctions.json` | Frozen OFAC snapshot, independent of train/test |
| `R3.min_value_eth` | `0.01` | Dusting-attack floor |
| `R3.min_direct_strength` | `0.10` | Severity floor for a confirmed direct link |
| `R4.continuous` | `true` | Continuous upper-tail PageRank contribution |
| `R4.min_percentile` | `90` | Lower bound of the continuous PageRank tail |
| `R4.min_strength` | `0.05` | Discard negligible PageRank tail mass |
| `R4.exclude_known_categories` | `[exchange, bridge, mixer, contract]` | Never flag benign high-volume hubs on PageRank |

### Scoring consequence

The five rule weights remain locked at `[25, 20, 30, 15, 20]`, but R3 and R4 now
contribute a fraction of their weight in proportion to evidence strength:
`contribution = weight × strength`, with `strength ∈ [0, 1]`. R1, R2 and R5
remain binary (strength 1.0). This preserves the locked maximum while allowing
exposure and centrality to degrade gracefully instead of acting as hard gates.
`raw_score` is therefore fractional; the final 0–100 score is still rounded and
downweighted exactly as before.

---

## 5. Updated Time Estimate (with paper-backed corrections)

| Phase | Original | With Rule Corrections | Delta |
|-------|----------|-----------------------|-------|
| Phase 0: Freeze artefacts | 2–3h | 3–4h (also freeze sanctions list, pagerank threshold) | +1h |
| Phase 1: On-demand ingestion | 4–6h | 4–6h (unchanged) | — |
| Phase 2: Adapt rules | 6–8h | **8–12h** (add dusting gate, value tolerance tightening, continuous PageRank, exposure fraction) | +2–4h |
| Phase 3: Explanation renderer | 4–6h | 5–7h (richer explanations citing paper methodology) | +1h |
| Phase 4: Dashboard | 8–12h | 8–12h (unchanged) | — |
| Phase 5: Testing | 4–6h | 5–7h (test new rule behaviours) | +1h |
| **Total (full product)** | **28–41h** | **33–48h (≈6–8 working days)** | +5–7h |
| **Total (CLI MVP)** | **16–23h** | **20–29h (≈4–5 working days)** | +4–6h |

> [!IMPORTANT]
> Adding the paper-backed corrections increases the estimate by about **1–2 days**, but produces a significantly more defensible and academically rigorous system. The biggest additions are the dusting-attack gate for R3, exposure fraction computation, and temporal ordering for R2.

---

## 6. Verification After Applying the Refinements

The refined rules were validated with `uv run pytest tests/ -q` (45 passed) and
then re-run on the cached real cohort with
`uv run python run.py --mode real --skip-fetch` (no new Etherscan calls).

| Signal / metric | Before refinements | After refinements |
|---|---:|---:|
| R1 cycle members | 0 | 163 (97 qualifying cycles) |
| R2 fan-out members | 240 | 33 (scatter→gather confirmed) |
| R3 sanctioned 1-hop members | 2,219 | 2,014 (dust-filtered, exposure-weighted) |
| R4 PageRank members | 4,108 (binary top 5%) | 7,758 (continuous ≥90th pct tail; 4,100 top-5%) |
| Notable | no cycles | length-2 cycles captured, ±2% tolerance |
| Test average precision | 0.5876 | **0.6181** |
| Medium-tier precision | 0.5000 | **0.5455** |
| Medium-tier recall | 0.0539 | 0.0223 (stricter, continuous severity) |
| High-tier F1 | 0.0000 | 0.0000 (still no test positive ≥60) |

The recall drop at the fixed Medium threshold is expected: continuous exposure
and centrality now reduce weak links below 25 instead of granting the full
weight. Ranking quality (average precision) improved. The High tier remains
empty because no test-set positive accumulates enough independent signals at
the locked 60-point threshold; this is a data/threshold interaction, not a
regression in the rules.
