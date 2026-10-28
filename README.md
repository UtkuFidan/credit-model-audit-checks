# credit-model-audit-checks

Re-performance checks for credit-risk scorecards. They run the tests an internal auditor or model
validator should not take on trust: stability, discrimination, calibration, rating-grade structure,
overrides, data quality, and a benchmark against a challenger model. Everything is computed from
record-level data, and the output is an audit-style report with an observation, a risk rating and a
recommendation for each finding, plus a traffic-light summary by area.

The repository includes a worked example. I build a reference WoE scorecard on a public credit
dataset, build a gradient-boosted challenger on the same rows, score a later "monitoring" sample with
documented drift injected into it, and run the checks. The same pipeline is also run with no drift at
all, so the false-alarm rate of the checks is on the page next to their detection rate. Every number
in this README comes from running the code.

> Demonstration only. The scorecard, the drift and the overrides are built or simulated on public
> data. Nothing here describes a production model or a real institution.

## Why I built this

In audit work I kept seeing model reviews that stopped at the documentation. The development metrics
were fine, but nobody had re-run anything on current data. I wanted a set of checks I could run
myself, with thresholds written down, so that a finding rests on evidence I produced and not on
assurance I inherited. Averages also hide things: in the drifted run below the score PSI is 0.168
(Watch) while one characteristic has a CSI of 0.295 (Fail).

The benchmark against a challenger is there because it is the cheapest honest question a validator can
ask of a scorecard: is there signal in the same data that the model leaves unused, and how much?

## Check catalogue

Thresholds are defaults in `config.yaml` and `src/model_audit/config.py`. "Convention" means a common
rule of thumb or a setting I picked for this repo. Align them with your own model-monitoring policy
before relying on them.

| ID | Check | Question it answers | Threshold (Watch / Fail) | Reference |
|---|---|---|---|---|
| ST-01 | Score PSI | Has the score distribution moved since development? | 0.10 / 0.25 | Siddiqi (2006) |
| ST-02 | CSI per characteristic | Which inputs moved, and by how many points on average? | 0.10 / 0.25 | Siddiqi (2006) |
| DI-01 | Gini drift | Is rank-ordering power intact out of time? Bootstrap CI on each Gini and on the drop | drop 0.05 / 0.10 | Convention |
| DI-02 | KS drift | Is the good/bad separation intact? Same bootstrap treatment | drop 0.03 / 0.07 | Convention |
| CA-01 | Hosmer-Lemeshow | Are PDs calibrated across the score range? | p 0.05 / 0.01 | Hosmer and Lemeshow (1980) |
| CA-02 | Binomial test per grade | Do observed defaults exceed a grade's PD? Holm-adjusted p reported | none below 0.05 / two grades at 0.01 | BCBS WP14 (2005) |
| CA-05 | Jeffreys test per grade | Same question, Bayesian: posterior probability that the true PD is at or below the grade PD | none below 0.05 / two grades at 0.01 | Brown, Cai and DasGupta (2001); Tasche (2008) |
| CA-03 | Brier score | Has overall probabilistic accuracy deteriorated? | +5% / +15% vs dev | Brier (1950) |
| CA-04 | Calibration in the large | Is the portfolio PD biased? | abs(obs/pred - 1) 10% / 25% | Convention |
| RG-01 | Grade concentration (HHI) | Does the master scale still differentiate the book? | 0.15 / 0.25 | HHI bands, convention |
| RG-02 | Monotonicity of default rates | Do default rates rise with worsening grade? | 1 / 2 inversions | BCBS WP14 (2005) |
| OV-01 | Override rate | Is the score being bypassed too often? | 5% / 10% | Convention |
| OV-02 | Override performance | Do low-side overrides add value over the score? | Fisher test: better = Pass, no evidence = Watch, worse = Fail | Convention |
| DQ-01 | Missing values | Is any input missing beyond tolerance? | 1% / 5% per column | Convention |
| DQ-02 | Out-of-range values | Are there values outside documented domains? | >0% / >0.1% | Convention |
| DQ-03 | Duplicate IDs | Is the account key unique? | >0% / >0.1% | Convention |
| BM-01 | Challenger gap, monitoring | Gini of challenger minus champion on monitoring data, paired bootstrap CI | 0.03 / 0.06 | Convention |
| BM-02 | Challenger gap, development CV | Same gap out of fold, with the scorecard re-binned and refitted in every fold | 0.03 / 0.06 | Convention |

Every check returns a `CheckResult` with value, status (Pass / Watch / Fail), threshold text,
rationale, observation and recommendation. Boundaries are inclusive, so a PSI of exactly 0.10 is
Watch. A check that cannot be computed (NaN) is a Fail, not a pass.

The traffic light rolls the checks up by area. An area is Red if any of its checks fails, Amber if any
is Watch, Green otherwise, and the overall rating is the worst area. It is a reading aid. The findings
table is where the evidence is.

## Data

UCI "Default of Credit Card Clients" (Taiwan, 2005), UCI id 350, licence CC BY 4.0 (Yeh and Lien,
2009; https://doi.org/10.24432/C55S3H). 30,000 accounts, 23 predictors, binary default flag.

I chose it over "Give Me Some Credit" because the licence is unambiguous. The OpenML mirror of that
one carries Kaggle competition terms and a licence field that only says "Public". I do not use sex or
marital status as scorecard or challenger inputs.

The raw data is not committed. `model-audit download` fetches it from UCI and caches it as
`data/raw/taiwan_default.csv` (git-ignored).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,model,dev]"

model-audit download                              # fetch + cache the public dataset
model-audit run --config config.yaml              # drifted run: report + figures in reports/, docs/
model-audit run --config config.control.yaml      # control run: no drift, reports/control/
model-audit run --config config.idorder.yaml      # ordered holdout, reports/idorder/
pytest -q && ruff check .                         # tests run offline on a synthetic fixture
```

The `model` extra installs `optbinning`, `lightgbm` and `optuna`. Without it, set
`scorecard.binning: builtin` and `challenger.enabled: false` and everything else still runs. A full
run takes a few minutes on a laptop, mostly the Optuna search and the cross-validated champion.

To audit your own scores, call the functions in `src/model_audit/checks/` (plain arrays and a config
dict) and `model_audit.metrics` for the statistics.

## How the worked example is built

1. Reference scorecard (`scorecard.py`). Optimal binning with `optbinning` (exact solve, monotone
   event rate per characteristic, at least 5% of accounts per bin, at most 6 bins, separate missing
   bin). WoE is computed in the repo so that the sign convention is one and the same. Thirteen
   candidate characteristics: limit, age, the six repayment statuses, and five derived ratios
   (latest repayment ratio, six-month repayment ratio, utilisation, payment to limit, bill trend). A
   logistic regression on WoE with the L1/L2 mix and C chosen by 5-fold cross-validated log loss
   (ridge, C = 0.1 won). A characteristic with a wrong-sign coefficient is dropped and recorded; here
   AGE (its monotone binning runs against its effect once the other inputs are in) and the six-month
   repayment ratio went. Points are scaled to 600 at 50:1 good:bad odds with PDO 20. The
   score-to-PD map is a cubic logistic recalibration on the development sample: with the points scale
   alone the model fails Hosmer-Lemeshow in development, because shrinkage compresses the slope.
2. Challenger (`challenger.py`). LightGBM on every raw predictor except sex and marital status, plus
   the derived ratios. Forty Optuna TPE trials on 5-fold log loss, seed fixed. The champion-challenger
   comparison out of fold uses a second set of folds, so that the tuning folds are not reused for the
   headline number. Tuning on the same rows still leaves a little optimism for the challenger, which
   works against the scorecard.
3. Development / monitoring split (`simulate.py`). A random 50/50 split (or an ID-ordered one, below).
4. Drift, injected into the monitoring half only. The data has no real later period (see below), so
   these are known perturbations the checks should find:

   | Injection | Detected by |
   |---|---|
   | 80% of accounts with `LIMIT_BAL >= 200,000` removed | CSI LIMIT_BAL, score PSI |
   | `PAY_AMT1` multiplied by 0.6 | CSI PAY_RATIO |
   | 5% of non-defaulting current payers relabelled as defaults | Gini/KS, calibration, Brier |
   | `PAY_AMT1` blanked for ~3% of rows | DQ-01 |
   | `AGE = 999` sentinel for ~0.3% of rows | DQ-02 |
   | 20 rows duplicated | DQ-03 |

5. Master scale: 8 grades from fixed development score percentiles; grade PD is the mean calibrated
   PD in development.
6. Overrides (simulated): policy declines the lowest 25% of development scores; 18% of those are
   overridden to approve and 2% of the rest are overridden to decline, chosen at random, so overrides
   carry no information by construction.

### Why the monitoring sample is perturbed

The file has no genuine later period. The only temporal content is the six months of bills,
payments and repayment status inside each row, and the label is a single next-month outcome, so
there is no second cohort to hold out. I looked at `ID` as a possible time axis: default rate and
the mean of `PAY_0` do move between ID blocks (default rate 22.5% in the first half of the file, 21.7%
in the second), but nothing in the source says why, so I do not call it time. The perturbations above
are the controlled substitute, where the right answer is known. Two more runs bracket them: a random
split with nothing injected (what a false alarm looks like) and the ID-ordered holdout (a shift that is
in the data as published).

## Results (seed 42)

Development 15,000 accounts (default rate 21.66%); monitoring 10,505 rows (default rate 28.78%, mean
predicted PD 25.08%). 28 checks: 7 Fail, 10 Watch, 11 Pass. Overall rating Red.

| Area | Rating | Fail / Watch / Pass |
|---|---|---|
| Stability | Red | 1 / 2 / 9 |
| Discrimination | Amber | 0 / 2 / 0 |
| Calibration | Red | 4 / 1 / 0 |
| Rating structure | Green | 0 / 0 / 2 |
| Overrides | Amber | 0 / 2 / 0 |
| Data quality | Red | 2 / 1 / 0 |
| Benchmark | Amber | 0 / 2 / 0 |

| Area | Result |
|---|---|
| Score stability | Score PSI 0.168 (Watch). CSI: LIMIT_BAL 0.295 (Fail), PAY_RATIO 0.217 (Watch), every other characteristic below 0.05 (Pass) |
| Discrimination | Gini 0.520 (95% CI 0.502-0.540) in development vs 0.445 (0.424-0.465) in monitoring, drop 0.075 (CI of the drop 0.046-0.102, Watch). KS 0.402 vs 0.349, drop 0.053 (0.027-0.084, Watch) |
| Calibration | Predicted default rate 25.08% vs observed 28.78% (Watch). Hosmer-Lemeshow p = 7e-29 (Fail). Binomial test: 5 grades at p <= 0.01 (Fail). Jeffreys test: 6 grades (Fail). Brier +26.7% vs development (Fail) |
| Ratings | HHI 0.147 against a floor of 0.125 (Pass); no monotonicity inversions (Pass) |
| Overrides | 7.0% override rate (Watch). Low-side overrides default at 50.8% vs 53.3% for non-overridden accounts in the same zone, one-sided p = 0.147: no evidence of value added (Watch) |
| Data quality | PAY_AMT1 and PAY_RATIO 2.9% missing (Watch); 0.38% out-of-range AGE (Fail); 20 duplicate IDs (Fail) |
| Benchmark | Challenger Gini 0.485 vs champion 0.445 on monitoring data, gap 0.040 (paired bootstrap CI 0.027-0.053, Watch). Out of fold on development data, 0.565 +/- 0.008 vs 0.516 +/- 0.008, gap 0.049 +/- 0.006, challenger ahead in 5 of 5 folds (Watch) |

The score PSI alone is only a Watch. The per-characteristic CSI isolates what moved, and the
outcome-based checks show the model has become less discriminating and too optimistic. The benchmark
gap is a different kind of finding: it is already there in the control run below, so it comes from the
scorecard structure, not from the drift. The full findings table is in
[`reports/model_audit_report.md`](reports/model_audit_report.md).

### Control run: nothing injected

Same code, random split, no drift, no overrides (`config.control.yaml`, report in
`reports/control/`). The checks should be quiet here, and mostly they are: 0 Fail,
5 Watch, 21 Pass on 26 checks. The Watches are Hosmer-Lemeshow (p = 0.043), the riskiest
rating grade, whose default rate (71.3% vs 67.6%) is above its PD at p = 0.001 under both the binomial
and the Jeffreys test, and the two benchmark checks (challenger gap 0.053 on monitoring data, 0.049 out of fold). Gini is 0.520 in
development and 0.524 in monitoring, score PSI 0.001. So with 15,000 accounts per sample a clean
split still produces a few Watch flags from the grade-level tests, which is why I report Holm-adjusted
p-values next to the raw ones and why a single Watch on a calibration test is a prompt to look, not a
finding. The benchmark gap is real and stable across folds: the scorecard gives up about 0.05 Gini
to a gradient-boosted model on the same inputs.

### Ordered holdout: first half of the file against the second

`config.idorder.yaml` develops on the first 15,000 accounts by ID and monitors on the last 15,000,
nothing injected (`reports/idorder/`). The file does not say that ID is chronological, so I treat it
as an ordered holdout and nothing more. The halves do differ: default rate 22.5% against 21.7%.
Result: 3 Fail, 2 Watch, 22 Pass. Gini is higher in the second half (0.559 against 0.488 in
development), which the drift checks read as no deterioration, but the calibration checks fail:
Hosmer-Lemeshow p = 2e-12 and two grades above their PD under both grade tests, while the portfolio
mean is close (predicted 22.0% against observed 21.7%). The score ranks the later accounts better and
prices some grades wrongly, which is the kind of finding a PSI-only review would not surface.

### Figures

![Traffic light](docs/traffic_light.png)

![PSI and CSI](docs/psi_bars.png)

![ROC and KS](docs/roc_ks.png)

![Calibration](docs/calibration.png)

## Repository layout

```
src/model_audit/
  metrics.py        PSI, AUC/Gini, KS, bootstrap and paired-bootstrap CI, Hosmer-Lemeshow,
                    Wilson and Jeffreys intervals, Jeffreys test, Holm adjustment
  scorecard.py      optimal / builtin WoE binning, tuned logistic, points scale, PD map
  challenger.py     Optuna-tuned LightGBM and the out-of-fold comparison
  simulate.py       split, drift injection, override flag
  checks/           stability, discrimination, calibration, ratings, overrides, quality, benchmark
  summary.py        traffic-light roll-up
  status.py         CheckResult and the Pass/Watch/Fail rule
  pipeline.py, report.py, plots.py, cli.py
tests/              PSI vs hand calculation, KS/Gini vs sklearn/scipy, HL sanity, Jeffreys vs the
                    beta posterior, Holm by hand, binning monotonicity, thresholds, traffic light
```

## Limitations

- The data is one cross-section from 2005 with a single next-month label. There is no real later
  period, so drift can only be injected, and the results show that the checks find known injections.
  They say nothing about how often real portfolios drift this way.
- Credit-card accounts in one Taiwanese bank, one month: no LGD or EAD, no macro variables, and the
  score-to-PD map is a point-in-time one, not through-the-cycle and not an IFRS 9 PD.
- With 10,000 to 15,000 accounts per sample the formal tests reject small deviations. Read the
  p-values next to the effect sizes in the band and grade tables.

## References

- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
- Hosmer, D. W. and Lemeshow, S. (1980). Goodness of fit tests for the multiple logistic regression
  model. *Communications in Statistics*.
- Basel Committee on Banking Supervision (2005). *Studies on the Validation of Internal Rating
  Systems*, Working Paper 14.
- Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a binomial proportion.
  *Statistical Science* 16(2).
- Tasche, D. (2008). Validation of internal rating systems and PD estimates. In *The Analytics of Risk
  Model Validation*, Elsevier.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability.
- Navas-Suarez, G. (2020). Optimal binning: mathematical programming formulation. arXiv:2001.08025
  (the `optbinning` package, Apache-2.0).
- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS*.
- Akiba, T. et al. (2019). Optuna: a next-generation hyperparameter optimization framework. *KDD*.
- Yeh, I-C. and Lien, C-H. (2009). The comparisons of data mining techniques for the predictive
  accuracy of probability of default of credit card clients. *Expert Systems with Applications*.

## License

MIT, see [LICENSE](LICENSE). The UCI dataset is licensed separately under CC BY 4.0.

By Utku Fidan · [utkufidan.com](https://utkufidan.com)
