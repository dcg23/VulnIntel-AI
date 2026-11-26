# VulnIntel-AI

VulnIntel-AI is a reproducible retrospective classical machine-learning study
that classifies whether a CVE belongs to the downloaded snapshot of the CISA
Known Exploited Vulnerabilities (KEV) catalog.

The binary target is defined only by normalized CVE-ID membership:

* `kev\_exploited = 1` when the CVE appears in CISA KEV.
* `kev\_exploited = 0` when the CVE does not appear in CISA KEV.

This is an **ever-in-KEV snapshot label**, not a label frozen at each CVE's
publication date or at the train/test cutoff. A CVE may have been added to KEV
long after it was published, and current NVD fields may also include later
revisions. Consequently, results must not be described as predicting future
exploitation. They measure retrospective KEV-membership classification and
generalisation from earlier-published CVEs to later-published CVEs.

## Research questions

VulnIntel-AI addresses two research questions: How accurately can classical
machine-learning and natural-language-processing techniques retrospectively
classify whether a CVE belongs to the downloaded CISA Known Exploited
Vulnerabilities (KEV) snapshot using TF-IDF representations of NVD vulnerability
descriptions together with CVSS and CWE features? Furthermore, which of Logistic
Regression, Random Forest, and XGBoost performs best when generalising from
earlier-published CVEs to a later-published test cohort, primarily measured by
average precision (PR-AUC) for the KEV class? The target is set to `1` when a
normalised CVE identifier appears in the downloaded CISA KEV snapshot and to `0`
otherwise. EPSS scores, KEV metadata, and target-derived fields are excluded from
the predictors to reduce feature leakage. Because the labels represent
membership in the current downloaded KEV snapshot rather than KEV status frozen
at the historical train/test cutoff, the study is defined as retrospective KEV
classification and does not claim to predict future exploitation.

## Research design

* Data: NVD CVE API 2.0 and the official CISA KEV CSV.
* Text: lowercase TF-IDF with English stop words, unigrams, bigrams, and a
10,000-feature limit.
* Structured inputs: three CVSS numeric fields plus CVSS/CWE categorical
fields listed in the source.
* Baseline: majority-class `DummyClassifier`.
* Main models: Logistic Regression, Random Forest, and XGBoost—exactly these
three.
* Split: chronological, approximately 80/20 by NVD CVE publication date. All
test CVEs are published after all training CVEs; equal timestamps remain on
the same side of the boundary.
* Tuning: randomized search with expanding-window temporal folds, using average
precision (PR-AUC) on training data only. Folds without both classes are not
used because their PR-AUC would not be meaningful.
* Imbalance: balanced class weights for Logistic Regression and Random Forest;
XGBoost `scale\_pos\_weight` is calculated from training labels only.
* Selection and reporting: PR-AUC, positive-class precision, recall, F1,
TN/FP/FN/TP, and confusion matrices. Accuracy is additional only.

TF-IDF is the only text representation. The evaluated classifiers are Logistic
Regression, Random Forest, and XGBoost, with a majority-class dummy baseline.

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

An NVD API key is optional but strongly recommended for a full download:

```powershell
$env:NVD\_API\_KEY = "YOUR\_NVD\_API\_KEY"
```

On Linux or macOS, activate with `source .venv/bin/activate` and set the key
with `export NVD\_API\_KEY="YOUR\_NVD\_API\_KEY"`.

## Run

The normal research run supports the full NVD history from 1999 through the
current year. It reuses range-specific CSV caches in `data/raw/` unless
`--refresh` is supplied.

```powershell
python src/vulnintel\_classical\_ml.py
```

The full download and temporal hyperparameter searches are intentionally substantial. A
clearly labelled development run caps the NVD records and may skip tuning:

```powershell
python src/vulnintel\_classical\_ml.py --development --skip-tuning
```

Development outputs are code-check artifacts, not final experimental results.
If either chronological partition, or too many temporal tuning folds, contain
too few KEV records, increase `--development-max-records` or broaden the range.

Useful options:

```text
--first-year YEAR              first NVD publication year
--last-year YEAR               last NVD publication year
--refresh                      redownload NVD and CISA data
--development                  use the capped development cache/mode
--development-max-records N    cap development NVD records (default 10000)
--skip-tuning                  fit defaults for a quick engineering check
--tuning-max-rows N            training-only rows used for parameter search
--search-jobs N                parallel search workers (safe default: 1)
```

For a final research run, do not use `--development` or `--skip-tuning`.

The full historical dataset is large. Hyperparameter search therefore uses a
fixed-seed stratified subset of at most 75,000 training rows by default, then
refits each selected pipeline on the complete training set. The test set remains
untouched. Search uses one worker by default to avoid exhausting memory and
closing an IDE. Computers with more memory can opt into limited parallelism, for
example `--search-jobs 2`. Increase it cautiously.

## Outputs

The pipeline creates:

```text
data/raw/                         downloaded range-specific NVD and CISA caches
data/processed/
  vulnintel\_model\_dataset.csv    cleaned modelling dataset
outputs/results/
  dataset\_dimensions.csv
  data\_types.csv
  duplicate\_cve\_ids.csv
  missing\_values.csv
  cvss\_numeric\_summary.csv
  publication\_date\_coverage.csv
  target\_temporal\_audit.csv
  temporal\_split\_summary.csv
  cleaning\_report.csv
  class\_balance.csv
  class\_imbalance\_assessment.txt
  hyperparameter\_tuning\_results.csv
  hyperparameter\_search\_candidates.csv
  model\_comparison.csv
  model\_selection\_summary.txt
  \*\_false\_negatives.csv
outputs/figures/
  class\_balance.png
  \*\_confusion\_matrix.png
  \*\_pr\_curve.png
  combined\_ml\_precision\_recall\_curves.png
  model\_comparison\_dashboard.png
outputs/feature\_importance/
  random\_forest\_feature\_importance.csv
  random\_forest\_feature\_importance.png
  xgboost\_feature\_importance.csv
  xgboost\_feature\_importance.png
```

False-negative CSVs retain the CVE ID, dates, description, selected CVSS/CWE
details, and predicted probability. A false negative means a genuinely
exploited KEV vulnerability was missed, which can cause dangerous
under-prioritization during remediation.

A false positive is a non-KEV CVE predicted as KEV. False positives can create
unnecessary patching urgency, analyst work, and inefficient remediation effort;
this operational cost is distinct from the security risk of missing an actually
exploited vulnerability. `model\_selection\_summary.txt` identifies the strongest
measured PR-AUC model from that run and separately reports the model with the
fewest false negatives, so XGBoost or any other model is never assumed to win.

Tree-model importance is descriptive model evidence, not proof that a feature
causes exploitation.

## Cleaning decisions

* CVE IDs are normalized, missing IDs are removed, and duplicates are resolved
by keeping the last record.
* Empty descriptions receive an explicit placeholder so they remain observable.
* Numeric CVSS fields are converted with invalid values set to missing; values
outside the defined 0–10 score range are not treated as statistical outliers.
* Numeric missing values are median-imputed inside each fitted pipeline.
* Missing/blank categorical values become `UNKNOWN`; pipeline imputation and
one-hot encoding safely handle unseen categories.
* Rare KEV observations are never removed merely for looking like outliers.

The repository excludes downloaded datasets but includes selected development-run results for transparency. Run the full pipeline without development or skip-tuning options to conduct the complete research experiment.



\## Development Experiment: Results and Limitations



The included experimental outputs are from a development run using 10,000 NVD records.



Dataset split:

\- Training records: 8,000

\- Testing records: 2,000

\- KEV-positive records: 51

\- Positive test cases: 9



\### Model comparison



| Model | PR-AUC | Precision | Recall |

|---|---:|---:|---:|

| Logistic Regression | 0.01116 | 0.000 | 0.000 |

| XGBoost | 0.00562 | 0.000 | 0.000 |

| Random Forest | 0.00542 | 0.000 | 0.000 |

| Dummy baseline | 0.00450 | 0.000 | 0.000 |



All three main classifiers missed all nine positive KEV cases at the default classification threshold.



Logistic Regression achieved the highest PR-AUC, but none of the models successfully identified a positive case using the reported threshold.



These results are development findings, not final research results or evidence of future exploitation prediction.



\### Model comparison dashboard



!\[Model Comparison Dashboard](outputs/figures/model\_comparison\_dashboard.png)



\### Limitations and future work



\- Investigate decision-threshold selection.

\- Explore class imbalance handling.

\- Evaluate recall and false-negative reduction.

\- Conduct the full historical NVD experiment.

\- Assess generalisation using a more representative positive test cohort.

