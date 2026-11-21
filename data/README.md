# Data

PTB-XL v1.0.3 (https://physionet.org/content/ptb-xl/1.0.3/, DOI 10.13026/kfzx-aw45, CC BY 4.0).

`pixi run download` fetches the official zip (~1.8 GB), extracts only `records100/` and the
metadata CSVs into `data/ptbxl/`, then deletes the zip (`--keep-zip` to keep it,
`--with-500` to also extract `records500/`).

Layout after download and preprocessing:

```text
data/
├── ptbxl/
│   ├── ptbxl_database.csv     # metadata, strat_fold, noise annotations
│   ├── scp_statements.csv
│   └── records100/            # WFDB, 100 Hz, 12 leads, 10 s (1000 samples)
└── processed/                 # cached filtered Lead II per split (.npz), one file per filter setting
```

Split: official `strat_fold` (patient-wise). Folds 1-8 train, 9 validation, 10 test.
`src/ptbxl.py` checks that no patient appears in two splits.
