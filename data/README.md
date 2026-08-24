# Datasets

Two corpora. Both are public and free, but both require a short registration
form on the UNSW Canberra site.

Place the downloaded files exactly as shown below. Nothing else goes in `raw/`.

    aegis-iot/
    └── raw/
        ├── train_test_network.csv        TON_IoT, one file, about 29 MB
        └── botiot_full/                  Bot-IoT, 74 CSV shards, about 16 GB
            ├── UNSW_2018_IoT_Botnet_Dataset_1.csv
            ├── ...
            └── UNSW_2018_IoT_Botnet_Dataset_74.csv

Create the directory first:

    mkdir -p raw/botiot_full

## TON_IoT network, required

Source: https://research.unsw.edu.au/projects/toniot-datasets

Mirror: https://ieee-dataport.org/documents/toniot-datasets

Navigate to `Train_Test_datasets/` and download **`Train_Test_Network.csv`**.
Rename it to lower case and place it at `raw/train_test_network.csv`.

Verify before running anything:

    wc -l raw/train_test_network.csv        # expect 211044 including the header

Contents you should see:

| Class | Records |
|---|---|
| normal | 50,000 |
| backdoor, ddos, dos, injection, password, ransomware, scanning, xss | 20,000 each |
| mitm | 1,043 |
| **total** | **211,043** |

Do not use `Processed_datasets/Processed_Network_dataset/`. That is the full
capture, tens of millions of rows, and is not what the paper reports.

### A note on mitm

The man-in-the-middle class has 1,043 records against 20,000 for every other
family. `02_build_eval_set.py` is run with `--allow-unequal`, which lets a small
family keep its natural size rather than dragging every other family down to it.
Forcing uniformity would discard roughly nine tenths of the usable data to
accommodate one class.

This matters for the results: mitm is the class that determines how far the
accuracy floor can be relaxed. Section 5.2 of the paper reports that it loses
about a tenth of its F1 at a tolerance where common families lose under one
percent, which is the practical limit on relaxing the floor.

## Bot-IoT, required for the replication

Source: https://research.unsw.edu.au/projects/bot-iot-dataset

Download the **full-feature CSVs**, `UNSW_2018_IoT_Botnet_Dataset_1.csv`
through `_74.csv`, and place all 74 in `raw/botiot_full/`.

    ls raw/botiot_full/*.csv | wc -l        # expect 74

**Do not use the 5% subset.** It contains 477 benign flows. Because the protocol
rebalances to a benign fraction of 0.35 and benign traffic is the limiting
resource, that subset caps the entire evaluation pool at roughly 1,363 flows,
which cannot support multiclass attribution. `01_prepare.py` detects this and
exits with an explanation rather than silently producing an unusable evaluation
set.

**Do not use the "10-best-features" file.** It does not carry the full label
columns.

Full-release class counts:

| Class | Records |
|---|---|
| ddos | 38,532,480 |
| dos | 33,005,194 |
| reconnaissance | 1,821,639 |
| normal | 9,543 |
| theft | 1,587 |

Theft is excluded by `run_all.sh` using `--drop-small`, because at 1,587 records
it would cap the per-family sample size for the entire study. Section 3.1 of the
paper states this as a judgement rather than a technicality, and a reader may
reasonably prefer the alternative.

## Disk and time

| Step | Time | Disk |
|---|---|---|
| TON_IoT download | 1 min | 29 MB |
| Bot-IoT download | 30 to 90 min | 16 GB |
| Bot-IoT prepare, reads all 74 shards | 15 to 20 min | 4 GB intermediate |
| TON_IoT full pipeline | about 1.5 h | small |
| Bot-IoT full pipeline | about 2 h | small |

`raw/` and `data/` are both git-ignored. The corpora are redistributed by UNSW
under their own terms and should not be committed to this repository.
