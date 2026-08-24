#!/usr/bin/env python3
"""
01_prepare.py — load a raw corpus, route it into groups, write a clean CSV.

    python scripts/01_prepare.py --corpus ton_iot \
        --input raw/Train_Test_Network.csv --out data/ton_iot/

    python scripts/01_prepare.py --corpus bot_iot \
        --input raw/botiot_full/ --out data/bot_iot/ --drop-small
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aegis.data import load_bot_iot, load_ton_iot, load_ton_telemetry

a = argparse.ArgumentParser()
a.add_argument("--corpus", required=True,
               choices=["ton_iot", "ton_telemetry", "bot_iot"])
a.add_argument("--input", required=True, help="CSV file, or directory for Bot-IoT shards")
a.add_argument("--out", required=True)
a.add_argument("--drop-small", action="store_true",
               help="Bot-IoT: drop categories under 3000 rows (Theft)")
a.add_argument("--max-rows", type=int, default=0,
               help="cap rows read (testing only; leave 0 for real runs)")
a.add_argument("--min-group-rows", type=int, default=5000,
               help="a service/protocol needs this many flows for its own detectors")
a.add_argument("--min-group-classes", type=int, default=4,
               help="...and this many distinct classes")
a = a.parse_args()

print(f"\n=== preparing {a.corpus} ===")
if a.corpus == "ton_iot":
    df = load_ton_iot(a.input, max_rows=a.max_rows,
                      min_rows=a.min_group_rows, min_classes=a.min_group_classes)
elif a.corpus == "ton_telemetry":
    df = load_ton_telemetry(a.input, min_rows=min(a.min_group_rows, 3000),
                            min_classes=3)
else:
    df = load_bot_iot(a.input, drop_small=a.drop_small,
                      min_rows=a.min_group_rows, min_classes=a.min_group_classes)

os.makedirs(a.out, exist_ok=True)
p = os.path.join(a.out, "prepared.csv")
df.to_csv(p, index=False)
print(f"\nwrote {p}  ({len(df):,} rows)")
print("next:  python scripts/02_build_eval_set.py --data-dir", a.out)
