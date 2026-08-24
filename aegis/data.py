from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

# TON_IoT network
TON_DROP = {
    "src_ip", "dst_ip", "src_port", "dst_port", "ts", "label",
    "src_mac", "dst_mac", "dns_query", "ssl_subject", "ssl_issuer",
    "http_uri", "http_referrer", "http_user_agent", "weird_name",
    "weird_addl", "http_orig_mime_types", "http_resp_mime_types",
}


TON_EXPECTED_ROWS = 211043      # Train_Test_datasets/Train_Test_Network.csv


def _viable_groups(df, key, min_rows: int, min_classes: int) -> set:
    """A group earns its own detectors only if it has enough rows AND enough
    classes. Everything else merges into the residual group.

    Both tests matter. A group with many rows but one class scores a trivial
    macro-F1 of 1.0. A group with several classes but few rows receives almost
    no validation incidents, so tier comparison degenerates into ties.
    """
    keep, merged = set(), []
    counts = key.value_counts()
    for g, n in counts.items():
        n_cls = df.loc[key == g, "true_type"].nunique()
        if n >= min_rows and n_cls >= min_classes:
            keep.add(g)
        else:
            merged.append(f"{g}(n={n},classes={n_cls})")
    if merged:
        print(f"  merged (too small or too few classes): {', '.join(merged)}")
    if not keep:
        raise ValueError("no group met the viability threshold; lower "
                         "--min-group-rows or --min-group-classes")
    return keep


def _absorb_small_residual(df, min_rows: int):
    """The residual group collects everything that failed the viability test,
    so it can itself end up too small -- on Bot-IoT it held 2,912 flows from
    arp/ipv6-icmp/igmp/rarp. Incidents in a group with no fitted detectors
    receive no evidence and default to benign, so they must be re-homed rather
    than left stranded. Fold a small residual into the largest group."""
    counts = df.device.value_counts()
    resid = [g for g in counts.index if g.endswith("_other")]
    if not resid:
        return df
    r = resid[0]
    if counts.get(r, 0) >= min_rows:
        return df
    biggest = counts.drop(labels=[r], errors="ignore").idxmax()
    print(f"  residual group {r} has {counts[r]:,} flows (< {min_rows:,}); "
          f"folding into {biggest}")
    df.loc[df.device == r, ["device", "domain"]] = biggest
    return df


def load_ton_iot(path: str, max_rows: int = 0, min_rows: int = 5000,
                 min_classes: int = 4) -> pd.DataFrame:
    """`path` may be Train_Test_Network.csv (one file, ~461k rows) or a
    directory of Processed_Network_dataset shards.

    There is deliberately NO row cap unless you ask for one. Benign traffic is
    the binding constraint on the whole evaluation set, so silently reading a
    subset shrinks everything downstream.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"no CSV files under {path}")
        print(f"  reading {len(files)} network shards from {path} ...")
        frames = []
        for i, f in enumerate(files, 1):
            frames.append(pd.read_csv(f, low_memory=False))
            if i % 5 == 0:
                print(f"    {i}/{len(files)}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, nrows=max_rows or None, low_memory=False)

    df.columns = [c.strip() for c in df.columns]
    print(f"  read {len(df):,} rows from {os.path.basename(path) or path}")
    if not os.path.isdir(path) and not max_rows and len(df) < 180_000:
        print(f"\n  WARNING: expected ~{TON_EXPECTED_ROWS:,} rows in")
        print( "  Train_Test_datasets/Train_Test_Network.csv but read "
              f"{len(df):,}.")
        print( "  You may have a reduced copy. Benign flows are the binding")
        print( "  constraint on the evaluation pool, so a subset shrinks the")
        print( "  test split and widens every confidence interval. Check the")
        print( "  file before running the headline experiment.\n")

    if "type" not in df.columns:
        raise ValueError(
            "expected a 'type' column (TON_IoT label). Got: "
            f"{list(df.columns)[:12]}")
    df = df.rename(columns={"type": "true_type"})
    df["true_type"] = df["true_type"].astype(str).str.strip().str.lower()

    # Route by application-layer service. Groups under MIN_GROUP are merged,
    # because a detector fitted on a few dozen flows models sampling noise.
    svc = df["service"].astype(str).str.strip().str.lower() if "service" in df else "-"
    svc = svc.replace({"-": "other", "nan": "other", "": "other"})
    keep = _viable_groups(df, svc, min_rows, min_classes)
    df["device"] = ["net_" + (s if s in keep else "other") for s in svc]
    df["domain"] = df["device"]

    drop = [c for c in df.columns if c.lower() in TON_DROP]
    df = df.drop(columns=drop, errors="ignore")
    df = _absorb_small_residual(df, min_rows)
    return _finalise(df, "ton_iot")


# TON_IoT telemetry — the WEAK-EVIDENCE regime
# Each device file exposes only two to four sensor readings. That is the whole
# point of including it: it is the same collection, the same attack families
# and the same pipeline, with the tool evidence deliberately impoverished. It
# is the control that makes any claim about evidence strength testable.
TELEM_DROP = {"date", "time", "ts", "label", "type"}


def load_ton_telemetry(path: str, min_rows: int = 3000,
                       min_classes: int = 3) -> pd.DataFrame:
    """`path` is the directory holding Train_Test_IoT_*.csv (one per device)."""
    if not os.path.isdir(path):
        raise NotADirectoryError(
            path + " is not a directory. Point this at the folder holding "
            "the per-device files: Train_Test_IoT_Fridge.csv, "
            "Train_Test_IoT_Garage_Door.csv, Train_Test_IoT_GPS_Tracker.csv, "
            "Train_Test_IoT_Modbus.csv, Train_Test_IoT_Motion_Light.csv, "
            "Train_Test_IoT_Thermostat.csv, Train_Test_IoT_Weather.csv")
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSV files under {path}")

    frames = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        if "type" not in d.columns:
            print(f"  skipping {os.path.basename(f)} (no 'type' column)")
            continue
        dev = (os.path.basename(f).replace("Train_Test_IoT_", "")
               .replace(".csv", "").lower())
        d["device"] = "iot_" + dev
        n_feat = len([c for c in d.columns
                      if c.lower() not in TELEM_DROP and c != "device"])
        print(f"  {dev:<16} {len(d):>7,} rows, {n_feat} sensor features")
        frames.append(d)
    if not frames:
        raise ValueError("no usable telemetry files found")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"type": "true_type"})
    df["true_type"] = df["true_type"].astype(str).str.strip().str.lower()
    df["domain"] = df["device"]

    keep = _viable_groups(df, df.device, min_rows, min_classes)
    df.loc[~df.device.isin(keep), ["device", "domain"]] = "iot_other"

    drop = [c for c in df.columns if c.lower() in TELEM_DROP]
    df = df.drop(columns=drop, errors="ignore")
    df = _absorb_small_residual(df, min_rows)
    out = _finalise(df, "ton_telemetry")
    n_feat = len([c for c in out.columns
                  if c not in ("true_type", "device", "domain")])
    print(f"\n  NOTE: {n_feat} features across all devices. Telemetry is the")
    print( "  weak-evidence regime by construction -- expect a low ceiling and")
    print( "  read it as the CONTRAST to the network corpus, not as a failure.")
    return out


# Bot-IoT
BOT_DROP = {
    "pkSeqID", "stime", "ltime", "saddr", "daddr", "sport", "dport",
    "smac", "dmac", "soui", "doui", "seq", "attack", "subcategory",
}
BOT_MIN_NORMAL = 2000     # below this the 35% target is unreachable


def load_bot_iot(path: str, drop_small: bool = False,
                 min_rows: int = 5000, min_classes: int = 3) -> pd.DataFrame:
    """`path` may be a single CSV or a directory of the numbered CSV shards."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"no CSV files under {path}")
        print(f"  reading {len(files)} Bot-IoT shards from {path} ...")
        frames = []
        for i, f in enumerate(files, 1):
            frames.append(pd.read_csv(f, low_memory=False))
            if i % 10 == 0:
                print(f"    {i}/{len(files)}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)

    df.columns = [c.strip() for c in df.columns]
    if "category" not in df.columns:
        raise ValueError(
            "expected a 'category' column (Bot-IoT label). If you downloaded "
            "the '10-best-features' file it does not carry full labels; use "
            "the full-feature CSVs instead.")

    df["true_type"] = (df["category"].astype(str).str.strip().str.lower()
                       .replace({"normal": "normal"}))

    n_normal = int((df.true_type == "normal").sum())
    print(f"  Bot-IoT: {len(df):,} flows, {n_normal:,} benign "
          f"({100 * n_normal / len(df):.4f}%)")

    if n_normal < BOT_MIN_NORMAL:
        sys.exit(
            f"\nSTOP: only {n_normal} benign flows found.\n\n"
            "This is the 5% Bot-IoT subset, which contains 477 normals. The\n"
            "evaluation protocol rebalances to a benign fraction of 0.35, so\n"
            f"this file caps the entire pool at {n_normal}/0.35 = "
            f"{int(n_normal / 0.35)} flows -- far too few for multiclass\n"
            "attribution across five categories.\n\n"
            "Download the FULL Bot-IoT release (73M records, ~16 GB, 74 CSV\n"
            "shards) which contains 9,543 benign flows, and point this script\n"
            "at the directory containing them.\n")

    # Theft is tiny (1,587 in the full release) and binds the per-family cap.
    if drop_small:
        small = [c for c in df.true_type.unique()
                 if (df.true_type == c).sum() < 3000 and c != "normal"]
        if small:
            print(f"  dropping small categories {small} (--drop-small)")
            df = df[~df.true_type.isin(small)]

    # Route by transport protocol -- the closest Bot-IoT analogue to the
    # application-layer service grouping used for TON_IoT.
    proto = pd.Series(df["proto"].astype(str).str.strip().str.lower()
                      if "proto" in df else "other", index=df.index)
    keep = _viable_groups(df, proto, min_rows, min_classes)
    df["device"] = ["bot_" + (p if p in keep else "other") for p in proto]
    df["domain"] = df["device"]

    drop = [c for c in df.columns if c in BOT_DROP]
    df = df.drop(columns=drop, errors="ignore")
    df = _absorb_small_residual(df, min_rows)
    return _finalise(df, "bot_iot")


def _finalise(df: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """Coerce features to numeric, drop constants, report what survived."""
    meta = ["true_type", "device", "domain"]
    feats = [c for c in df.columns if c not in meta]

    for c in feats:
        # Test for NON-NUMERIC rather than == object. Under pandas 2.x string
        # columns carry dtype 'str', so an object check misses them entirely,
        # and they silently coerce to NaN -> 0 -> constant -> dropped.
        if not pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique(dropna=True) <= 32:
                df[c] = pd.factorize(df[c])[0]      # low-cardinality -> codes
            else:
                df = df.drop(columns=[c])           # free text, no signal
                continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    feats = [c for c in df.columns if c not in meta]
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    const = [c for c in feats if df[c].nunique() <= 1]
    if const:
        print(f"  dropped {len(const)} constant features: {const}")
        df = df.drop(columns=const)
        feats = [c for c in feats if c not in const]

    df = df[df.true_type.notna() & (df.true_type != "nan")]
    n_norm = int((df.true_type == "normal").sum())
    print(f"  {corpus}: {len(df):,} flows, {len(feats)} features, "
          f"{df.true_type.nunique()} classes, {df.device.nunique()} groups")
    print(f"  groups: {dict(df.device.value_counts())}")
    print(f"  class counts: {dict(df.true_type.value_counts())}")
    # Benign is almost always what caps the rebalanced pool -- say so here,
    # before the user discovers it three scripts later.
    n_fam = max(df.true_type.nunique() - 1, 1)
    cap = int(n_norm * 0.65 / (0.35 * n_fam))
    print(f"\n  benign flows: {n_norm:,}  ->  at 35% benign this supports")
    print(f"  ~{cap:,} per attack family and a pool of ~{n_norm + cap * n_fam:,} flows")
    return df.reset_index(drop=True)


def live_features(frame: pd.DataFrame, feats) -> list:
    """Features that actually vary within this group. A constant column
    carries no information and inflates the apparent feature count."""
    return [c for c in feats if frame[c].nunique() > 1]
