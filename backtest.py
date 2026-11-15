"""
Backtest: point-in-time scores for companies that later failed vs large clean controls.

    python backtest.py [cache_dir]      # fetches from Screener once, caches, writes backtest_results.json

t0 is the last fiscal year whose accounts were public before the event. Events are regulatory orders,
defaults, insolvency, auditor resignations or disclosed fraud — not short-seller allegations.

Caveats (also shown in the app):
* Screener shows the latest restated figures, so a company that later restated its accounts is scored on
  numbers investors did not see at the time. That can flatter detection (CG Power, Coffee Day restated).
* Small sample: 13 failures, 20 controls. Survivorship: delisted companies missing from Screener drop out.
* Only the statement-based layers are replayed; pledges, filings and peers are excluded.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

import data_sources as ds
import sector as sc
import timeline as tl

warnings.filterwarnings("ignore")

FAIL = {"VAKRANGEE": (2017, "Price collapse and auditor resignation, 2018"),
        "PCJEWELLER": (2018, "Price collapse 2018, loan defaults 2019–20"),
        "COFFEEDAY": (2018, "Founder's death and fund-diversion disclosure, Jul 2019"),
        "BCG": (2023, "SEBI interim order on fabricated revenue, Aug 2023"),
        "GENSOL": (2024, "SEBI order on fund diversion, Apr 2025"),
        "MANPASAND": (2018, "Auditor resignation and GST-fraud arrests, 2019"),
        "SINTEX": (2018, "Loan defaults 2019, insolvency 2021"),
        "JETAIRWAYS": (2018, "Grounded Apr 2019, insolvency"),
        "RCOM": (2017, "Debt default 2017–18, insolvency 2019"),
        "GITANJALI": (2017, "PNB fraud disclosure, Feb 2018"),
        "ZEEL": (2018, "Promoter pledge crisis Jan 2019; SEBI diversion order 2023"),
        "CGPOWER": (2018, "Irregularities by prior management disclosed, Aug 2019"),
        "KWALITY": (2018, "Loan default Oct 2018, bank-fraud probe, insolvency")}
CTRL = ["HINDUNILVR", "ASIANPAINT", "NESTLEIND", "PIDILITIND", "TCS", "INFY", "MARUTI", "BAJAJ-AUTO", "ITC",
        "DMART", "TITAN", "LT", "HCLTECH", "DABUR", "BRITANNIA", "ULTRACEMCO", "HAVELLS", "EICHERMOT", "COLPAL",
        "MARICO"]
CTRL_YEARS = range(2017, 2026)
NEW_CHECKS = {"dpo", "interest_rate", "exceptional", "dilution", "contingent"}


def fetch(cache):
    os.makedirs(cache, exist_ok=True)
    c = ds.ScreenerClient()
    for sym in list(FAIL) + CTRL:
        f = os.path.join(cache, f"{sym}.pkl")
        if os.path.exists(f):
            continue
        res, err = c.load(sym)
        if res is None:
            print(f"{sym}: {err}")
            continue
        res.pop("peers", None)
        pickle.dump(res, open(f, "wb"))
        time.sleep(1)


def score_rows(res, drop=()):
    kind, _ = sc.classify(res["meta"].get("classification"), {}, res["meta"].get("is_financial", False))
    prof = sc.PROFILES[kind]
    rows = {}
    for fy in sorted({int(str(c)[2:]) for c in res["fin"].columns if str(c).startswith("FY")}):
        flags = tl.flags_at(fy, res["fin"], prof, {}, kind, res["tables"], res["sched"])
        if not flags:
            continue
        for f in flags:
            if f.key in drop:
                f.severity = None
        from forensics import composite, RED
        rows[fy] = (composite(flags)["score"], [f.key for f in flags if f.severity == RED])
    return kind, rows


def run(cache):
    data = {s: pickle.load(open(os.path.join(cache, f"{s}.pkl"), "rb"))
            for s in list(FAIL) + CTRL if os.path.exists(os.path.join(cache, f"{s}.pkl"))}
    for r in data.values():   # rebuild the frame so cached pages pick up parser changes
        r["fin"] = ds.ScreenerClient._to_frame(r["tables"], r["sched"])
    out = {"generated": pd.Timestamp.now().strftime("%d %b %Y"), "failures": [], "controls": [], "summary": {}}
    for variant, drop in (("current", ()), ("without_new_checks", NEW_CHECKS)):
        f_t0, f_t1, c_all, red_f, red_c = [], [], [], Counter(), Counter()
        for sym, (fy, ev) in FAIL.items():
            if sym not in data:
                continue
            kind, rows = score_rows(data[sym], drop)
            s0, s1, s2 = (rows.get(y, (np.nan, []))[0] for y in (fy, fy - 1, fy - 2))
            if pd.notna(s0):
                f_t0.append(s0)
                red_f.update(rows[fy][1])
            if pd.notna(s1):
                f_t1.append(s1)
            if variant == "current":
                out["failures"].append({"symbol": sym, "type": kind, "event": ev, "t0": f"FY{fy}",
                                        "score_t0": s0, "score_t1": s1, "score_t2": s2,
                                        "red_t0": rows.get(fy, (0, []))[1]})
        for sym in CTRL:
            if sym not in data:
                continue
            kind, rows = score_rows(data[sym], drop)
            s = [rows[y][0] for y in CTRL_YEARS if y in rows and pd.notna(rows[y][0])]
            for y in CTRL_YEARS:
                if y in rows:
                    red_c.update(rows[y][1])
            c_all += s
            if variant == "current":
                out["controls"].append({"symbol": sym, "type": kind, "median": float(np.median(s)),
                                        "max": float(np.max(s)), "years": len(s)})
        f0, f1, ca = np.array(f_t0), np.array(f_t1), np.array(c_all)
        summ = {"n_fail": len(f0), "n_fail_t1": len(f1), "n_control_years": len(ca),
                "median_fail_t0": float(np.median(f0)), "median_fail_t1": float(np.median(f1)),
                "median_control": float(np.median(ca))}
        for th in (20, 35):
            summ[f"hit_t0_{th}"] = int((f0 >= th).sum())
            summ[f"hit_t1_{th}"] = int((f1 >= th).sum())
            summ[f"fp_{th}"] = int((ca >= th).sum())
        if variant == "current":
            summ["red_rate"] = {k: {"fail": red_f[k] / len(f0), "control": red_c[k] / len(ca)}
                                for k in sorted(set(red_f) | set(red_c), key=lambda k: -red_f[k])}
        out["summary"][variant] = summ
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
    return out


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else "bt_cache"
    fetch(cache)
    o = run(cache)
    print(json.dumps(o["summary"], indent=1, default=str))
