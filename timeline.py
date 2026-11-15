"""
Red-flag timeline: the accounting score as it would have read at the end of each past fiscal year.

For every year Y the statements are cut to the columns dated on or before Y's year-end (TTM and later
half-year columns dropped) and the same checks are re-run. Only the statement-based layers are
replayed. Ownership, filings, peers and the Screener pros/cons extras are current-day data, so they
are left out. Market value is not known historically, so Altman uses the Z″ form throughout.
Restated figures: Screener shows the latest reported numbers for each year, not the numbers as first
published, so a year that was later restated is scored on the restated figures.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

import forensics as fx

MIN_YEARS = 3   # need at least this many years of history before a year is scored


def _fy_of(label: str):
    """'Mar 2024' -> 2024 ; 'Dec 2023' -> 2024 (Indian FY ends March) ; 'TTM' -> None."""
    try:
        ts = pd.to_datetime(str(label), format="%b %Y")
    except Exception:
        return None
    return ts.year + (1 if ts.month > 3 else 0)


def _cut_tables(tables: dict, fy: int, annual_only: bool = True) -> dict:
    out = {}
    for k, df in (tables or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            out[k] = df
            continue
        keep = []
        for c in df.columns:
            y = _fy_of(c)
            if y is None:
                continue
            if k == "quarters" or k.endswith("_quarters"):
                if y <= fy:
                    keep.append(c)
            elif y <= fy and (pd.to_datetime(str(c), format="%b %Y").month == 3 or not annual_only):
                keep.append(c)
        out[k] = df[keep]
    return out


def _cut_sched(sched: dict, fy: int) -> dict:
    out = {}
    for parent, items in (sched or {}).items():
        out[parent] = {it: {p: v for p, v in vals.items() if (_fy_of(p) or 9999) <= fy}
                       for it, vals in items.items()}
    return out


def _no_mcap_profile(profile):
    p = copy.deepcopy(profile)
    p.th["altman"]["model"] = "z2"
    return p


def flags_at(fy: int, fin: pd.DataFrame, profile, info: dict, kind: str,
             tables: dict | None = None, sched: dict | None = None, manual: dict | None = None):
    cols = [c for c in fin.columns if str(c).startswith("FY") and int(str(c)[2:]) <= fy]
    if len(cols) < MIN_YEARS:
        return None
    f_cut = fin[cols]
    flags = []
    if profile.financial_layer:
        fl, _ = fx.financial_flags(f_cut, info, _no_mcap_profile(profile), np.nan, extra={})
        if profile.keep_only:
            fl = [f for f in fl if f.key in profile.keep_only]
        flags += fl
    if tables is not None and (profile.lender_layer or profile.broker_layer or profile.insurer_layer):
        t_cut, s_cut = _cut_tables(tables, fy), _cut_sched(sched, fy)
        try:
            if profile.lender_layer:
                flags += fx.lender_flags(f_cut, t_cut, s_cut, kind)[0]
            elif profile.broker_layer:
                flags += fx.broker_flags(t_cut, s_cut)[0]
            else:
                flags += fx.insurer_flags(t_cut, s_cut, profile.insurer_layer, manual or {})[0]
        except Exception:
            pass
    return flags


def build(fin: pd.DataFrame, profile, info: dict, kind: str, tables: dict | None = None,
          sched: dict | None = None, manual: dict | None = None) -> pd.DataFrame:
    """One row per fiscal year: score, band, red/watch counts, coverage and which checks were red or watch."""
    if fin is None or fin.empty:
        return pd.DataFrame()
    years = sorted({int(str(c)[2:]) for c in fin.columns if str(c).startswith("FY")})
    rows = []
    for fy in years:
        flags = flags_at(fy, fin, profile, info, kind, tables, sched, manual)
        if not flags:
            continue
        c = fx.composite(flags)
        rows.append({
            "year": f"FY{fy}", "score": c["score"], "band": c["band"], "red": c["red"], "watch": c["amber"],
            "coverage": c["coverage"],
            "red_checks": ", ".join(f.title for f in flags if f.severity == fx.RED),
            "watch_checks": ", ".join(f.title for f in flags if f.severity == fx.AMBER),
            "_red_keys": [f.key for f in flags if f.severity == fx.RED],
        })
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()
