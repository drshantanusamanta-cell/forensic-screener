"""
Sector peer comparison.

Every company (target and peers) is measured from the same Screener main-page
tables, so the comparison is like-for-like. A flag is then tempered when the
company is no worse than its typical peer (a sector norm, not a company quirk),
and sharpened when a "clear" company sits near the bottom of its peer group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_sources import _period_to_fy
import forensics as fx

MIN_PEERS = 4
# Only core manipulation / quality signals may be sharpened by a poor peer position;
# immaterial items (small CWIP, ordinary tax rates) are only ever tempered.
UPGRADE_OK = {"cfo_ebitda", "cfo_ni", "dso", "dio", "sloan", "other_income",
              "gnpa", "pcr", "loan_growth", "roa", "leverage",
              "b_leverage", "b_growth", "i_backing", "i_cfo", "i_other_income"}

# key, label, which direction is worse, flag it informs, display format
SPECS_INDUSTRIAL = [
    ("cfo_ebitda", "CFO ÷ EBITDA (5-yr)", "low", "cfo_ebitda", "pct"),
    ("cfo_ni", "CFO ÷ net profit (5-yr)", "low", "cfo_ni", "pct"),
    ("dso_change", "Change in debtor days (5-yr)", "high", "dso", "days"),
    ("dio_change", "Change in inventory days (5-yr)", "high", "dio", "days"),
    ("sloan", "Sloan accrual ratio (latest)", "high", "sloan", "pct"),
    ("other_income", "Other income ÷ PBT (3-yr)", "high", "other_income", "pct"),
    ("tax", "Effective tax rate (3-yr)", "low", "tax", "pct"),
    ("de", "Debt ÷ equity", "high", "debt", "x"),
    ("cwip", "CWIP ÷ fixed assets", "high", "cwip", "pct"),
    ("dso_level", "Debtor days (latest)", "high", None, "d"),
    ("dio_level", "Inventory days (latest)", "high", None, "d"),
]
SPECS_LENDER = [
    ("gnpa", "Gross NPA % (latest)", "high", "gnpa", "num"),
    ("pcr", "Provision coverage (approx.)", "low", "pcr", "pct"),
    ("growth", "Balance-sheet growth (3-yr)", "high", "loan_growth", "pct"),
    ("roa", "Return on assets", "low", "roa", "pct2"),
    ("lev", "Borrowings ÷ net worth", "high", "leverage", "x"),
]
SPECS_BROKER = [
    ("b_lev", "Borrowings ÷ net worth", "high", "b_leverage", "x"),
    ("b_gap", "Borrowing growth minus net-worth growth (3-yr)", "high", "b_growth", "pct"),
    ("b_liab_nw", "Client liabilities ÷ net worth", "high", None, "x"),
]
SPECS_INSURER = [
    ("i_backing", "Investments ÷ policyholder liabilities", "low", "i_backing", "x"),
    ("i_cushion", "Net worth ÷ policyholder liabilities", "low", "i_cushion", "pct"),
    ("i_cfo", "Operating cash flow ÷ premiums (3-yr)", "low", "i_cfo", "pct"),
    ("i_oi", "Other income ÷ PBT (3-yr)", "high", "i_other_income", "pct"),
    ("i_margin", "Operating margin (latest)", "low", "i_margin", "pct"),
]
SPECS = {"industrial": SPECS_INDUSTRIAL, "lender": SPECS_LENDER, "broker": SPECS_BROKER, "insurer": SPECS_INSURER}


def _annual(df: pd.DataFrame, ye_month: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = {}
    for p in df.columns:
        fy, m = _period_to_fy(p)
        if fy and m == ye_month:
            cols[p] = fy
    return df[list(cols)].rename(columns=cols) if cols else pd.DataFrame()


def _r(df, *labels):
    for lab in labels:
        if df is not None and not df.empty and lab in df.index:
            return pd.to_numeric(df.loc[lab], errors="coerce")
    return pd.Series(dtype=float)


def _mode(m) -> str:
    return m if isinstance(m, str) else ("lender" if m else "industrial")


def metrics_from_tables(tables: dict, mode) -> dict:
    """Peer metrics from Screener main-page tables only (no schedule calls)."""
    mode = _mode(mode)
    lender = mode == "lender"
    if not tables:
        return {}
    pl_raw = tables.get("profit-loss", pd.DataFrame())
    months = [_period_to_fy(p)[1] for p in pl_raw.columns if _period_to_fy(p)[0]]
    if not months:
        return {}
    ye = int(pd.Series(months).mode().iloc[0])
    pl, bs = _annual(pl_raw, ye), _annual(tables.get("balance-sheet"), ye)
    cf, ra = _annual(tables.get("cash-flow"), ye), _annual(tables.get("ratios"), ye)
    out = {}

    ni, pbt = _r(pl, "Net Profit"), _r(pl, "Profit before tax")
    eq = _r(bs, "Equity Capital").add(_r(bs, "Reserves"), fill_value=0)
    ta = _r(bs, "Total Assets")
    debt = _r(bs, "Borrowings", "Borrowing")

    if mode == "broker":
        cols = list(bs.columns)
        if cols and pd.notna(eq.get(cols[-1])) and eq[cols[-1]] > 0:
            c = cols[-1]
            if pd.notna(debt.get(c)):
                out["b_lev"] = debt[c] / eq[c]
            ol = _r(bs, "Other Liabilities")
            if pd.notna(ol.get(c)):
                out["b_liab_nw"] = ol[c] / eq[c]
            if len(cols) >= 4:
                gb, gn = fx._cagr(debt[cols[-4:]]), fx._cagr(eq[cols[-4:]])
                if pd.notna(gb) and pd.notna(gn):
                    out["b_gap"] = gb - gn
        return {k: v for k, v in out.items() if pd.notna(v)}

    if mode == "insurer":
        cols = list(bs.columns)
        ol, inv = _r(bs, "Other Liabilities"), _r(bs, "Investments")
        if cols:
            c = cols[-1]
            out["i_backing"] = fx._safe_div(inv.get(c), ol.get(c))
            out["i_cushion"] = fx._safe_div(eq.get(c), ol.get(c))
        sales = _r(pl, "Sales", "Revenue")
        cfo = _r(cf, "Cash from Operating Activity")
        y3 = [c for c in cf.columns if c in pl.columns][-3:]
        if y3 and sales.reindex(y3).sum() > 0:
            out["i_cfo"] = cfo.reindex(y3).sum() / sales.reindex(y3).sum()
        p3 = pbt.reindex(list(pl.columns[-3:]))
        if p3.sum() > 0:
            out["i_oi"] = _r(pl, "Other Income").reindex(p3.index).sum() / p3.sum()
        if len(pl.columns):
            c = pl.columns[-1]
            out["i_margin"] = fx._safe_div(sales.get(c) - _r(pl, "Expenses").get(c), sales.get(c))
        return {k: v for k, v in out.items() if pd.notna(v)}

    if lender:
        q = tables.get("npa_quarters")
        if q is None or q.empty or "Gross NPA %" not in q.index:
            q = tables.get("quarters")
        if q is not None and not q.empty and "Gross NPA %" in q.index:
            g = pd.to_numeric(q.loc["Gross NPA %"], errors="coerce").dropna()
            n = pd.to_numeric(q.loc["Net NPA %"], errors="coerce").dropna() if "Net NPA %" in q.index else pd.Series(dtype=float)
            if len(g):
                out["gnpa"] = g.iloc[-1]
                if len(n) and g.iloc[-1] > 0:
                    out["pcr"] = 1 - n.iloc[-1] / g.iloc[-1]
        oa = _r(bs, "Other Assets").dropna()
        if len(oa) >= 4:
            out["growth"] = fx._cagr(oa.iloc[-4:])
        if len(ni.dropna()) and len(ta.dropna()) >= 2:
            c = ta.dropna().index[-1]
            avg_ta = (ta[c] + ta.shift(1)[c]) / 2 if pd.notna(ta.shift(1).get(c)) else ta[c]
            out["roa"] = fx._safe_div(ni.get(c), avg_ta)
        if len(debt.dropna()) and len(eq.dropna()):
            c = debt.dropna().index[-1]
            out["lev"] = fx._safe_div(debt.get(c), eq.get(c))
        return {k: v for k, v in out.items() if pd.notna(v)}

    ebitda = _r(pl, "Operating Profit")
    cfo, cfi = _r(cf, "Cash from Operating Activity"), _r(cf, "Cash from Investing Activity")
    yrs = [c for c in pl.columns if c in cf.columns][-5:]
    if yrs:
        e, c_, n_ = ebitda[yrs].sum(min_count=2), cfo[yrs].sum(min_count=2), ni[yrs].sum(min_count=2)
        if pd.notna(e) and e > 0 and pd.notna(c_):
            out["cfo_ebitda"] = c_ / e
        if pd.notna(n_) and n_ > 0 and pd.notna(c_):
            out["cfo_ni"] = c_ / n_
    for lab, key in (("Debtor Days", "dso"), ("Inventory Days", "dio")):
        s = _r(ra, lab).dropna()
        if len(s):
            out[f"{key}_level"] = s.iloc[-1]
        if len(s) >= 2:
            s5 = s.iloc[-6:]
            out[f"{key}_change"] = s5.iloc[-1] - s5.iloc[0]
    y3 = list(pl.columns[-3:])
    p3 = pbt[y3].sum(min_count=2) if y3 else np.nan
    if pd.notna(p3) and p3 > 0:
        oi = _r(pl, "Other Income")[y3].sum(min_count=2)
        if pd.notna(oi):
            out["other_income"] = oi / p3
        tx = (pbt[y3] * _r(pl, "Tax %").reindex(y3) / 100).sum(min_count=2)
        if pd.notna(tx):
            out["tax"] = tx / p3
    if len(ta.dropna()) >= 2 and len(cfo.dropna()):
        c = cf.columns[-1] if len(cf.columns) else None
        if c in ta.index and c in ni.index:
            avg_ta = (ta[c] + ta.shift(1).get(c, ta[c])) / 2
            out["sloan"] = fx._safe_div(ni[c] - cfo.get(c, np.nan) - cfi.get(c, np.nan), avg_ta)
    if len(eq.dropna()):
        c = eq.dropna().index[-1]
        if eq[c] > 0 and pd.notna(debt.get(c)):
            out["de"] = debt[c] / eq[c]
        fa, cw = _r(bs, "Fixed Assets").get(c), _r(bs, "CWIP").get(c)
        if pd.notna(fa) and fa > 0 and pd.notna(cw):
            out["cwip"] = cw / fa
    return {k: v for k, v in out.items() if pd.notna(v)}


def compare(target: dict, peers: dict[str, dict], mode) -> pd.DataFrame:
    """
    target: metrics dict for the company; peers: {name: metrics}.
    -> DataFrame with company value, peer median, n peers and 'worse_than' (share of peers the company is worse than).
    """
    rows = []
    for key, label, worse, flag_key, fmt in SPECS[_mode(mode)]:
        vals = [m[key] for m in peers.values() if key in m and pd.notna(m[key])]
        v = target.get(key, np.nan)
        if not vals or pd.isna(v):
            rows.append({"key": key, "Metric": label, "Company": v, "Peer median": np.median(vals) if vals else np.nan,
                         "Peers": len(vals), "worse_than": np.nan, "flag": flag_key, "fmt": fmt, "worse": worse})
            continue
        arr = np.array(vals, dtype=float)
        worse_than = float(np.mean(arr < v)) if worse == "high" else float(np.mean(arr > v))
        rows.append({"key": key, "Metric": label, "Company": v, "Peer median": float(np.median(arr)),
                     "Peers": len(vals), "worse_than": worse_than, "flag": flag_key, "fmt": fmt, "worse": worse})
    return pd.DataFrame(rows)


def fmt_value(v, fmt):
    if pd.isna(v):
        return "—"
    return {"pct": f"{v:.0%}", "pct2": f"{v:.2%}", "days": f"{v:+.0f} days", "d": f"{v:.0f} days",
            "x": f"{v:.2f}×", "num": f"{v:.2f}"}.get(fmt, f"{v:.2f}")


def apply_to_flags(flags: list, comp: pd.DataFrame) -> None:
    """Temper or sharpen flags using peer position. Records the original severity on the flag."""
    if comp is None or comp.empty:
        return
    by_flag = {r["flag"]: r for _, r in comp.iterrows() if r["flag"]}
    for f in flags:
        r = by_flag.get(f.key)
        if r is None or f.severity is None or pd.isna(r["worse_than"]) or r["Peers"] < MIN_PEERS:
            continue
        med = fmt_value(r["Peer median"], r["fmt"])
        pos = f"worse than {r['worse_than']:.0%} of {int(r['Peers'])} peers (peer median {med})"
        f.peer_note = pos
        if f.severity in (fx.RED, fx.AMBER) and r["worse_than"] <= 0.5:
            f.original_severity = f.severity
            f.severity -= 1
            f.peer_note = f"Typical for its sector — {pos}. Downgraded one step."
        elif f.severity == fx.GREEN and r["worse_than"] >= 0.85 and r["Peers"] >= 5 and f.key in UPGRADE_OK:
            f.original_severity = f.severity
            f.severity = fx.AMBER
            f.peer_note = f"Clear on absolute rules but near the bottom of its sector — {pos}."
