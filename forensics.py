"""
Forensic engine: turns the standard financial frame plus NSE filings into flags.

Every check returns a Flag. severity: 0 = clear, 1 = watch, 2 = red flag,
None = not evaluated (data missing or not applicable). Only evaluated flags
count toward the score; coverage tells you how much of the model ran.

These are screening signals, not findings. Each has legitimate explanations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GREEN, AMBER, RED = 0, 1, 2
SEV_LABEL = {None: "Not evaluated", 0: "Clear", 1: "Watch", 2: "Red flag"}


@dataclass
class Flag:
    key: str
    layer: str
    title: str
    weight: float
    severity: int | None = None
    value: str = "—"
    detail: str = ""
    why: str = ""
    benign: str = ""
    series: pd.Series | None = field(default=None, repr=False)
    peer_note: str = ""
    original_severity: int | None = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _row(fin: pd.DataFrame, k: str) -> pd.Series:
    if fin is None or k not in fin.index:
        return pd.Series(dtype=float)
    return pd.to_numeric(fin.loc[k], errors="coerce")


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        if isinstance(a, pd.Series) or isinstance(b, pd.Series):
            out = a / b
            return out.replace([np.inf, -np.inf], np.nan)
        if b is None or pd.isna(b) or b == 0 or pd.isna(a):
            return np.nan
        return a / b


def _ebitda(fin):
    e = _row(fin, "ebitda")
    alt = _row(fin, "ebit") + _row(fin, "depreciation")
    alt2 = _row(fin, "pbt") + _row(fin, "interest_expense").fillna(0) + _row(fin, "depreciation")
    return e.fillna(alt).fillna(alt2)


def _avg(s: pd.Series) -> pd.Series:
    """Average of this year and last year's balance (first year uses itself)."""
    return (s + s.shift(1).fillna(s)) / 2


def _cagr(s: pd.Series):
    s = s.dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return np.nan
    n = len(s) - 1
    return (s.iloc[-1] / s.iloc[0]) ** (1 / n) - 1


def _fmt_pct(x, d=0):
    return "—" if pd.isna(x) else f"{x * 100:.{d}f}%"


def _fmt(x, d=2):
    return "—" if pd.isna(x) else f"{x:.{d}f}"


def is_financial(info: dict) -> bool:
    sector = str(info.get("sector", "")).lower()
    industry = str(info.get("industry", "")).lower()
    if "financial" in sector:
        return True
    return any(w in industry for w in ("bank", "credit", "insurance", "capital markets",
                                       "asset management", "mortgage", "financial"))


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------
BENEISH_COEF = {
    "DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892,
    "DEPI": 0.115, "SGAI": -0.172, "TATA": 4.679, "LVGI": -0.327,
}
BENEISH_CONST = -4.84


def beneish(fin: pd.DataFrame) -> pd.DataFrame:
    """
    8-variable Beneish M-Score for every year that has a prior year.
    Missing components are set to the neutral value 1 (TATA cannot be imputed)
    and recorded in 'imputed'.
    """
    rev, cogs = _row(fin, "revenue"), _row(fin, "cogs")
    gp = _row(fin, "gross_profit").fillna(rev - cogs)
    ar, ca, ppe = _row(fin, "receivables"), _row(fin, "current_assets"), _row(fin, "net_ppe")
    sec = _row(fin, "securities").fillna(0)
    ta, dep, sga = _row(fin, "total_assets"), _row(fin, "depreciation"), _row(fin, "sga")
    cl, ltd = _row(fin, "current_liabilities"), _row(fin, "lt_debt").fillna(0)
    ni, cfo = _row(fin, "net_income"), _row(fin, "cfo")

    gm = _safe_div(gp, rev)
    comp = pd.DataFrame(index=fin.columns, dtype=float)
    comp["DSRI"] = _safe_div(_safe_div(ar, rev), _safe_div(ar, rev).shift(1))
    comp["GMI"] = _safe_div(gm.shift(1), gm)
    aq = 1 - _safe_div(ca + ppe + sec, ta)
    comp["AQI"] = _safe_div(aq, aq.shift(1))
    comp["SGI"] = _safe_div(rev, rev.shift(1))
    dr = _safe_div(dep, dep + ppe)
    comp["DEPI"] = _safe_div(dr.shift(1), dr)
    sr = _safe_div(sga, rev)
    comp["SGAI"] = _safe_div(sr, sr.shift(1))
    lv = _safe_div(cl + ltd, ta)
    comp["LVGI"] = _safe_div(lv, lv.shift(1))
    comp["TATA"] = _safe_div(ni - cfo, ta)
    comp = comp.iloc[1:]  # first year has no prior

    rows = []
    for yr, r in comp.iterrows():
        if pd.isna(r["TATA"]):
            rows.append({"year": yr, "M": np.nan, "imputed": "TATA missing", **r.to_dict()})
            continue
        imputed = [k for k in BENEISH_COEF if k != "TATA" and pd.isna(r[k])]
        vals = {k: (1.0 if pd.isna(r[k]) else r[k]) for k in BENEISH_COEF}
        m = BENEISH_CONST + sum(BENEISH_COEF[k] * vals[k] for k in BENEISH_COEF)
        rows.append({"year": yr, "M": m, "imputed": ", ".join(imputed), **r.to_dict()})
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


def altman_z2(fin: pd.DataFrame) -> pd.Series:
    """Altman Z'' (non-manufacturer / emerging-market form, without the +3.25 constant)."""
    ta = _row(fin, "total_assets")
    wc = _row(fin, "current_assets") - _row(fin, "current_liabilities")
    x1 = _safe_div(wc, ta)
    x2 = _safe_div(_row(fin, "retained_earnings"), ta)
    x3 = _safe_div(_row(fin, "ebit"), ta)
    x4 = _safe_div(_row(fin, "equity"), _row(fin, "total_liabilities"))
    return 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4


def altman_original(fin: pd.DataFrame, market_cap: float) -> float:
    """Original 1968 Altman Z for manufacturers, latest year, using current market cap (Rs crore)."""
    if fin is None or fin.empty or pd.isna(market_cap):
        return np.nan
    c = fin.columns[-1]
    ta, tl = _row(fin, "total_assets").get(c), _row(fin, "total_liabilities").get(c)
    wc = _row(fin, "current_assets").get(c) - _row(fin, "current_liabilities").get(c)
    re_, ebit, sales = _row(fin, "retained_earnings").get(c), _row(fin, "ebit").get(c), _row(fin, "revenue").get(c)
    vals = [ta, tl, wc, re_, ebit, sales]
    if any(pd.isna(v) for v in vals) or not ta or not tl:
        return np.nan
    return 1.2 * wc / ta + 1.4 * re_ / ta + 3.3 * ebit / ta + 0.6 * market_cap / tl + 1.0 * sales / ta


def sloan_accruals(fin: pd.DataFrame) -> pd.Series:
    """Cash-flow Sloan ratio: (NI − CFO − CFI) / average total assets."""
    ni, cfo, cfi = _row(fin, "net_income"), _row(fin, "cfo"), _row(fin, "cfi")
    return _safe_div(ni - cfo - cfi, _avg(_row(fin, "total_assets")))


def montier_c(fin: pd.DataFrame) -> pd.DataFrame:
    """Montier C-Score (0–6), latest year vs prior year; each test is 1 if it points to manipulation."""
    rev, cogs = _row(fin, "revenue"), _row(fin, "cogs")
    ni, cfo = _row(fin, "net_income"), _row(fin, "cfo")
    dso = _safe_div(_row(fin, "receivables"), rev) * 365
    dsi = _safe_div(_row(fin, "inventory"), cogs.fillna(rev)) * 365
    oca = _safe_div(_row(fin, "other_current_assets"), rev)
    dep_rate = _safe_div(_row(fin, "depreciation"), _row(fin, "gross_ppe").fillna(_row(fin, "net_ppe")))
    ta = _row(fin, "total_assets")
    gap = ni - cfo
    tests = pd.DataFrame(index=fin.columns, dtype=float)
    tests["NI–CFO gap widening"] = (gap > gap.shift(1)).astype(float).where(gap.notna() & gap.shift(1).notna())
    tests["DSO rising"] = (dso > dso.shift(1)).astype(float).where(dso.notna() & dso.shift(1).notna())
    tests["Inventory days rising"] = (dsi > dsi.shift(1)).astype(float).where(dsi.notna() & dsi.shift(1).notna())
    tests["Other current assets/revenue rising"] = (oca > oca.shift(1)).astype(float).where(oca.notna() & oca.shift(1).notna())
    tests["Depreciation rate falling"] = (dep_rate < dep_rate.shift(1)).astype(float).where(dep_rate.notna() & dep_rate.shift(1).notna())
    tests["Total assets growth >10%"] = ((ta / ta.shift(1) - 1) > 0.10).astype(float).where(ta.notna() & ta.shift(1).notna())
    tests = tests.iloc[1:]
    tests["available"] = tests.notna().sum(axis=1)
    tests["C"] = tests.drop(columns="available").sum(axis=1, min_count=1)
    return tests


def piotroski_f(fin: pd.DataFrame):
    """Piotroski F-Score for the latest year (quality context, not a fraud test). Returns (score, available, detail)."""
    if fin is None or fin.shape[1] < 2:
        return np.nan, 0, {}
    c, p = fin.columns[-1], fin.columns[-2]
    ta = _row(fin, "total_assets")
    roa = _safe_div(_row(fin, "net_income"), ta)
    cfo = _row(fin, "cfo")
    lev = _safe_div(_row(fin, "lt_debt").fillna(_row(fin, "total_debt")), ta)
    cr = _safe_div(_row(fin, "current_assets"), _row(fin, "current_liabilities"))
    gm = _safe_div(_row(fin, "gross_profit").fillna(_row(fin, "revenue") - _row(fin, "cogs")), _row(fin, "revenue"))
    at = _safe_div(_row(fin, "revenue"), ta)
    checks = {
        "ROA > 0": roa.get(c, np.nan) > 0 if pd.notna(roa.get(c)) else None,
        "CFO > 0": cfo.get(c, np.nan) > 0 if pd.notna(cfo.get(c)) else None,
        "ROA improving": roa[c] > roa[p] if pd.notna(roa.get(c)) and pd.notna(roa.get(p)) else None,
        "CFO > net income": cfo[c] > _row(fin, "net_income")[c] if pd.notna(cfo.get(c)) and pd.notna(_row(fin, "net_income").get(c)) else None,
        "Leverage falling": lev[c] < lev[p] if pd.notna(lev.get(c)) and pd.notna(lev.get(p)) else None,
        "Current ratio rising": cr[c] > cr[p] if pd.notna(cr.get(c)) and pd.notna(cr.get(p)) else None,
        "Gross margin rising": gm[c] > gm[p] if pd.notna(gm.get(c)) and pd.notna(gm.get(p)) else None,
        "Asset turnover rising": at[c] > at[p] if pd.notna(at.get(c)) and pd.notna(at.get(p)) else None,
    }
    avail = [v for v in checks.values() if v is not None]
    # share dilution test needs share counts; not in the standard frame, so F is out of 8
    return (float(sum(avail)) if avail else np.nan), len(avail), checks


# --------------------------------------------------------------------------
# Financial-statement flags
# --------------------------------------------------------------------------
def financial_flags(fin: pd.DataFrame, info: dict, profile=None, market_cap: float = np.nan) -> tuple[list[Flag], dict]:
    """Accounting tests. `profile` (sector.Profile) sets thresholds, weights and skips."""
    from sector import DEFAULT_TH
    L = "Financial forensics"
    flags: list[Flag] = []
    extras: dict = {}
    if fin is None or fin.empty:
        return flags, extras
    TH = profile.th if profile is not None else DEFAULT_TH

    cols = list(fin.columns)
    last = cols[-1]
    win = cols[-5:]  # up to five most recent years

    rev, ni, cfo = _row(fin, "revenue"), _row(fin, "net_income"), _row(fin, "cfo")
    ebitda = _ebitda(fin)
    ta = _row(fin, "total_assets")

    # F1 cash conversion of EBITDA
    f = Flag("cfo_ebitda", L, "Operating cash flow vs EBITDA (cumulative)", 3,
             why="Profits that never turn into cash are the most common first sign of inflated earnings.",
             benign="Fast growth tying up working capital, long-cycle project businesses, one-off working-capital build.")
    e_sum, c_sum = ebitda[win].sum(min_count=2), cfo[win].sum(min_count=2)
    if pd.notna(e_sum) and pd.notna(c_sum) and e_sum > 0:
        r = c_sum / e_sum
        t = TH["cfo_ebitda"]
        f.severity = GREEN if r >= t["green"] else AMBER if r >= t["red"] else RED
        f.value = _fmt_pct(r)
        f.detail = (f"{len(ebitda[win].dropna())} yrs: CFO ₹{c_sum:,.0f} Cr vs EBITDA ₹{e_sum:,.0f} Cr "
                    f"(≥{t['green']:.0%} clear, <{t['red']:.0%} red).")
        f.series = _safe_div(cfo, ebitda)
    flags.append(f)

    # F2 cash conversion of net profit
    f = Flag("cfo_ni", L, "Operating cash flow vs net profit (cumulative)", 2,
             why="Same idea as above at the bottom line; persistent shortfall means accruals are doing the work.",
             benign="Large non-cash income such as deferred tax credits or fair-value gains.")
    n_sum = ni[win].sum(min_count=2)
    if pd.notna(n_sum) and pd.notna(c_sum) and n_sum > 0:
        r = c_sum / n_sum
        t = TH["cfo_ni"]
        f.severity = GREEN if r >= t["green"] else AMBER if r >= t["red"] else RED
        f.value = _fmt_pct(r)
        f.detail = f"CFO ₹{c_sum:,.0f} Cr vs net profit ₹{n_sum:,.0f} Cr (≥{t['green']:.0%} clear, <{t['red']:.0%} red)."
    flags.append(f)

    # F3 profits but persistently negative FCF
    f = Flag("fcf", L, "Profitable every year but free cash flow negative", 1.5,
             why="A business that reports profits for years yet never generates free cash depends on outside money.",
             benign="Deliberate heavy expansion capex; check whether capacity and revenue followed.")
    fcf = cfo + _row(fin, "capex").fillna(0)
    sub = pd.DataFrame({"ni": ni[win], "fcf": fcf[win]}).dropna()
    if len(sub) >= 3:
        if (sub["ni"] > 0).all():
            neg = int((sub["fcf"] < 0).sum())
            f.severity = RED if neg == len(sub) else AMBER if neg >= len(sub) - 1 else GREEN
            f.value = f"{neg}/{len(sub)} yrs negative"
            f.detail = "Free cash flow = CFO − capex."
        else:
            f.severity, f.value, f.detail = GREEN, "n/a", "Not profitable in every year; test not meaningful."
        f.series = fcf
    flags.append(f)

    # F4 receivable days
    f = Flag("dso", L, "Receivable days trend", 2,
             why="Receivables growing faster than sales can mean booked revenue that customers have not paid — or fictitious sales.",
             benign="Change in customer mix, government or export customers, credit offered to win market share.")
    dso = _safe_div(_row(fin, "receivables"), rev) * 365
    d = dso[win].dropna()
    if len(d) >= 2:
        chg_abs, chg_rel = d.iloc[-1] - d.iloc[0], _safe_div(d.iloc[-1] - d.iloc[0], d.iloc[0])
        t = TH["dso"]
        f.severity = (RED if (chg_rel > t["red_rel"] and chg_abs > t["red_abs"]) else
                      AMBER if (chg_rel > t["amber_rel"] and chg_abs > t["amber_abs"]) else GREEN)
        f.value = f"{d.iloc[0]:.0f} → {d.iloc[-1]:.0f} days"
        f.detail = f"{d.index[0]} to {d.index[-1]}; red if up >{t['red_rel']:.0%} and >{t['red_abs']} days."
        f.series = dso
    flags.append(f)

    # F5 inventory days
    f = Flag("dio", L, "Inventory days trend", 1.5,
             why="Inventory piling up faster than sales can hide unsold or overstated stock.",
             benign="Planned stocking ahead of demand, commodity price swings, new store or product launches.")
    base = _row(fin, "cogs").fillna(rev)
    dio = _safe_div(_row(fin, "inventory"), base) * 365
    d = dio[win].dropna()
    if len(d) >= 2 and d.iloc[0] > 0:
        chg_abs, chg_rel = d.iloc[-1] - d.iloc[0], _safe_div(d.iloc[-1] - d.iloc[0], d.iloc[0])
        t = TH["dio"]
        f.severity = (RED if (chg_rel > t["red_rel"] and chg_abs > t["red_abs"]) else
                      AMBER if (chg_rel > t["amber_rel"] and chg_abs > t["amber_abs"]) else GREEN)
        f.value = f"{d.iloc[0]:.0f} → {d.iloc[-1]:.0f} days"
        f.detail = (f"{d.index[0]} to {d.index[-1]}; red if up >{t['red_rel']:.0%} and >{t['red_abs']} days. "
                    "Computed on cost of goods sold where available, otherwise on revenue.")
        f.series = dio
    flags.append(f)

    # F6 Beneish
    f = Flag("beneish", L, "Beneish M-Score (latest year)", 3,
             why="Statistical model built on companies later caught manipulating earnings.",
             benign="High sales growth alone pushes the score up; it also produces meaningful false positives.")
    bm = beneish(fin)
    extras["beneish"] = bm
    if not bm.empty and pd.notna(bm["M"].iloc[-1]):
        m = bm["M"].iloc[-1]
        imputed = bm["imputed"].iloc[-1]
        n_imp = len([x for x in imputed.split(",") if x.strip()]) if imputed else 0
        if n_imp <= 3:
            f.severity = RED if m > -1.78 else AMBER if m > -2.22 else GREEN
            f.value = _fmt(m)
            f.detail = ("Above −1.78 = red, −2.22 to −1.78 = watch."
                        + (f" Set to neutral (missing data): {imputed}." if imputed else ""))
        else:
            f.value = _fmt(m)
            f.detail = f"Too many components missing ({imputed}); shown but not scored."
        f.series = bm["M"]
    flags.append(f)

    # F7 Sloan
    f = Flag("sloan", L, "Sloan accrual ratio (cash-flow method, latest)", 2,
             why="High accruals mean earnings are running ahead of cash; Sloan showed such earnings tend to reverse.",
             benign="Large one-off investing inflows or outflows distort the cash-flow version.")
    sl = sloan_accruals(fin)
    if pd.notna(sl.get(last, np.nan)):
        x = sl[last]
        f.severity = RED if x > 0.25 else AMBER if (x > 0.10 or x < -0.25) else GREEN
        f.value = _fmt_pct(x, 1)
        f.detail = "(Net profit − CFO − CFI) ÷ average total assets. Within ±10% clear; above 25% red."
        f.series = sl
    flags.append(f)

    # F8 Montier C
    f = Flag("montier", L, "Montier C-Score (persistence over 3 yrs)", 1.5,
             why="Counts six classic 'cooking the books' symptoms; 4 or more is a warning.",
             benign="Each test can trip during normal expansion; read it alongside the cash-flow checks.")
    mc = montier_c(fin)
    extras["montier"] = mc
    if not mc.empty and mc["available"].iloc[-1] >= 4:
        c = mc["C"].iloc[-1]
        ok = mc[mc["available"] >= 4]["C"].iloc[-3:]
        avg = ok.mean()
        # single-year binary tests trip easily in growing companies: require persistence for red
        f.severity = RED if (avg >= 4 and c >= 4) else AMBER if (avg >= 3 or c >= 4) else GREEN
        f.value = f"{int(c)} / {int(mc['available'].iloc[-1])} (3-yr avg {avg:.1f})"
        f.detail = ("Latest year's tests that tripped: "
                    + (", ".join(k for k, v in mc.iloc[-1].drop(["available", "C"]).items() if v == 1) or "none")
                    + ". Red needs a 3-year average of 4 or more; a single year at 4+ is watch.")
    flags.append(f)

    # F9 cash yield
    f = Flag("cash_yield", L, "Return earned on reported cash", 2.5,
             why="A big cash pile that earns almost nothing may not exist, may be pledged, or may be parked with related parties.",
             benign="Cash raised near year-end, cash held abroad at low rates, gains booked elsewhere (e.g. mutual-fund units).")
    cash = _row(fin, "cash")
    inc = _row(fin, "interest_income").fillna(_row(fin, "other_income"))
    used_other = pd.isna(_row(fin, "interest_income").get(last, np.nan))
    avg_cash = _avg(cash)
    if pd.notna(avg_cash.get(last)) and pd.notna(ta.get(last)) and pd.notna(inc.get(last)):
        share = avg_cash[last] / ta[last]
        if share < 0.05:
            f.severity, f.value = GREEN, "cash small"
            f.detail = f"Average cash is {_fmt_pct(share)} of assets; test not meaningful."
        else:
            y = _safe_div(inc[last], avg_cash[last])
            f.severity = RED if y < TH["cash_yield"]["red"] else AMBER if y < TH["cash_yield"]["amber"] else GREEN
            f.value = _fmt_pct(y, 1)
            f.detail = (f"{'Other income' if used_other else 'Interest income'} ÷ average cash & short-term investments "
                        f"(cash = {_fmt_pct(share)} of assets). Below 2% red, 2–4% watch.")
        f.series = _safe_div(inc, avg_cash)
    flags.append(f)

    # F10 cash-and-debt paradox
    f = Flag("cash_debt", L, "High cash and high debt together", 1.5,
             why="Borrowing heavily while claiming large cash balances makes little economic sense and was a feature of several Indian frauds.",
             benign="Cash ring-fenced for a project, trapped in subsidiaries, or timing around a fund-raise.")
    debt = _row(fin, "total_debt")
    if pd.notna(cash.get(last)) and pd.notna(debt.get(last)) and pd.notna(ta.get(last)) and ta[last] > 0:
        cs, ds = cash[last] / ta[last], debt[last] / ta[last]
        t = TH["cash_debt"]
        f.severity = RED if (cs > t["red"] and ds > t["red"]) else AMBER if (cs > t["amber"] and ds > t["amber"]) else GREEN
        f.value = f"cash {_fmt_pct(cs)} / debt {_fmt_pct(ds)}"
        f.detail = "Both as a share of total assets, latest year."
    flags.append(f)

    # F11 effective tax rate
    f = Flag("tax", L, "Tax charge vs reported profit (3-yr)", 1.5,
             why="Profits the tax authorities never see are a classic sign that book profits are not real.",
             benign="SEZ or other exemptions, carried-forward losses, MAT credits, large exempt income.")
    pbt, tax = _row(fin, "pbt")[cols[-3:]], _row(fin, "tax")[cols[-3:]]
    if pbt.notna().sum() >= 2 and pbt.sum() > 0:
        etr = tax.sum() / pbt.sum()
        f.severity = RED if etr < TH["tax"]["red"] else AMBER if etr < TH["tax"]["amber"] else GREEN
        f.value = _fmt_pct(etr, 1)
        f.detail = "Book tax ÷ profit before tax; India's base corporate rate is about 25%."
        tp = _row(fin, "tax_paid")[cols[-3:]]
        if tp.notna().sum() >= 2 and tax.sum() > 0:
            ratio = tp.sum() / tax.sum()
            f.detail += f" Cash tax paid is {_fmt_pct(ratio)} of the book tax charge."
            if ratio < 0.3 and f.severity == GREEN:
                f.severity = AMBER
                f.detail += " Cash tax well below the book charge — check deferred-tax notes."
    flags.append(f)

    # F12 non-operating income dependence
    f = Flag("other_income", L, "Profit dependence on non-operating income (3-yr)", 1.5,
             why="When 'other income' carries the profit line, the core business may be weaker than it looks.",
             benign="Genuine treasury income for cash-rich companies, or holding-company structures.")
    oi = _row(fin, "other_income")[cols[-3:]]
    pbt3 = _row(fin, "pbt")[cols[-3:]]
    if oi.notna().sum() >= 2 and pbt3.sum() > 0:
        s = oi.sum() / pbt3.sum()
        t = TH["other_income"]
        f.severity = RED if s > t["red"] else AMBER if s > t["amber"] else GREEN
        f.value = _fmt_pct(s)
        f.detail = f"Non-operating income ÷ profit before tax. Above {t['red']:.0%} red."
    flags.append(f)

    # F13 CWIP build-up
    f = Flag("cwip", L, "Capital work-in-progress build-up", 1,
             why="Spending can be parked in CWIP for years without being depreciated, flattering profits.",
             benign="Genuine large capacity expansion still under construction.")
    cw, ppe = _row(fin, "cwip"), _row(fin, "net_ppe")
    ratio = _safe_div(cw, ppe)
    r = ratio[win].dropna()
    if len(r) >= 2:
        rising = len(r) >= 3 and (r.diff().dropna() > 0).all()
        t = TH["cwip"]
        f.severity = RED if r.iloc[-1] > t["red"] else AMBER if (r.iloc[-1] > t["amber"] and rising) else GREEN
        f.value = f"{_fmt_pct(r.iloc[-1])} of net block"
        f.detail = f"CWIP ÷ net fixed assets; watch if above {t['amber']:.0%} and rising every year, red above {t['red']:.0%}."
        f.series = ratio
    flags.append(f)

    # F14 intangibles running ahead of revenue
    f = Flag("intangibles", L, "Intangibles / goodwill growing faster than revenue", 1,
             why="Capitalising costs as intangibles moves expenses off the profit and loss account.",
             benign="Acquisitions, or genuine product development under Ind AS 38.")
    it = _row(fin, "intangibles")[win].dropna()
    if len(it) >= 2 and pd.notna(ta.get(last)):
        g_it, g_rev = _cagr(it), _cagr(rev[win])
        share_now = _safe_div(it.iloc[-1], ta[it.index[-1]])
        share_then = _safe_div(it.iloc[0], ta[it.index[0]])
        if pd.notna(share_now):
            fast = pd.notna(g_it) and pd.notna(g_rev) and g_it > max(2 * g_rev, g_rev + 0.15)
            jump = pd.notna(share_then) and (share_now - share_then) > 0.05
            f.severity = AMBER if (fast and jump) else GREEN
            f.value = f"{_fmt_pct(share_now)} of assets"
            f.detail = f"Intangibles growth {_fmt_pct(g_it)} a year vs revenue {_fmt_pct(g_rev)}."
    flags.append(f)

    # F15 leverage creep
    f = Flag("debt", L, "Debt rising faster than the business", 1,
             why="Borrowing outpacing revenue while profits look healthy suggests the cash is not coming from operations.",
             benign="Funded expansion with a clear payback; check the capex and CWIP lines.")
    eq = _row(fin, "equity")
    if pd.notna(debt.get(last)) and pd.notna(eq.get(last)) and eq[last] > 0:
        de = debt[last] / eq[last]
        g_d, g_r = _cagr(debt[win]), _cagr(rev[win])
        fast = pd.notna(g_d) and pd.notna(g_r) and g_d > g_r + 0.20
        f.severity = RED if (fast and de > TH["debt"]["red"]) else AMBER if (fast and de > TH["debt"]["amber"]) else GREEN
        f.value = f"D/E {de:.2f}"
        f.detail = f"Debt growth {_fmt_pct(g_d)} a year vs revenue {_fmt_pct(g_r)}."
    flags.append(f)

    # F16 Altman Z'' (distress, not fraud)
    f = Flag("altman", L, "Altman Z″ (financial distress)", 1.5,
             why="Distress gives management a motive to dress up the numbers.",
             benign="Asset-light or high-growth firms can score low without being in trouble.")
    z = altman_z2(fin)
    extras["altman"] = z
    zo = altman_original(fin, market_cap) if TH["altman"]["model"] == "z_orig" else np.nan
    if pd.notna(zo):
        f.title = "Altman Z (manufacturing, financial distress)"
        f.severity = RED if zo < 1.81 else AMBER if zo < 2.99 else GREEN
        f.value = _fmt(zo)
        f.detail = "Original Z with today's market value of equity: below 1.81 distress, 1.81–2.99 grey zone."
    elif pd.notna(z.get(last)):
        f.severity = RED if z[last] < 1.1 else AMBER if z[last] < 2.6 else GREEN
        f.value = _fmt(z[last])
        f.detail = "Z″ form for non-manufacturers: below 1.1 distress, 1.1–2.6 grey zone."
        f.series = z
    flags.append(f)

    extras["piotroski"] = piotroski_f(fin)

    if profile is not None:
        for f in flags:
            if profile.keep_only and f.key not in profile.keep_only:
                f.severity, f.value = None, "n/a"
                f.detail = f"Not scored for {profile.label.lower()}; see the {profile.label.lower()} checks."
                continue
            if f.key in profile.skip:
                f.severity, f.value = None, "n/a"
                f.detail = f"Not scored for {profile.label.lower()}: {profile.skip[f.key]}."
            if f.key in profile.weights:
                f.weight = profile.weights[f.key]
        if not profile.financial_layer:
            for f in flags:
                f.severity, f.value = None, "n/a"
                f.detail = f"Not applicable to {profile.label.lower()} accounts."
            extras["financial_company"] = True
        return flags, extras

    if is_financial(info):
        # Lending balance sheets break these models (receivables = loans, cash = float, debt = deposits)
        for f in flags:
            f.severity, f.value = None, "n/a"
            f.detail = "Not applicable to banks, NBFCs or insurers."
        extras["financial_company"] = True
    return flags, extras


# --------------------------------------------------------------------------
# Ownership / governance flags
# --------------------------------------------------------------------------
def ownership_flags(sh: pd.DataFrame | None, pl: pd.DataFrame | None,
                    ins: pd.DataFrame | None) -> list[Flag]:
    L = "Ownership & pledges"
    flags = []

    f = Flag("promoter_change", L, "Promoter holding change (about 1 year)", 2,
             why="Insiders cutting their stake while the story stays bullish is a classic warning.",
             benign="Sales to meet minimum public shareholding, estate planning, a strategic investor coming in.")
    if sh is not None and len(sh) >= 2:
        latest = sh.iloc[-1]
        ref = sh[sh["date"] <= latest["date"] - pd.Timedelta(days=330)]
        ref = ref.iloc[-1] if len(ref) else sh.iloc[0]
        chg = latest["promoter_pct"] - ref["promoter_pct"]
        f.severity = RED if chg < -5 else AMBER if chg < -2 else GREEN
        f.value = f"{ref['promoter_pct']:.2f}% → {latest['promoter_pct']:.2f}%"
        f.detail = f"{ref['date']:%b %Y} to {latest['date']:%b %Y}. Drop of more than 5 points is red."
    flags.append(f)

    f = Flag("pledge", L, "Promoter shares pledged", 2.5,
             why="Heavily pledged promoters face margin calls if the price falls, which can force selling — and pledges often fund other ventures.",
             benign="Pledges for group borrowing that is well serviced; still a risk in a sharp fall.")
    if pl is not None and len(pl):
        now = pl["pledged_pct_of_promoter"].iloc[-1]
        prior = pl[pl["date"] <= pl["date"].iloc[-1] - pd.Timedelta(days=330)]
        then = prior["pledged_pct_of_promoter"].iloc[-1] if len(prior) else np.nan
        rising = pd.notna(then) and now - then > 5
        f.severity = RED if now > 30 else AMBER if (now > 10 or rising) else GREEN
        f.value = f"{now:.1f}% of promoter holding"
        hist = f"A year earlier: {then:.1f}%." if pd.notna(then) else "NSE returned only the latest quarter, so the trend isn't checked."
        f.detail = f"As of {pl['date'].iloc[-1]:%b %Y}. {hist} Above 30% red; above 10% or rising >5 points watch."
    flags.append(f)

    f = Flag("insider", L, "Promoter-group trading (12 months)", 1.5,
             why="Net selling or fresh pledge creation by the promoter group while results look strong deserves a question.",
             benign="Gifts and inter-se transfers within the family show up here too.")
    if ins is not None and len(ins):
        prom = ins[ins["category"].str.contains("promoter", case=False, na=False)]
        sells = prom[prom["txn_type"].str.contains("sell|sale|disposal", case=False, na=False)]["shares"].sum()
        buys = prom[prom["txn_type"].str.contains("buy|purchase|acquisition", case=False, na=False)]["shares"].sum()
        pledges = prom[prom["txn_type"].str.contains("pledge", case=False, na=False) &
                       ~prom["txn_type"].str.contains("revok|release|invok", case=False, na=False)]
        net = buys - sells
        f.severity = AMBER if (sells > buys * 1.5 and sells > 0) or len(pledges) else GREEN
        f.value = f"net {net:,.0f} shares"
        f.detail = f"Promoter-group buys {buys:,.0f}, sells {sells:,.0f}, pledge creations {len(pledges)}."
    flags.append(f)
    return flags


# --------------------------------------------------------------------------
# Announcement scan
# --------------------------------------------------------------------------
ANN_RULES = [
    # (severity, label, regex applied to subject + text)
    (RED, "Auditor resignation", r"resign\w*.{0,80}(statutory )?auditor|auditor.{0,80}resign"),
    (RED, "Forensic audit", r"forensic audit"),
    (RED, "Fraud", r"\bfraud"),
    (RED, "Search / seizure / ED", r"search (and|&) seizure|search operation|enforcement directorate|\bED\b raid"),
    (RED, "SEBI order / show-cause", r"\bsebi\b.{0,60}(order|show[- ]cause|adjudicat|interim)"),
    (RED, "Default / delayed debt payment", r"\bdefault\b|delay in (payment|servicing)|non[- ]payment of (interest|principal)"),
    (RED, "Insolvency / NCLT", r"insolvency|\bnclt\b|\bcirp\b"),
    (RED, "Modified audit opinion", r"qualified opinion|modified opinion|adverse opinion|disclaimer of opinion"),
    (RED, "Credit rating downgrade", r"rating.{0,60}downgrad|downgrad.{0,60}rating"),
    (AMBER, "CFO resignation", r"(chief financial officer|\bcfo\b).{0,80}resign|resign.{0,80}(chief financial officer|\bcfo\b)"),
    (AMBER, "Company secretary resignation", r"company secretary.{0,80}resign|resign.{0,80}company secretary"),
    (AMBER, "Independent director resignation",
     r"(?<!non-)(?<!non )\bindependent director.{0,80}resign|resign.{0,80}(?<!non-)(?<!non )\bindependent director"),
    (AMBER, "Auditor change", r"change in (the )?(statutory )?auditor|(appointment|re-?appointment).{0,40}statutory auditor|casual vacancy"),
    (AMBER, "Preferential issue / warrants", r"preferential (issue|allot)|\bwarrants?\b"),
    (AMBER, "Pledge / encumbrance", r"pledge|encumbrance"),
    (AMBER, "Tax / GST demand or notice", r"(gst|income[- ]tax|tax).{0,60}(demand|notice|order|penalt)"),
    (AMBER, "Penalty / fine", r"\bpenalt|\bfine\b|levied"),
]


def scan_announcements(ann: pd.DataFrame | None) -> pd.DataFrame:
    if ann is None or ann.empty:
        return pd.DataFrame(columns=["date", "severity", "category", "subject", "text", "link"])
    hits = []
    for _, r in ann.iterrows():
        blob = f"{r.get('subject', '')} {r.get('text', '')}".lower()
        # secretarial / cost / internal auditor filings are routine (secretarial is now mandatory)
        blob = re.sub(r"(secretarial|cost|internal) auditors?", " ", blob)
        for sev, label, rx in ANN_RULES:
            if re.search(rx, blob, flags=re.I):
                hits.append({"date": r.get("date"), "severity": sev, "category": label,
                             "subject": r.get("subject", ""), "text": str(r.get("text", ""))[:400],
                             "link": r.get("link", "")})
                break  # one category per filing: the most severe rule listed first wins
    return pd.DataFrame(hits)


def announcement_flag(hits: pd.DataFrame | None, evaluated: bool) -> Flag:
    f = Flag("announcements", "Filings & events", "Exchange filings with warning keywords", 2.5,
             why="Auditor exits, regulatory orders, defaults and resignations are often the first public trace of trouble.",
             benign="Keyword matches can be routine (e.g. a scheduled auditor rotation); open each filing.")
    if not evaluated:
        return f
    if hits is None or hits.empty:
        f.severity, f.value, f.detail = GREEN, "none", "No filings matched the warning keywords."
        return f
    reds = (hits["severity"] == RED).sum()
    ambers = (hits["severity"] == AMBER).sum()
    f.severity = RED if reds else AMBER if ambers >= 2 else GREEN
    f.value = f"{reds} red, {ambers} watch"
    top = hits.sort_values(["severity", "date"], ascending=[False, False]).head(3)
    f.detail = "; ".join(f"{d:%d %b %Y} {c}" if pd.notna(d) else c
                         for d, c in zip(top["date"], top["category"]))
    return f


# --------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------
def composite(flags: list[Flag]) -> dict:
    total_w = sum(f.weight for f in flags)
    ev = [f for f in flags if f.severity is not None]
    ev_w = sum(f.weight for f in ev)
    score = (100 * sum(f.weight * f.severity for f in ev) / (2 * ev_w)) if ev_w else np.nan
    band = ("Not enough data" if pd.isna(score) else
            "Low concern" if score < 20 else
            "Some questions" if score < 35 else
            "Elevated — dig deeper" if score < 55 else
            "High — serious scrutiny warranted")
    by_layer = {}
    for layer in dict.fromkeys(f.layer for f in flags):
        lf = [f for f in flags if f.layer == layer]
        le = [f for f in lf if f.severity is not None]
        lw = sum(f.weight for f in le)
        by_layer[layer] = {
            "score": (100 * sum(f.weight * f.severity for f in le) / (2 * lw)) if lw else np.nan,
            "evaluated": len(le), "total": len(lf),
            "red": sum(f.severity == RED for f in le), "amber": sum(f.severity == AMBER for f in le),
        }
    return {"score": score, "band": band, "coverage": (ev_w / total_w) if total_w else 0.0,
            "red": sum(f.severity == RED for f in ev), "amber": sum(f.severity == AMBER for f in ev),
            "evaluated": len(ev), "total": len(flags), "by_layer": by_layer}


# --------------------------------------------------------------------------
# Lender (bank / NBFC) flags — built from Screener's bank layout
# --------------------------------------------------------------------------
def _quarter_series(tables: dict, label: str) -> pd.Series:
    q = (tables or {}).get("npa_quarters")
    if q is None or q.empty or label not in q.index:
        q = (tables or {}).get("quarters")
    if q is None or q.empty or label not in q.index:
        return pd.Series(dtype=float)
    s = pd.to_numeric(q.loc[label], errors="coerce")
    s.index = [pd.to_datetime(c, format="%b %Y", errors="coerce") for c in s.index]
    return s[s.index.notna()].dropna().sort_index()


def lender_flags(fin: pd.DataFrame, tables: dict, sched: dict, kind: str) -> tuple[list[Flag], dict]:
    """kind: 'bank' or 'nbfc'."""
    L = "Lender forensics"
    flags: list[Flag] = []
    extras: dict = {}
    bank = kind == "bank"
    cols = list(fin.columns) if fin is not None else []
    last = cols[-1] if cols else None

    gnpa, nnpa = _quarter_series(tables, "Gross NPA %"), _quarter_series(tables, "Net NPA %")
    extras["npa"] = pd.DataFrame({"Gross NPA %": gnpa, "Net NPA %": nnpa})

    # L1 asset quality
    f = Flag("gnpa", L, "Gross NPA level and trend", 3,
             why="Rising bad loans are the core risk in lending; sudden jumps often follow years of 'evergreening'.",
             benign="A one-off recognition drive or regulatory reclassification can raise NPAs while cleaning the book.")
    if len(gnpa) >= 2:
        now = gnpa.iloc[-1]
        ref = gnpa[gnpa.index <= gnpa.index[-1] - pd.Timedelta(days=330)]
        then = ref.iloc[-1] if len(ref) else gnpa.iloc[0]
        rise = now - then
        lvl_red, lvl_amber = (6, 3) if bank else (5, 3)
        f.severity = RED if (now > lvl_red or rise > 1.5) else AMBER if (now > lvl_amber or rise > 0.5) else GREEN
        f.value = f"{now:.2f}%"
        f.detail = (f"{gnpa.index[-1]:%b %Y}; a year earlier {then:.2f}% ({rise:+.2f} pts). "
                    f"Red above {lvl_red}% or a rise of more than 1.5 points.")
        f.series = gnpa
    flags.append(f)

    # L2 provision coverage (proxy)
    f = Flag("pcr", L, "Provision coverage (approx.)", 2,
             why="Thin provisioning against bad loans flatters profit and leaves less buffer if recoveries disappoint.",
             benign="Well-collateralised loans (gold, housing) can justify lower coverage; write-offs aren't counted here.")
    if len(gnpa) and len(nnpa) and gnpa.iloc[-1] > 0:
        pcr = 1 - nnpa.iloc[-1] / gnpa.iloc[-1]
        f.severity = RED if pcr < 0.50 else AMBER if pcr < 0.65 else GREEN
        if gnpa.iloc[-1] < 1.0 and f.severity == RED:
            f.severity = AMBER  # tiny NPA base
        f.value = _fmt_pct(pcr)
        f.detail = "1 − Net NPA ÷ Gross NPA (ignores technical write-offs). Below 50% red, 50–65% watch."
    flags.append(f)

    # loan book proxy: Screener 'Other Assets' minus cash (advances sit inside it)
    loans = _row(fin, "current_assets") - _row(fin, "cash").fillna(0)
    deposits, borrow, eq, ta = _row(fin, "deposits"), _row(fin, "total_debt"), _row(fin, "equity"), _row(fin, "total_assets")
    extras["loans_proxy"] = loans

    # L3 growth
    f = Flag("loan_growth", L, "Loan book growth (3-yr)", 2,
             why="Very fast lending growth tends to come before asset-quality trouble (the Yes Bank / DHFL pattern).",
             benign="A small or newly listed lender can grow fast from a low base with sound underwriting.")
    lg = loans[cols[-4:]].dropna() if cols else pd.Series(dtype=float)
    if len(lg) >= 3:
        g = _cagr(lg)
        amber_g, red_g = (0.30, 0.40) if bank else (0.35, 0.50)
        rising_npa = len(gnpa) >= 5 and gnpa.iloc[-1] - gnpa.iloc[-5] > 0.3
        f.severity = RED if (g > red_g or (g > amber_g and rising_npa)) else AMBER if g > amber_g else GREEN
        f.value = f"{_fmt_pct(g)} a year"
        f.detail = (f"Loan-book proxy (Screener 'Other Assets' less cash). Watch above {amber_g:.0%} a year, "
                    f"red above {red_g:.0%} or when fast growth coincides with rising NPAs.")
    flags.append(f)

    # L4 funding / leverage
    if bank:
        f = Flag("funding", L, "Loan growth vs deposit growth", 2,
                 why="Lending well beyond the deposit base means reliance on costlier, flightier wholesale money.",
                 benign="Merged entities (e.g. a parent NBFC folded in) carry high ratios for a few years.")
        gd, gl = _cagr(deposits[cols[-4:]]) if cols else np.nan, _cagr(loans[cols[-4:]]) if cols else np.nan
        if pd.notna(gd) and pd.notna(gl):
            gap = gl - gd
            f.severity = RED if gap > 0.15 else AMBER if gap > 0.08 else GREEN
            f.value = f"loans {_fmt_pct(gl)} vs deposits {_fmt_pct(gd)} a year"
            cd = _safe_div(loans.get(last), deposits.get(last))
            f.detail = ("3-year growth. Loans outgrowing deposits by more than 8 points a year is watch, 15 red. "
                        f"Rough loan/deposit level from Screener's groupings: {_fmt_pct(cd)} (reads high; not scored).")
    else:
        f = Flag("leverage", L, "Leverage (borrowings ÷ net worth)", 2,
                 why="Highly geared NBFCs are exposed to funding shocks (IL&FS 2018).",
                 benign="Housing-finance companies run structurally higher leverage than other NBFCs.")
        if last and pd.notna(borrow.get(last)) and pd.notna(eq.get(last)) and eq[last] > 0:
            de = borrow[last] / eq[last]
            f.severity = RED if de > 8 else AMBER if de > 6 else GREEN
            f.value = f"{de:.1f}×"
            f.detail = "Above 6× watch, above 8× red."
    flags.append(f)

    # L5 capital thinness (banks)
    f = Flag("capital", L, "Equity cushion (net worth ÷ total assets)", 1.5,
             why="A thin equity cushion means small loan losses can wipe out capital; this is a rough stand-in for CRAR.",
             benign="Book equity understates regulatory capital (Tier-2 bonds, etc.); check the reported CRAR.")
    if bank and last and pd.notna(eq.get(last)) and pd.notna(ta.get(last)) and ta[last] > 0:
        r = eq[last] / ta[last]
        f.severity = RED if r < 0.055 else AMBER if r < 0.07 else GREEN
        f.value = _fmt_pct(r, 1)
        f.detail = "Below 7% watch, below 5.5% red."
    elif not bank:
        f.value, f.detail = "n/a", "Covered by the leverage check for NBFCs."
    flags.append(f)

    # L7 profitability trend
    f = Flag("roa", L, "Return on assets trend", 1,
             why="Falling profitability raises the incentive to delay recognising bad loans.",
             benign="Deliberate investment phases, rate cycles compressing margins.")
    roa = _safe_div(_row(fin, "net_income"), _avg(ta))
    r3 = roa[cols[-4:]].dropna() if cols else pd.Series(dtype=float)
    if len(r3) >= 3:
        falling = (r3.diff().dropna() < 0).all()
        floor = 0.005 if bank else 0.01
        f.severity = RED if r3.iloc[-1] < 0 else AMBER if (falling and r3.iloc[-1] < floor) else GREEN
        f.value = _fmt_pct(r3.iloc[-1], 2)
        f.detail = f"Net profit ÷ average total assets; watch if falling every year and below {floor:.1%}."
        f.series = roa
    flags.append(f)
    return flags, extras


# --------------------------------------------------------------------------
# Shared helpers for Screener-table based modules
# --------------------------------------------------------------------------
def annual_tables(tables: dict) -> dict:
    """Screener tables -> annual-only frames keyed by FY label (drops TTM and interim columns)."""
    from data_sources import _period_to_fy
    pl_raw = (tables or {}).get("profit-loss", pd.DataFrame())
    months = [_period_to_fy(p)[1] for p in pl_raw.columns if _period_to_fy(p)[0]]
    ye = int(pd.Series(months).mode().iloc[0]) if months else 3
    out = {}
    for sec in ("profit-loss", "balance-sheet", "cash-flow", "ratios"):
        df = (tables or {}).get(sec, pd.DataFrame())
        cols = {p: _period_to_fy(p)[0] for p in df.columns if _period_to_fy(p)[0] and _period_to_fy(p)[1] == ye}
        out[sec] = df[list(cols)].rename(columns=cols) if cols else pd.DataFrame()
    out["ye_month"] = ye
    return out


def _t(df: pd.DataFrame, *labels) -> pd.Series:
    for lab in labels:
        if df is not None and not df.empty and lab in df.index:
            return pd.to_numeric(df.loc[lab], errors="coerce")
    return pd.Series(dtype=float)


def _sched_series(sched: dict, parent: str, item: str, ye_month: int) -> pd.Series:
    from data_sources import _period_to_fy
    d = (sched or {}).get(parent, {}).get(item, {})
    return pd.Series({_period_to_fy(k)[0]: v for k, v in d.items() if _period_to_fy(k)[1] == ye_month}, dtype=float)


# --------------------------------------------------------------------------
# Stockbroker flags
# --------------------------------------------------------------------------
def broker_flags(tables: dict, sched: dict) -> tuple[list[Flag], dict]:
    L = "Broker forensics"
    flags, extras = [], {}
    a = annual_tables(tables)
    bs, ye = a["balance-sheet"], a["ye_month"]
    cols = list(bs.columns)
    nw = _t(bs, "Equity Capital").add(_t(bs, "Reserves"), fill_value=0)
    ol = _t(bs, "Other Liabilities")
    nci = _sched_series(sched, "Other Liabilities", "Non controlling int", ye).reindex(ol.index).fillna(0)
    client_liab = ol - nci
    cash = _sched_series(sched, "Other Assets", "Cash Equivalents", ye).reindex(ol.index)
    borrow = _t(bs, "Borrowings", "Borrowing")
    recv = _sched_series(sched, "Other Assets", "Trade receivables", ye).reindex(ol.index)

    f = Flag("b_cover", L, "Client-money cover (cash & bank ÷ client liabilities)", 3,
             why="Client funds must be held in segregated bank accounts. Cash well short of what the firm owes "
                 "clients can mean client money is funding the business — the Karvy pattern.",
             benign="Client money placed as margin with exchanges or in fixed deposits can sit outside 'cash' on "
                    "Screener; liabilities also include non-client items.")
    cover = _safe_div(cash, client_liab)
    extras["cover"] = cover
    cv = cover[cols].dropna() if cols else pd.Series(dtype=float)
    if len(cv):
        now = cv.iloc[-1]
        then = cv.iloc[-4] if len(cv) >= 4 else cv.iloc[0]
        drop = (then - now) / then if then > 0 else np.nan
        f.severity = RED if now < 0.8 else AMBER if (now < 1.0 or (pd.notna(drop) and drop > 0.30)) else GREEN
        f.value = f"{now:.2f}×"
        f.detail = (f"{cv.index[-1]}; {cv.index[-4] if len(cv) >= 4 else cv.index[0]} was {then:.2f}×. "
                    "Below 0.8× red, below 1× or down >30% in 3 yrs watch. Liabilities = Screener's 'Other "
                    "Liabilities' less minority interest.")
        f.series = cover
    flags.append(f)

    f = Flag("b_leverage", L, "Borrowings ÷ net worth", 2,
             why="Brokers borrow mainly to fund margin-trading loans; heavy gearing magnifies any client default "
                 "or market shock.",
             benign="A growing margin-trading book funded by bank lines at sensible limits.")
    if cols and pd.notna(borrow.get(cols[-1])) and pd.notna(nw.get(cols[-1])) and nw[cols[-1]] > 0:
        lv = borrow[cols[-1]] / nw[cols[-1]]
        f.severity = RED if lv > 4 else AMBER if lv > 2.5 else GREEN
        f.value = f"{lv:.2f}×"
        f.detail = "Above 2.5× watch, above 4× red."
    flags.append(f)

    f = Flag("b_growth", L, "Debt-funded growth (borrowings vs net worth, 3-yr)", 1.5,
             why="Borrowings racing ahead of the firm's own capital shows growth bought with leverage.",
             benign="A deliberate push into margin funding, usually visible in interest income.")
    if len(cols) >= 4:
        gb, gn = _cagr(borrow[cols[-4:]]), _cagr(nw[cols[-4:]])
        big = pd.notna(borrow.get(cols[-1])) and pd.notna(nw.get(cols[-1])) and borrow[cols[-1]] > 0.5 * nw[cols[-1]]
        if pd.notna(gb) and pd.notna(gn):
            gap = gb - gn
            f.severity = RED if (gap > 0.50 and big) else AMBER if (gap > 0.25 and big) else GREEN
            f.value = f"debt {_fmt_pct(gb)} vs net worth {_fmt_pct(gn)} a year"
            f.detail = "Scored only when borrowings exceed half of net worth. Gap above 25 points watch, 50 red."
    flags.append(f)

    f = Flag("b_receivables", L, "Client receivables ÷ net worth", 1,
             why="Large unpaid client debit balances are credit risk that the firm's own capital must absorb.",
             benign="Timing of settlement around the year-end (T+1 balances).")
    if cols and pd.notna(recv.get(cols[-1])) and pd.notna(nw.get(cols[-1])) and nw[cols[-1]] > 0:
        r = recv[cols[-1]] / nw[cols[-1]]
        f.severity = RED if r > 1.0 else AMBER if r > 0.5 else GREEN
        f.value = _fmt_pct(r)
        f.detail = "Above 50% watch, above 100% red."
    flags.append(f)
    return flags, extras


# --------------------------------------------------------------------------
# Insurer flags
# --------------------------------------------------------------------------
def insurer_flags(tables: dict, sched: dict, kind: str, manual: dict | None = None) -> tuple[list[Flag], dict]:
    """kind: 'life' or 'general'. manual: optional {'solvency': %, 'combined': %, 'persistency13': %}."""
    L = "Insurer forensics"
    life = kind == "life"
    flags, extras = [], {}
    manual = manual or {}
    a = annual_tables(tables)
    pl, bs, cf = a["profit-loss"], a["balance-sheet"], a["cash-flow"]
    cols = list(bs.columns)
    nw = _t(bs, "Equity Capital").add(_t(bs, "Reserves"), fill_value=0)
    ol, inv = _t(bs, "Other Liabilities"), _t(bs, "Investments")
    sales, cfo = _t(pl, "Sales", "Revenue"), _t(cf, "Cash from Operating Activity")
    pbt, oi = _t(pl, "Profit before tax"), _t(pl, "Other Income")

    # M1 solvency (manual)
    f = Flag("i_solvency", L, "Solvency ratio (from IRDAI disclosure)", 3,
             why="The regulator's control level is 150%; insurers near it may be forced to raise capital or slow down.",
             benign="A planned capital raise can restore it quickly.")
    v = manual.get("solvency")
    if v:
        f.severity = RED if v < 150 else AMBER if v < 170 else GREEN
        f.value = f"{v:.0f}%"
        f.detail = "Entered manually. Below 150% (IRDAI control level) red, 150–170% watch."
    else:
        f.detail = "Type the latest solvency ratio from the insurer's public disclosure into the sidebar to score this."
    flags.append(f)

    # M2 combined ratio (general) or persistency (life)
    if life:
        f = Flag("i_persistency", L, "13th-month persistency (from IRDAI disclosure)", 2,
                 why="Policies lapsing in year one point to mis-selling or poor-quality growth.",
                 benign="Product-mix shifts (e.g. towards single-premium) change the measure.")
        v = manual.get("persistency13")
        if v:
            f.severity = RED if v < 70 else AMBER if v < 78 else GREEN
            f.value = f"{v:.0f}%"
            f.detail = "Entered manually (premium basis). Below 70% red, 70–78% watch."
        else:
            f.detail = "Type the 13th-month persistency from the public disclosure into the sidebar to score this."
    else:
        f = Flag("i_combined", L, "Combined ratio (from IRDAI disclosure)", 2.5,
                 why="Above 100% the insurer loses money on underwriting and relies on investment income.",
                 benign="Catastrophe years (floods, pandemic) spike it temporarily.")
        v = manual.get("combined")
        if v:
            f.severity = RED if v > 110 else AMBER if v > 104 else GREEN
            f.value = f"{v:.1f}%"
            f.detail = "Entered manually. Above 110% red, 104–110% watch."
        else:
            f.detail = "Type the latest combined ratio from the public disclosure into the sidebar to score this."
    flags.append(f)

    # I1 asset backing
    f = Flag("i_backing", L, "Investments ÷ policyholder liabilities", 2.5,
             why="Policy liabilities should be backed by investments; a falling ratio means obligations are growing "
                 "faster than the assets set aside for them.",
             benign="Liabilities on Screener include non-policy items; receivables and cash also back reserves.")
    back = _safe_div(inv, ol)
    bk = back[cols].dropna() if cols else pd.Series(dtype=float)
    if len(bk):
        now = bk.iloc[-1]
        then = bk.iloc[-4] if len(bk) >= 4 else bk.iloc[0]
        f.severity = RED if now < 0.90 else AMBER if (now < 0.95 or then - now > 0.08) else GREEN
        f.value = f"{now:.2f}×"
        f.detail = f"Three years earlier {then:.2f}×. Below 0.90× red; below 0.95× or down >0.08 watch."
        f.series = back
    flags.append(f)

    # I2 equity cushion
    f = Flag("i_cushion", L, "Net worth ÷ policyholder liabilities", 1.5,
             why="A rough stand-in for solvency headroom when the reported ratio isn't entered.",
             benign="Regulatory solvency counts more than book equity; use the reported ratio when available.")
    cush = _safe_div(nw, ol)
    if cols and pd.notna(cush.get(cols[-1])):
        c = cush[cols[-1]]
        red_t, amb_t = (0.015, 0.03) if life else (0.15, 0.22)
        f.severity = RED if c < red_t else AMBER if c < amb_t else GREEN
        f.value = _fmt_pct(c, 1)
        f.detail = f"{'Life' if life else 'General'} insurer thresholds: below {amb_t:.0%} watch, below {red_t:.1%} red."
    flags.append(f)

    # I3 operating cash flow
    f = Flag("i_cfo", L, "Operating cash flow", 2,
             why="A growing insurer collects premiums before paying claims, so operating cash flow should be "
                 "positive. Persistent outflows mean claims and costs exceed premiums.",
             benign="A run-off book, or a catastrophe year's claims.")
    c3 = cfo[[c for c in cols if c in cfo.index][-3:]].dropna() if cols else pd.Series(dtype=float)
    if len(c3) >= 2:
        neg = int((c3 < 0).sum())
        f.severity = RED if neg == len(c3) else AMBER if neg >= 2 else GREEN
        ratio = _safe_div(c3.sum(), sales[c3.index].sum())
        f.value = f"{neg}/{len(c3)} yrs negative"
        f.detail = f"3-yr operating cash flow is {_fmt_pct(ratio)} of premium income."
        f.series = cfo
    flags.append(f)

    # I4 profit reliance on investment income
    f = Flag("i_other_income", L, "Profit reliance on shareholders' investment income (3-yr)", 1.5,
             why="When profit comes mostly from investment income, the insurance business itself may be "
                 "unprofitable — and gains can be timed to smooth results.",
             benign="Life insurers' shareholder funds earn investment income by design.")
    y3 = cols[-3:]
    if y3 and pbt.reindex(y3).sum() > 0 and oi.reindex(y3).notna().sum() >= 2:
        sh = oi.reindex(y3).sum() / pbt.reindex(y3).sum()
        amb_t, red_t = (0.60, 0.85) if life else (0.40, 0.70)
        f.severity = RED if sh > red_t else AMBER if sh > amb_t else GREEN
        f.value = _fmt_pct(sh)
        f.detail = f"Other income ÷ profit before tax. Above {amb_t:.0%} watch, above {red_t:.0%} red."
    flags.append(f)

    # I5 margin trend
    f = Flag("i_margin", L, "Operating margin trend", 1,
             why="A steadily shrinking margin raises the pressure to under-reserve.",
             benign="Deliberate growth in lower-margin products or price competition.")
    opm = _safe_div(sales - _t(pl, "Expenses"), sales)
    o4 = opm[[c for c in cols if c in opm.index][-4:]].dropna() if cols else pd.Series(dtype=float)
    if len(o4) >= 4:
        falling = (o4.diff().dropna() < 0).all()
        shrink = o4.iloc[0] > 0 and o4.iloc[-1] < 0.6 * o4.iloc[0]
        f.severity = RED if o4.iloc[-1] < 0 and falling else AMBER if (falling and shrink) else GREEN
        f.value = f"{_fmt_pct(o4.iloc[0], 1)} → {_fmt_pct(o4.iloc[-1], 1)}"
        f.detail = "(Premium income − total outgo) ÷ premium income. Watch if it fell every year to under 60% of its level 3 yrs ago."
        f.series = opm
    flags.append(f)

    # I6 tax (general insurers only)
    f = Flag("i_tax", L, "Tax charge vs profit (3-yr)", 1,
             why="Book profit the tax authorities don't see is a warning sign.",
             benign="Carried-forward losses or deferred-tax credits.")
    if not life and y3:
        t = _t(pl, "Tax %").reindex(y3)
        p = pbt.reindex(y3)
        if p.sum() > 0 and t.notna().sum() >= 2:
            etr = (p * t / 100).sum() / p.sum()
            f.severity = RED if etr < 0.08 else AMBER if etr < 0.15 else GREEN
            f.value = _fmt_pct(etr, 1)
            f.detail = "Below 15% watch, below 8% red."
    elif life:
        f.value, f.detail = "n/a", "Life-insurance surplus is taxed differently; not scored."
    flags.append(f)
    return flags, extras
