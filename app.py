"""
Forensic Red-Flag Screener — NSE
Screens one listed company for the accounting, ownership and filing patterns
that forensic short-sellers start from. A screening aid, not a finding of fraud.

© Dr Shantanu Samanta
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import data_sources as ds
import forensics as fx
import peers as pr
import sector as sc

st.set_page_config(page_title="Forensic Red-Flag Screener", page_icon="🔎", layout="wide")

SEV_BADGE = {None: "⚪ Not evaluated", 0: "🟢 Clear", 1: "🟠 Watch", 2: "🔴 Red flag"}
BAND_COLOR = {"Low concern": "#1e8e3e", "Some questions": "#b08900",
              "Elevated — dig deeper": "#d9480f", "High — serious scrutiny warranted": "#c92a2a",
              "Not enough data": "#6c757d"}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.band {display:inline-block;padding:.25rem .7rem;border-radius:999px;color:#fff;font-weight:600;font-size:.9rem}
.flagcard {border-left:4px solid var(--c);padding:.45rem .8rem;margin:.35rem 0;border-radius:6px;
           background:rgba(127,127,127,.07)}
.flagcard b {font-size:.95rem}
.small {font-size:.82rem;opacity:.8}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_universe():
    return ds.load_nse_equity_list()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_financials(symbol: str):
    return ds.load_yf_financials(symbol, yf)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_screener(symbol: str, consolidated: bool, name: str | None = None):
    return ds.ScreenerClient().load(symbol, prefer_consolidated=consolidated, name=name)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_peer_metrics(path: str, mode: str):
    tables, name = ds.ScreenerClient().load_light(path)
    if tables is None:
        return {}
    if mode == "lender":
        tables["npa_quarters"] = tables.get("quarters")
    return pr.metrics_from_tables(tables, mode)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_nse(symbol: str, months: int):
    c = ds.NSEClient()
    out, errs = {}, {}
    raw_sh, errs["shareholding"] = c.shareholding(symbol)
    out["shareholding"] = ds.normalise_shareholding(raw_sh)
    raw_pl, errs["pledges"] = c.pledges(symbol)
    out["pledges"] = ds.normalise_pledges(raw_pl)
    raw_an, errs["announcements"] = c.announcements(symbol, months)
    out["announcements"] = ds.normalise_announcements(raw_an)
    raw_in, errs["insider"] = c.insider_trades(symbol, 12)
    out["insider"] = ds.normalise_insider(raw_in)
    # keep raw frames so unfamiliar field layouts can still be inspected
    out["_raw"] = {"shareholding": raw_sh, "pledges": raw_pl}
    return out, errs


# --------------------------------------------------------------------------
# Sidebar: symbol picker (same behaviour as the Forward Matrix / Fair Value apps)
# --------------------------------------------------------------------------
st.sidebar.title("🔎 Forensic Screener")
universe, uni_err = get_universe()
if universe is None:
    st.sidebar.caption("NSE symbol list unreachable — using Yahoo search instead.")

query = st.sidebar.text_input("NSE symbol or company name", key="query",
                              placeholder="e.g. TITAN or titan company")
if query and query != st.session_state.get("_last_query"):
    st.session_state["_last_query"] = query
    kind, res, name = ds.resolve_symbol(query, universe, yf)
    if kind == "exact":
        st.session_state["symbol"], st.session_state["company"] = res, name
        st.session_state["choices"] = []
    else:
        st.session_state["choices"] = res

choices = st.session_state.get("choices") or []
if choices:
    labels = [f"{s} — {n}" for s, n in choices]
    pick = st.sidebar.selectbox("Did you mean…", labels, key="pick")
    if st.sidebar.button("Open selected", width="stretch"):
        i = labels.index(pick)
        st.session_state["symbol"], st.session_state["company"] = choices[i]
        st.session_state["choices"] = []
        st.rerun()
elif query and not st.session_state.get("symbol"):
    st.sidebar.warning("No matching NSE scrip found.")

months = st.sidebar.select_slider("Filings look-back", options=[12, 24, 36], value=24,
                                  format_func=lambda m: f"{m} months")
src_choice = st.sidebar.radio("Financial data source", ["Screener.in (≈12 yrs)", "Yahoo Finance (≈4 yrs)"],
                              help="Screener is read live from the public company page. If it fails, "
                                   "the app falls back to Yahoo automatically.")
consolidated = st.sidebar.toggle("Prefer consolidated accounts", value=True,
                                 help="Falls back to standalone when a company has no consolidated figures.")
with st.sidebar.expander("Upload a Screener Excel export instead"):
    upload = st.file_uploader("Screener.in 'Export to Excel' file", type=["xlsx"],
                              help="If uploaded, its figures replace the live source for the financial tests.")
TYPE_LABELS = {p.label: k for k, p in sc.PROFILES.items()}
type_choice = st.sidebar.selectbox("Company type (rule book)", ["Auto-detect"] + list(TYPE_LABELS),
                                   help="Auto-detect uses NSE's industry classification from Screener. "
                                        "Override it if the business model differs from its label.")
use_peers = st.sidebar.toggle("Compare with sector peers", value=True,
                              help="Scores up to 8 Screener peers on the same metrics (adds ~10 s the first time).")
fetch_nse = st.sidebar.toggle("Fetch NSE filings & shareholding", value=True,
                              help="NSE sometimes blocks cloud servers; switch off for a faster financial-only run.")

symbol = st.session_state.get("symbol")

# --------------------------------------------------------------------------
# Landing
# --------------------------------------------------------------------------
if not symbol:
    st.title("Forensic Red-Flag Screener")
    st.write("Type an NSE symbol or company name in the sidebar. The app pulls the accounts, "
             "shareholding, pledge and filing history and runs the checks forensic short-sellers "
             "start from: cash conversion, accrual quality, working-capital drift, cash that earns "
             "nothing, tax vs profit, promoter pledges and selling, and warning-keyword filings.")
    st.info("It flags things worth asking about. Every flag has legitimate explanations, "
            "and a clean result does not prove the accounts are sound.")
    st.stop()

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
with st.spinner(f"Collecting data for {symbol}…"):
    fin_yf, info, fin_err = get_financials(symbol)
    fin, source, src_note = fin_yf, "Yahoo Finance", None
    sc_res, sc_err = get_screener(symbol, consolidated, st.session_state.get("company"))
    sc_meta = sc_res["meta"] if sc_res else {}
    if src_choice.startswith("Screener"):
        if sc_res is not None and sc_res["fin"].shape[1] >= 2:
            fin, source, src_note = sc_res["fin"], f"Screener.in · {sc_meta['view']}", ds.SCREENER_NOTE
            fin_err = None
        else:
            st.warning(f"Screener.in unavailable ({sc_err}); using Yahoo Finance instead.")
    if upload is not None:
        sc_fin, sc_note = ds.parse_screener_export(upload.getvalue())
        if sc_fin is not None:
            fin, source, src_note = sc_fin, "Screener.in export (uploaded)", sc_note
        else:
            st.warning(f"Screener file not used: {sc_note}")
    if fetch_nse:
        nse, nse_err = get_nse(symbol, months)
    else:
        nse, nse_err = {}, {k: "switched off" for k in ("shareholding", "pledges", "announcements", "insider")}

company = info.get("longName") or sc_meta.get("name") or st.session_state.get("company") or symbol
info = dict(info)
kind, kind_why = sc.classify(sc_meta.get("classification"), info, sc_meta.get("is_financial", False))
if type_choice != "Auto-detect":
    kind, kind_why = TYPE_LABELS[type_choice], "set manually in the sidebar"
profile = sc.PROFILES[kind]
lender = profile.lender_layer
mode = profile.peer_mode
manual = {}
if profile.insurer_layer:
    st.sidebar.markdown("**Insurer figures (optional)**")
    st.sidebar.caption("From the latest IRDAI public disclosure on the insurer's website. Leave 0 to skip.")
    manual["solvency"] = st.sidebar.number_input("Solvency ratio %", 0.0, 1000.0, 0.0, 1.0) or None
    if profile.insurer_layer == "life":
        manual["persistency13"] = st.sidebar.number_input("13th-month persistency %", 0.0, 100.0, 0.0, 1.0) or None
    else:
        manual["combined"] = st.sidebar.number_input("Combined ratio %", 0.0, 300.0, 0.0, 0.5) or None
mcap = sc_meta.get("market_cap", np.nan)
if pd.isna(mcap) and info.get("marketCap"):
    mcap = info["marketCap"] / ds.CRORE

# Shareholding: NSE first, Screener's quarterly pattern as fallback
sc_shp = ds.normalise_screener_shareholding(sc_res["shareholding"]) if sc_res else None
sh_for_flags, sh_source = nse.get("shareholding"), "NSE"
if sh_for_flags is None and sc_shp is not None:
    sh_for_flags, sh_source = sc_shp, "Screener.in"

fin_flags, extras, lend_flags, lend_extras = [], {}, [], {}
if profile.financial_layer:
    base = fin if fin is not None else pd.DataFrame(index=ds.METRICS, columns=["FY0", "FY1"], dtype=float)
    fin_flags, extras = fx.financial_flags(base, info, profile, mcap)
    if profile.keep_only:   # broker: only the kept accounting tests are shown
        fin_flags = [f for f in fin_flags if f.key in profile.keep_only]
# Sector-specific module ("lend_flags" holds lender, broker or insurer checks)
special_name = ("Lender forensics" if lender else "Broker forensics" if profile.broker_layer
                else "Insurer forensics" if profile.insurer_layer else "")
if special_name:
    if sc_res is not None:
        if lender:
            lend_flags, lend_extras = fx.lender_flags(sc_res["fin"], sc_res["tables"], sc_res["sched"], kind)
        elif profile.broker_layer:
            lend_flags, lend_extras = fx.broker_flags(sc_res["tables"], sc_res["sched"])
        else:
            lend_flags, lend_extras = fx.insurer_flags(sc_res["tables"], sc_res["sched"], profile.insurer_layer, manual)
    else:
        st.warning(f"{special_name} need Screener.in data, which is unavailable right now.")

# Sector peers
peer_cmp, peer_raw = None, {}
if use_peers and sc_res is not None and sc_res.get("peers") and mode:
    with st.spinner(f"Scoring {len(sc_res['peers'])} sector peers…"):
        for name, path in sc_res["peers"]:
            pth = path.replace("consolidated/", "") if lender else path
            m = get_peer_metrics(pth, mode)
            if m:
                peer_raw[name] = m
    target_m = pr.metrics_from_tables(sc_res["tables"], mode)
    peer_cmp = pr.compare(target_m, peer_raw, mode)
    pr.apply_to_flags(lend_flags + fin_flags, peer_cmp)

own_flags = fx.ownership_flags(sh_for_flags, nse.get("pledges"), nse.get("insider"))
if sh_for_flags is not None:
    own_flags[0].detail += f" Source: {sh_source}."
ann_eval = nse.get("announcements") is not None or (nse_err.get("announcements") or "").startswith("no announcements")
hits = fx.scan_announcements(nse.get("announcements"))
ann_flag = fx.announcement_flag(hits, ann_eval)
all_flags = fin_flags + lend_flags + own_flags + [ann_flag]
comp = fx.composite(all_flags)

# --------------------------------------------------------------------------
# Header & score
# --------------------------------------------------------------------------
st.title(company)
meta = [f"**{symbol}**", info.get("sector"), info.get("industry")]
st.caption(" · ".join(m for m in meta if m))
yrs = f"{fin.columns[0]}–{fin.columns[-1]}" if fin is not None and len(fin.columns) else "none"
st.caption(f"Financials: {source} ({yrs}) · Data pulled {datetime.now():%d %b %Y %H:%M}"
           + (f" · [Screener page]({sc_meta['url']})" if sc_meta.get("url") else ""))
if fin_err and fin is fin_yf:
    st.error(f"Financial statements unavailable: {fin_err}")
if src_note:
    st.caption(f"ℹ️ {src_note}")
cls = sc_meta.get("classification") or {}
st.info(f"**Rule book: {profile.label}** ({kind_why}). {profile.note}"
        + (f" Compared with {len(peer_raw)} sector peers." if peer_raw else ""))

c1, c2, c3, c4 = st.columns(4)
score_txt = "—" if pd.isna(comp["score"]) else f"{comp['score']:.0f} / 100"
c1.metric("Red-flag score", score_txt)
c2.metric("Red flags", comp["red"])
c3.metric("Watch items", comp["amber"])
c4.metric("Model coverage", f"{comp['coverage'] * 100:.0f}%",
          help="Share of the weighted checks that had enough data to run.")
st.markdown(f"<span class='band' style='background:{BAND_COLOR[comp['band']]}'>{comp['band']}</span>",
            unsafe_allow_html=True)
if comp["coverage"] < 0.6:
    st.caption("Coverage is low, so read the score loosely — missing data isn't the same as a clean result.")

lay = comp["by_layer"]
lc = st.columns(len(lay))
for col, (name, d) in zip(lc, lay.items()):
    s = "—" if pd.isna(d["score"]) else f"{d['score']:.0f}"
    col.metric(name, s, help=f"{d['evaluated']}/{d['total']} checks ran · {d['red']} red · {d['amber']} watch")


def flag_card(f: fx.Flag):
    color = {2: "#c92a2a", 1: "#e8590c", 0: "#2b8a3e", None: "#868e96"}[f.severity]
    peer = ""
    if f.peer_note:
        was = f" (was {SEV_BADGE[f.original_severity]})" if f.original_severity is not None else ""
        peer = f"<div class='small'>👥 {f.peer_note}{was}</div>"
    st.markdown(
        f"<div class='flagcard' style='--c:{color}'><b>{SEV_BADGE[f.severity]} · {f.title}</b>"
        f" &nbsp;<code>{f.value}</code><div class='small'>{f.detail}</div>{peer}</div>",
        unsafe_allow_html=True)
    with st.expander("Why it matters / innocent explanations"):
        st.write(f"**Why it matters:** {f.why}")
        st.write(f"**Possible innocent explanations:** {f.benign}")


st.subheader("What stands out")
top = sorted([f for f in all_flags if f.severity in (1, 2)], key=lambda f: (-f.severity, -f.weight))
if top:
    for f in top:
        flag_card(f)
else:
    st.success("No red or watch flags among the checks that could run.")


def flags_table(flags):
    return pd.DataFrame([{"Status": SEV_BADGE[f.severity], "Check": f.title, "Value": f.value,
                          "Weight": f.weight, "Detail": f.detail, "Versus peers": f.peer_note}
                         for f in flags])


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
t1, t2, t6, t3, t4, t5 = st.tabs([special_name or "Financial forensics", "Ownership & pledges",
                                   "Peers", "Filings", "Data", "Method & limits"])

with t1:
    if lender:
        st.dataframe(flags_table(lend_flags), hide_index=True, width="stretch")
        npa = lend_extras.get("npa")
        if npa is not None and not npa.dropna(how="all").empty:
            g = go.Figure()
            for col in npa.columns:
                g.add_scatter(x=npa.index, y=npa[col], name=col, mode="lines+markers")
            g.update_layout(title="Asset quality by quarter (standalone, %)", height=320,
                            margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(g, width="stretch")
        st.caption("Screener doesn't split loans or provisions for lenders, so the loan book is approximated from "
                   "its 'Other Assets' grouping and provision coverage ignores technical write-offs. Check CRAR, "
                   "slippages and restructured book in the annual report or investor presentation.")
    elif profile.broker_layer or profile.insurer_layer:
        st.dataframe(flags_table(lend_flags + fin_flags), hide_index=True, width="stretch")
        series = {f.title: f.series for f in lend_flags if f.series is not None}
        for title, ser in series.items():
            ser = ser.dropna()
            if len(ser) >= 2:
                g = go.Figure()
                g.add_scatter(x=list(ser.index), y=ser.values, mode="lines+markers", name=title)
                g.update_layout(title=title, height=280, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(g, width="stretch")
        if profile.broker_layer:
            st.caption("Client liabilities are Screener's 'Other Liabilities' less minority interest; client money "
                       "parked as exchange margin or fixed deposits may sit outside 'cash', understating cover. "
                       "Check the segregation disclosures in the annual report.")
        else:
            st.caption("Screener shows premium income and total outgo but doesn't split claims, commission and "
                       "reserve changes. For solvency, combined ratio and persistency, use the IRDAI public "
                       "disclosures (L- and NL-forms) on the insurer's website and type the figures in the sidebar.")
    elif not profile.financial_layer:
        st.info(profile.note)
    else:
        st.dataframe(flags_table(fin_flags), hide_index=True, width="stretch")
    if fin is not None and profile.financial_layer and not profile.keep_only:
        f = fin
        yrs = list(f.columns)
        ebitda = fx._ebitda(f)

        g = go.Figure()
        g.add_bar(x=yrs, y=ebitda.values, name="EBITDA")
        g.add_bar(x=yrs, y=f.loc["net_income"].values, name="Net profit")
        g.add_bar(x=yrs, y=f.loc["cfo"].values, name="Operating cash flow")
        g.update_layout(title="Profit vs cash (₹ Cr)", barmode="group", height=360,
                        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(g, width="stretch")

        dso = fx._safe_div(f.loc["receivables"], f.loc["revenue"]) * 365
        dio = fx._safe_div(f.loc["inventory"], f.loc["cogs"].fillna(f.loc["revenue"])) * 365
        g = go.Figure()
        g.add_scatter(x=yrs, y=dso.values, name="Receivable days", mode="lines+markers")
        g.add_scatter(x=yrs, y=dio.values, name="Inventory days", mode="lines+markers")
        g.update_layout(title="Working-capital days", height=320, margin=dict(l=10, r=10, t=50, b=10),
                        legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(g, width="stretch")

        bm = extras.get("beneish")
        if bm is not None and not bm.empty:
            g = go.Figure()
            g.add_bar(x=bm.index, y=bm["M"], name="M-Score",
                      marker_color=["#c92a2a" if v > -1.78 else "#e8590c" if v > -2.22 else "#2b8a3e"
                                    for v in bm["M"].fillna(-9)])
            g.add_hline(y=-1.78, line_dash="dash", annotation_text="−1.78")
            g.add_hline(y=-2.22, line_dash="dot", annotation_text="−2.22")
            g.update_layout(title="Beneish M-Score by year", height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(g, width="stretch")
            with st.expander("Beneish components"):
                st.dataframe(bm.round(3), width="stretch")
                st.caption("DSRI receivables vs sales · GMI gross-margin decline · AQI asset quality · "
                           "SGI sales growth · DEPI depreciation slowdown · SGAI overhead · "
                           "LVGI leverage · TATA total accruals. Missing items are set to 1 (neutral).")

        mc = extras.get("montier")
        if mc is not None and not mc.empty:
            with st.expander("Montier C-Score tests (1 = warning)"):
                st.dataframe(mc, width="stretch")

        pf = extras.get("piotroski")
        if pf and pd.notna(pf[0]):
            with st.expander(f"Context: Piotroski F-Score {pf[0]:.0f}/{pf[1]} (quality, not a fraud test)"):
                st.write(pd.DataFrame({"Test": list(pf[2].keys()),
                                       "Pass": ["—" if v is None else ("✓" if v else "✗") for v in pf[2].values()]}))
                st.caption("Share-dilution test omitted; Yahoo data does not give consistent share counts.")

with t2:
    st.dataframe(flags_table(own_flags), hide_index=True, width="stretch")
    sh, pl = sh_for_flags, nse.get("pledges")
    if sh is not None:
        g = go.Figure()
        g.add_scatter(x=sh["date"], y=sh["promoter_pct"], name="Promoter %", mode="lines+markers")
        if sc_shp is not None:  # institutional holders from Screener's pattern
            g.add_scatter(x=sc_shp["date"], y=sc_shp["fii_pct"], name="FII %", mode="lines+markers")
            g.add_scatter(x=sc_shp["date"], y=sc_shp["dii_pct"], name="DII %", mode="lines+markers")
        if pl is not None:
            g.add_scatter(x=pl["date"], y=pl["pledged_pct_of_promoter"], name="Pledged % of promoter",
                          mode="lines+markers", yaxis="y2")
            g.update_layout(yaxis2=dict(overlaying="y", side="right", title="Pledged %"))
        g.update_layout(title="Promoter holding and pledges", height=340, margin=dict(l=10, r=10, t=50, b=10),
                        legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(g, width="stretch")
    ins = nse.get("insider")
    if ins is not None:
        with st.expander(f"Insider / promoter trades, last 12 months ({len(ins)})"):
            st.dataframe(ins, hide_index=True, width="stretch")
    for k in ("shareholding", "pledges", "insider"):
        if nse_err.get(k):
            st.caption(f"{k.title()}: {nse_err[k]}")
    raw = nse.get("_raw", {})
    for k, df in raw.items():
        if df is not None and (nse.get(k) is None):
            with st.expander(f"Raw NSE {k} data (fields not recognised)"):
                st.dataframe(df, width="stretch")
    if not fetch_nse or any(nse_err.get(k) for k in ("shareholding", "pledges")):
        st.info("NSE often refuses requests from cloud servers. If this layer is empty, try again later, "
                "or run the app on your own computer where NSE usually responds.")

with t6:
    if peer_cmp is None or peer_cmp.empty:
        st.info("No peer comparison: switched off, no peers listed on Screener, or not applicable to this company type.")
    else:
        view = peer_cmp.copy()
        view["Company"] = [pr.fmt_value(v, f) for v, f in zip(view["Company"], view["fmt"])]
        view["Peer median"] = [pr.fmt_value(v, f) for v, f in zip(view["Peer median"], view["fmt"])]
        view["Position"] = ["—" if pd.isna(w) else
                            ("🔴 bottom of group" if w >= 0.85 else "🟠 below median" if w > 0.5 else "🟢 median or better")
                            for w in view["worse_than"]]
        view["Worse than"] = ["—" if pd.isna(w) else f"{w:.0%} of peers" for w in view["worse_than"]]
        st.dataframe(view[["Metric", "Company", "Peer median", "Worse than", "Position", "Peers"]],
                     hide_index=True, width="stretch")
        st.caption("Company and peers are measured the same way, from each Screener page. A flag is downgraded "
                   "one step when the company is no worse than the typical peer (a sector norm), and a clear "
                   "core check is raised to watch when the company is worse than 85% of five or more peers.")
        with st.expander(f"Peer figures ({len(peer_raw)} companies)"):
            specs = pr.SPECS[mode]
            tbl = pd.DataFrame({n: {lab: pr.fmt_value(m.get(k), f) for k, lab, _, _, f in specs}
                                for n, m in peer_raw.items()})
            st.dataframe(tbl, width="stretch")
        st.caption("Peers are the companies Screener lists in the same NSE industry, largest first. "
                   "A peer group that shares the same problem can make a company look normal — "
                   "the original severity is shown on every adjusted flag.")

with t3:
    st.dataframe(flags_table([ann_flag]), hide_index=True, width="stretch")
    if hits is not None and not hits.empty:
        view = hits.copy()
        view["severity"] = view["severity"].map(SEV_BADGE)
        st.dataframe(view[["date", "severity", "category", "subject", "link"]], hide_index=True,
                     width="stretch",
                     column_config={"link": st.column_config.LinkColumn("Filing"),
                                    "date": st.column_config.DatetimeColumn("Date", format="DD MMM YYYY")})
        st.caption("Keyword matches only — open each filing before drawing any conclusion.")
    ann = nse.get("announcements")
    if ann is not None:
        with st.expander(f"All announcements in window ({len(ann)})"):
            st.dataframe(ann[["date", "subject", "link"]], hide_index=True, width="stretch",
                         column_config={"link": st.column_config.LinkColumn("Filing")})
    if nse_err.get("announcements"):
        st.caption(f"Announcements: {nse_err['announcements']}")

with t4:
    if fin is not None:
        st.caption(f"₹ crore · source: {source}")
        st.dataframe(fin.round(1), width="stretch")
        st.download_button("Download data (CSV)", fin.round(2).to_csv().encode(),
                           file_name=f"{symbol}_financials.csv", mime="text/csv")
    if sc_res is not None and fin_yf is not None:
        sc_fin = sc_res["fin"]
        common = [c for c in sc_fin.columns if c in fin_yf.columns][-3:]
        if common:
            st.markdown("**Source cross-check: Screener vs Yahoo** (₹ Cr)")
            rows = []
            for k, lab in [("revenue", "Revenue"), ("net_income", "Net profit"), ("cfo", "Operating cash flow"),
                           ("total_assets", "Total assets"), ("total_debt", "Borrowings")]:
                for c in common:
                    a, b = sc_fin.at[k, c], fin_yf.at[k, c]
                    gap = abs(a - b) / abs(b) if pd.notna(a) and pd.notna(b) and b != 0 else np.nan
                    rows.append({"Item": lab, "Year": c, "Screener": a, "Yahoo": b,
                                 "Gap": "—" if pd.isna(gap) else f"{gap:.1%}",
                                 "": "⚠️" if pd.notna(gap) and gap > 0.05 else ""})
            st.dataframe(pd.DataFrame(rows).round(0), hide_index=True, width="stretch")
            st.caption("Gaps above 5% are marked. Usually a definition difference (e.g. borrowings incl. leases, "
                       "consolidated vs standalone) — but worth knowing which figure the flags used.")
    st.download_button("Download flag report (CSV)", flags_table(all_flags).to_csv(index=False).encode(),
                       file_name=f"{symbol}_red_flags.csv", mime="text/csv")

with t5:
    st.markdown("""
**How the score works.** Each check is clear (0), watch (1) or red (2) and carries a weight.
The score is the weighted share of the maximum possible, using only checks that had data (coverage shows how many).
Bands: under 20 low · 20–35 some questions · 35–55 elevated · above 55 high.

**Default thresholds** (manufacturers; other company types adjust these — see the table below)
- Operating cash flow / EBITDA over up to 5 years: ≥70% clear, 50–70% watch, <50% red
- Operating cash flow / net profit: ≥80% clear, <50% red
- Receivable days: red if up >30% and >15 days; inventory days: red if up >30% and >20 days
- Beneish M-Score (8-variable): above −1.78 red, −2.22 to −1.78 watch; missing components set to 1
- Sloan accrual ratio (cash-flow form): within ±10% clear, above 25% red
- Montier C-Score: 3-year average of 4+ (with latest 4+) red; average 3+ or a single year at 4+ watch
- Return on cash: below 2% red, 2–4% watch, only when cash is ≥5% of assets
- Cash and debt both above 15% of assets watch, both above 25% red
- 3-year effective tax rate below 12% red, 12–18% watch
- Non-operating income above 40% of pre-tax profit red
- Altman Z″ below 1.1 red, 1.1–2.6 grey; manufacturers use the original Z (below 1.81 red, 1.81–2.99 grey) — distress, not fraud
- Banks / NBFCs: gross NPA above 6% (bank) / 5% (NBFC) or up >1.5 points in a year red; provision coverage below 50% red;
  loan growth above 40% / 50% a year red; bank loans outgrowing deposits by >15 points red; NBFC borrowings above 8× net worth red
- Promoter stake down >5 points in a year red; pledges above 30% of promoter shares red
- Filings: auditor resignation, forensic audit, fraud, search/ED, SEBI orders, defaults, insolvency,
  modified audit opinion or rating downgrade = red; officer resignations, auditor change, preferential
  issues, pledges, tax demands or penalties = watch

**Limits**
- Yahoo gives about four years of annual data for NSE stocks; the Screener export gives ten.
- Related-party transactions, contingent liabilities and auditor remarks live in the annual report, which this version does not read.
- Keyword matching can both miss and over-match; always open the filing.
- Models like Beneish were built on US companies and produce false positives, especially for fast growers.
- Banks and NBFCs get a lighter lender module (Screener doesn't split loans or provisions).
- Brokers get client-money, leverage and receivable checks; Screener's 'Other Liabilities' stands in for client liabilities, so client money parked as exchange margin or deposits can make cover look lower than it is.
- Insurers get balance-sheet and cash-flow checks; solvency, combined ratio and persistency come only from figures you type in from IRDAI disclosures, since claims aren't split out on Screener.
- Peer groups are Screener's industry lists; if a whole sector shares a problem, peers can make it look normal.
- No site visits, management checks or trade data — the parts of forensic work that can't be automated.
""")
    st.markdown("**How each company type changes the rules**")
    rows = []
    for k, p in sc.PROFILES.items():
        diffs = []
        for tk, tv in p.th.items():
            for kk, vv in tv.items():
                if sc.DEFAULT_TH[tk][kk] != vv:
                    diffs.append(f"{tk}.{kk} {sc.DEFAULT_TH[tk][kk]}→{vv}")
        rows.append({"Type": p.label,
                     "Threshold changes": "; ".join(diffs) or "—",
                     "Weight changes": "; ".join(f"{a}→{b}" for a, b in p.weights.items()) or "—",
                     "Not scored": "; ".join(p.skip) or ("all accounting tests" if not p.financial_layer else "—"),
                     "Why": p.note})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.divider()
st.caption("For research and education only. Not investment advice and not an allegation of wrongdoing "
           "against any company. Data from public sources may be incomplete or delayed. "
           "© Dr Shantanu Samanta")
