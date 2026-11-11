"""
Data layer for the Forensic Red-Flag Screener.

Every loader returns plain pandas objects and never raises to the UI:
failures come back as (None, "reason") so the app can show what is missing.

Standard financial frame ("fin"):
    index   = metric keys listed in METRICS
    columns = fiscal-year labels ("FY2022", "FY2023", ...) in ascending order
    values  = Rs crore (float), NaN where the source has no figure
"""
from __future__ import annotations

import difflib
import warnings
import io
import re
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

CRORE = 1e7
warnings.filterwarnings("ignore", message="Could not infer format")

METRICS = [
    "revenue", "cogs", "gross_profit", "sga", "depreciation", "ebit", "ebitda",
    "interest_expense", "interest_income", "other_income", "pbt", "tax", "net_income",
    "cfo", "cfi", "cff", "capex", "tax_paid",
    "receivables", "inventory", "other_current_assets", "current_assets",
    "current_liabilities", "total_assets", "total_liabilities", "equity",
    "retained_earnings", "net_ppe", "gross_ppe", "cwip", "intangibles", "cash",
    "securities", "lt_debt", "total_debt", "deposits",
]

# --------------------------------------------------------------------------
# Symbol universe and picker
# --------------------------------------------------------------------------
NSE_EQUITY_LIST_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
]

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def load_nse_equity_list(timeout: int = 15):
    """Return (DataFrame[symbol, name], None) or (None, reason)."""
    last_err = "unknown"
    for url in NSE_EQUITY_LIST_URLS:
        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout)
            if r.status_code != 200 or b"SYMBOL" not in r.content[:200]:
                last_err = f"HTTP {r.status_code}"
                continue
            df = pd.read_csv(io.BytesIO(r.content))
            df.columns = [c.strip().upper() for c in df.columns]
            out = pd.DataFrame({
                "symbol": df["SYMBOL"].astype(str).str.strip().str.upper(),
                "name": df["NAME OF COMPANY"].astype(str).str.strip(),
            })
            return out, None
        except Exception as e:  # network or parse failure
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err


def _norm(s: str) -> str:
    s = s.upper()
    s = re.sub(r"\b(LIMITED|LTD|THE|INDIA|CO|COMPANY|CORPORATION|CORP)\b", " ", s)
    return re.sub(r"[^A-Z0-9]", "", s)


def resolve_symbol(query: str, universe: pd.DataFrame | None, yf_module=None):
    """
    Picker rule shared with the Forward Matrix / Fair Value apps:
      - exact NSE symbol -> ("exact", SYMBOL, name)
      - otherwise        -> ("choices", [(SYMBOL, name), ...], None)
    """
    q = (query or "").strip().upper().replace(".NS", "").replace(".BO", "")
    if not q:
        return "choices", [], None

    if universe is not None and len(universe):
        hit = universe[universe["symbol"] == q]
        if len(hit):
            return "exact", q, hit.iloc[0]["name"]
        nq = _norm(q)
        u = universe.assign(_ns=universe["name"].map(_norm))
        contains = u[u["symbol"].str.contains(re.escape(q), regex=True) |
                     u["_ns"].str.contains(re.escape(nq), regex=True)] if nq else u.iloc[0:0]
        scored = []
        for _, row in u.iterrows():
            s = max(difflib.SequenceMatcher(None, nq, row["_ns"]).ratio(),
                    difflib.SequenceMatcher(None, q, row["symbol"]).ratio())
            if s >= 0.55:
                scored.append((s, row["symbol"], row["name"]))
        scored.sort(reverse=True)
        picks = [(r["symbol"], r["name"]) for _, r in contains.head(15).iterrows()]
        for _, sym, name in scored:
            if (sym, name) not in picks:
                picks.append((sym, name))
        return "choices", picks[:15], None

    # Fallback when NSE list is unreachable: Yahoo search limited to NSE listings
    if yf_module is not None:
        try:
            t = yf_module.Ticker(f"{q}.NS")
            h = t.history(period="5d")
            if h is not None and len(h):
                name = None
                try:
                    name = t.info.get("longName")
                except Exception:
                    pass
                return "exact", q, name or q
        except Exception:
            pass
        try:
            res = yf_module.Search(query, max_results=15).quotes
            picks = [(x["symbol"].replace(".NS", ""), x.get("longname") or x.get("shortname") or "")
                     for x in res if x.get("exchange") == "NSI" and x.get("symbol", "").endswith(".NS")]
            return "choices", picks, None
        except Exception:
            pass
    return "choices", [], None


# --------------------------------------------------------------------------
# yfinance financial statements -> standard frame
# --------------------------------------------------------------------------
YF_MAP = {
    # metric: [(statement, [candidate rows ...]), ...]  first non-empty wins
    "revenue": [("is", ["Total Revenue", "Operating Revenue"])],
    "cogs": [("is", ["Cost Of Revenue", "Reconciled Cost Of Revenue"])],
    "gross_profit": [("is", ["Gross Profit"])],
    "sga": [("is", ["Selling General And Administration"])],
    "depreciation": [("cf", ["Depreciation And Amortization", "Depreciation"]),
                     ("is", ["Reconciled Depreciation"])],
    "ebit": [("is", ["EBIT", "Operating Income"])],
    "ebitda": [("is", ["EBITDA", "Normalized EBITDA"])],
    "interest_expense": [("is", ["Interest Expense", "Interest Expense Non Operating"])],
    "interest_income": [("is", ["Interest Income Non Operating", "Interest Income"])],
    "pbt": [("is", ["Pretax Income"])],
    "tax": [("is", ["Tax Provision"])],
    "net_income": [("is", ["Net Income From Continuing Operation Net Minority Interest",
                           "Net Income Continuous Operations", "Net Income"])],
    "cfo": [("cf", ["Operating Cash Flow"])],
    "cfi": [("cf", ["Investing Cash Flow"])],
    "cff": [("cf", ["Financing Cash Flow"])],
    "capex": [("cf", ["Capital Expenditure", "Purchase Of PPE"])],
    "tax_paid": [("cf", ["Taxes Refund Paid"])],
    "receivables": [("bs", ["Accounts Receivable", "Receivables"])],
    "inventory": [("bs", ["Inventory"])],
    "other_current_assets": [("bs", ["Other Current Assets"])],
    "current_assets": [("bs", ["Current Assets"])],
    "current_liabilities": [("bs", ["Current Liabilities"])],
    "total_assets": [("bs", ["Total Assets"])],
    "total_liabilities": [("bs", ["Total Liabilities Net Minority Interest"])],
    "equity": [("bs", ["Stockholders Equity", "Common Stock Equity"])],
    "retained_earnings": [("bs", ["Retained Earnings"])],
    "net_ppe": [("bs", ["Net PPE"])],
    "gross_ppe": [("bs", ["Gross PPE"])],
    "cwip": [("bs", ["Construction In Progress"])],
    "intangibles": [("bs", ["Goodwill And Other Intangible Assets", "Other Intangible Assets"])],
    "cash": [("bs", ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])],
    "securities": [("bs", ["Investmentin Financial Assets", "Other Short Term Investments"])],
    "lt_debt": [("bs", ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])],
    "total_debt": [("bs", ["Total Debt"])],
}


def _fy_label(ts) -> str:
    # Indian FY ends in March (Mar-2026 -> FY2026); other year-ends use their calendar year
    return f"FY{pd.Timestamp(ts).year}"


def statements_to_frame(is_df, bs_df, cf_df) -> pd.DataFrame:
    """Map yfinance statements (rows=items, cols=dates, values in INR) to the standard frame."""
    stmts = {"is": is_df, "bs": bs_df, "cf": cf_df}
    dates = set()
    for d in stmts.values():
        if d is not None and not d.empty:
            dates.update(pd.to_datetime(d.columns))
    dates = sorted(dates)
    fin = pd.DataFrame(index=METRICS, columns=[_fy_label(d) for d in dates], dtype=float)
    col_of = {d: _fy_label(d) for d in dates}

    for metric, sources in YF_MAP.items():
        for st_key, rows in sources:
            d = stmts.get(st_key)
            if d is None or d.empty:
                continue
            got = False
            for row in rows:
                if row in d.index:
                    s = pd.to_numeric(d.loc[row], errors="coerce")
                    if s.notna().any():
                        for c, v in s.items():
                            lab = col_of.get(pd.Timestamp(c))
                            if lab is not None and pd.isna(fin.at[metric, lab]):
                                fin.at[metric, lab] = v / CRORE
                        got = True
                        break
            if got:
                break

    # Non-operating income = interest income + other non-operating items (net)
    if is_df is not None and not is_df.empty:
        other = pd.Series(0.0, index=fin.columns)
        found = False
        for row in ["Interest Income Non Operating", "Other Non Operating Income Expenses"]:
            if row in is_df.index:
                s = pd.to_numeric(is_df.loc[row], errors="coerce")
                for c, v in s.items():
                    lab = col_of.get(pd.Timestamp(c))
                    if lab is not None and pd.notna(v):
                        other[lab] += v / CRORE
                        found = True
        if found:
            fin.loc["other_income"] = other.where(other != 0.0, np.nan)
        elif fin.loc["interest_income"].notna().any():
            fin.loc["other_income"] = fin.loc["interest_income"]

    # tax_paid comes signed negative in cash flow; store as a positive outflow
    fin.loc["tax_paid"] = -fin.loc["tax_paid"]
    fin.loc["capex"] = -fin.loc["capex"].abs()

    # Drop columns that are essentially empty (yfinance often pads an old year)
    core = ["revenue", "total_assets", "cfo", "net_income"]
    keep = [c for c in fin.columns if fin.loc[core, c].notna().sum() >= 2]
    fin = fin[keep]
    # merge duplicate FY labels (keep latest non-null)
    fin = fin.T.groupby(level=0).last().T
    return fin[sorted(fin.columns)]


def load_yf_financials(symbol: str, yf_module):
    """Return (fin, info, None) or (None, info_or_{}, reason)."""
    info = {}
    try:
        t = yf_module.Ticker(f"{symbol}.NS")
        try:
            info = t.info or {}
        except Exception:
            info = {}
        is_df, bs_df, cf_df = t.income_stmt, t.balance_sheet, t.cashflow
        if (is_df is None or is_df.empty) and (bs_df is None or bs_df.empty):
            return None, info, "Yahoo returned no annual statements"
        fin = statements_to_frame(is_df, bs_df, cf_df)
        if fin.shape[1] < 2:
            return None, info, "fewer than two usable years from Yahoo"
        return fin, info, None
    except Exception as e:
        return None, info, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Screener.in "Export to Excel" (Data Sheet) -> standard frame
# --------------------------------------------------------------------------
SCREENER_ROWS = {
    "PL": {
        "sales": "revenue", "raw material cost": "_rm", "change in inventory": "_chinv",
        "power and fuel": "_power", "other mfr. exp": "_othmfr", "employee cost": "_emp",
        "selling and admin": "sga", "other expenses": "_othexp", "other income": "other_income",
        "depreciation": "depreciation", "interest": "interest_expense",
        "profit before tax": "pbt", "tax": "tax", "net profit": "net_income",
    },
    "BS": {
        "equity share capital": "_eqcap", "reserves": "_reserves", "borrowings": "total_debt",
        "other liabilities": "_othliab", "total": "total_assets", "net block": "net_ppe",
        "capital work in progress": "cwip", "investments": "securities", "other assets": "_othassets",
        "receivables": "receivables", "inventory": "inventory", "cash & bank": "cash",
    },
    "CF": {
        "cash from operating activity": "cfo", "cash from investing activity": "cfi",
        "cash from financing activity": "cff",
    },
}


def parse_screener_export(file_bytes: bytes):
    """
    Parse the 'Data Sheet' of a Screener.in Excel export (figures in Rs crore).
    Returns (fin, notes) or (None, reason). Layout is matched by row labels,
    so small template changes are tolerated; unknown rows are ignored.
    """
    try:
        import openpyxl  # noqa: F401
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet = next((s for s in xl.sheet_names if s.strip().lower() == "data sheet"), None)
        if sheet is None:
            return None, "no 'Data Sheet' tab found (is this a Screener.in export?)"
        raw = xl.parse(sheet, header=None)
    except Exception as e:
        return None, f"could not read workbook: {type(e).__name__}: {e}"

    section, dates = None, None
    data: dict[str, dict] = {}
    for _, row in raw.iterrows():
        label = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        low = label.lower().rstrip(":")
        if low.startswith("profit & loss"):
            section, dates = "PL", None
            continue
        if low.startswith("quarters"):
            section, dates = "Q", None
            continue
        if low.startswith("balance sheet"):
            section, dates = "BS", None
            continue
        if low.startswith("cash flow"):
            section, dates = "CF", None
            continue
        if low.startswith("price") or low.startswith("derived"):
            section, dates = None, None
            continue
        if low == "report date" and section:
            dates = []
            for v in row.iloc[1:]:
                try:
                    dates.append(pd.Timestamp(v) if pd.notna(v) else None)
                except Exception:
                    dates.append(None)
            continue
        if section in SCREENER_ROWS and dates:
            key = SCREENER_ROWS[section].get(low)
            if key is None:
                continue
            for d, v in zip(dates, row.iloc[1:]):
                if d is None or pd.isna(v):
                    continue
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue
                data.setdefault(key, {})[f"FY{d.year}"] = val

    if "revenue" not in data or "total_assets" not in data:
        return None, "Data Sheet found but Sales / Balance-sheet Total rows were not recognised"

    cols = sorted({c for v in data.values() for c in v})
    fin = pd.DataFrame(index=METRICS, columns=cols, dtype=float)
    for k, series in data.items():
        if k in fin.index:
            for c, v in series.items():
                fin.at[k, c] = v

    def g(k):
        return pd.Series(data.get(k, {}), dtype=float).reindex(cols)

    cogs = g("_rm").fillna(0) + g("_power").fillna(0) + g("_othmfr").fillna(0) - g("_chinv").fillna(0)
    if g("_rm").notna().any():
        fin.loc["cogs"] = cogs.where(cogs != 0, np.nan)
        fin.loc["gross_profit"] = fin.loc["revenue"] - fin.loc["cogs"]
    eq = g("_eqcap") + g("_reserves")
    fin.loc["equity"] = eq
    fin.loc["total_liabilities"] = fin.loc["total_assets"] - eq
    fin.loc["ebit"] = fin.loc["pbt"] + fin.loc["interest_expense"].fillna(0)
    fin.loc["ebitda"] = fin.loc["ebit"] + fin.loc["depreciation"].fillna(0)
    notes = ("Screener export: current assets/liabilities, interest income and cash taxes are not "
             "split out, so Altman Z'', parts of Beneish and the cash-tax check fall back or are skipped.")
    keep = [c for c in fin.columns if fin.loc[["revenue", "total_assets"], c].notna().sum() >= 1]
    return fin[keep], notes


# --------------------------------------------------------------------------
# Screener.in company page (live, no login) -> standard frame + shareholding
# --------------------------------------------------------------------------
SCREENER_BASE = "https://www.screener.in"


def _num(text) -> float:
    """'1,234' -> 1234.0, '22%' -> 22.0, '' / '-' -> NaN"""
    if text is None:
        return np.nan
    t = str(text).replace(",", "").replace("%", "").replace("₹", "").replace("Cr", "").strip()
    if t in ("", "-", "--"):
        return np.nan
    try:
        return float(t)
    except ValueError:
        return np.nan


def _clean_label(text: str) -> str:
    return re.sub(r"\s*\+\s*$", "", text.replace("\xa0", " ")).strip()


def _parse_html_table(table) -> pd.DataFrame:
    """Screener table -> DataFrame (rows = labels, cols = period strings like 'Mar 2025')."""
    trs = table.find_all("tr")
    if not trs:
        return pd.DataFrame()
    header = [c.get_text(strip=True) for c in trs[0].find_all(["th", "td"])][1:]
    data = {}
    for tr in trs[1:]:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        label = _clean_label(cells[0])
        if not label or label in data:
            continue
        data[label] = [_num(x) for x in cells[1:len(header) + 1]]
    return pd.DataFrame(data, index=header[:max((len(v) for v in data.values()), default=0)]).T


def _period_to_fy(p: str):
    try:
        ts = pd.to_datetime(p, format="%b %Y")
        return f"FY{ts.year}", ts.month
    except Exception:
        return None, None


class ScreenerClient:
    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update(BROWSER_HEADERS)

    def _get(self, url: str, **kw):
        """GET with backoff when Screener rate-limits (HTTP 429)."""
        r = None
        for wait in (0, 3, 8, 15):
            if wait:
                time.sleep(wait)
            r = self.s.get(url, timeout=self.timeout, **kw)
            if r.status_code != 429:
                return r
        return r

    def search(self, query: str):
        """Screener company search -> first matching company path, or None."""
        try:
            r = self._get(f"{SCREENER_BASE}/api/company/search/?q={requests.utils.quote(query)}&v=3&fts=1")
            for x in r.json():
                if str(x.get("url", "")).startswith("/company/"):
                    return x["url"]
        except Exception:
            pass
        return None

    def _page(self, symbol: str, consolidated: bool, path: str | None = None):
        from bs4 import BeautifulSoup
        if path:
            base = path.replace("consolidated/", "")
            url = f"{SCREENER_BASE}{base}{'consolidated/' if consolidated else ''}"
        else:
            url = f"{SCREENER_BASE}/company/{symbol}/{'consolidated/' if consolidated else ''}"
        r = self._get(url)
        if r.status_code == 404:
            return None, url, "symbol not found on Screener"
        if r.status_code != 200:
            return None, url, f"HTTP {r.status_code}"
        return BeautifulSoup(r.text, "html.parser"), url, None

    def _schedule(self, company_id: str, parent: str, section: str, consolidated: bool, referer: str):
        url = (f"{SCREENER_BASE}/api/company/{company_id}/schedules/"
               f"?parent={requests.utils.quote(parent)}&section={section}"
               f"&consolidated={'true' if consolidated else ''}")
        try:
            r = self._get(url, headers={"X-Requested-With": "XMLHttpRequest", "Referer": referer})
            js = r.json() if r.status_code == 200 else {}
        except Exception:
            js = {}
        out = {}
        for item, vals in (js or {}).items():
            if isinstance(vals, dict):
                out[item] = {k: _num(v) for k, v in vals.items()
                             if isinstance(v, (str, int, float)) and re.match(r"^[A-Z][a-z]{2} \d{4}$", str(k))}
        time.sleep(0.25)  # be polite
        return out

    @staticmethod
    def tables_from_soup(soup) -> dict:
        tables = {}
        for sec in ("quarters", "profit-loss", "balance-sheet", "cash-flow", "ratios"):
            el = soup.find("section", id=sec)
            t = el.find("table") if el else None
            tables[sec] = _parse_html_table(t) if t is not None else pd.DataFrame()
        return tables

    @staticmethod
    def classification(soup) -> dict:
        """NSE 4-level classification shown in Screener's peer section."""
        out = {}
        pe = soup.find("section", id="peers")
        for a in (pe.find_all("a") if pe else []):
            t = (a.get("title") or "").strip()
            if t in ("Broad Sector", "Sector", "Broad Industry", "Industry"):
                out[t] = a.get_text(strip=True)
        return out

    @staticmethod
    def market_cap(soup) -> float:
        for li in soup.select("#top-ratios li"):
            txt = " ".join(li.get_text(" ", strip=True).split())
            if txt.lower().startswith("market cap"):
                return _num(re.sub(r"[^0-9.,]", "", txt.split("₹")[-1]))
        return np.nan

    def peers(self, warehouse_id: str, referer: str, limit: int = 8):
        """-> list of (name, path) from Screener's peer table, excluding the company itself."""
        from bs4 import BeautifulSoup
        try:
            r = self._get(f"{SCREENER_BASE}/api/company/{warehouse_id}/peers/",
                          headers={"X-Requested-With": "XMLHttpRequest", "Referer": referer})
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception:
            return []
        out = []
        for a in soup.select("tr a[href^='/company/']"):
            path = a["href"]
            if path.rstrip("/") and path not in referer:
                out.append((a.get_text(strip=True), path))
        return out[:limit]

    def load_light(self, path: str):
        """Main page tables only (no schedule calls) — used for peers. -> (tables, name) or (None, reason)."""
        from bs4 import BeautifulSoup
        try:
            r = self._get(f"{SCREENER_BASE}{path}")
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}"
            soup = BeautifulSoup(r.text, "html.parser")
            h1 = soup.find("h1")
            return self.tables_from_soup(soup), (h1.get_text(strip=True) if h1 else path)
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
        finally:
            time.sleep(0.35)

    def load(self, symbol: str, prefer_consolidated: bool = True, name: str | None = None):
        """
        Returns dict(fin, shareholding, meta, tables, sched, peers) or (None, reason).
        Falls back from the consolidated to the standalone view when consolidated is empty.
        """
        tried = [True, False] if prefer_consolidated else [False, True]
        last_err = "no data"
        path = None
        for cons in tried:
            try:
                soup, url, err = self._page(symbol, cons, path)
            except Exception as e:
                soup, url, err = None, None, f"{type(e).__name__}: {e}"
            if soup is None and err == "symbol not found on Screener" and path is None:
                # Screener sometimes lists a company under a different code: search by symbol, then name
                path = self.search(symbol) or (self.search(name) if name else None)
                if path:
                    try:
                        soup, url, err = self._page(symbol, cons, path)
                    except Exception as e:
                        soup, url, err = None, None, f"{type(e).__name__}: {e}"
            if soup is None:
                last_err = err
                if err == "symbol not found on Screener":
                    break
                continue
            tables = self.tables_from_soup(soup)
            pl = tables["profit-loss"]
            rev_lab = "Sales" if "Sales" in pl.index else "Revenue" if "Revenue" in pl.index else None
            if rev_lab is None or pl.loc[rev_lab].notna().sum() < 2:
                last_err = f"{'consolidated' if cons else 'standalone'} view has no annual figures"
                continue
            cid_el = soup.find(attrs={"data-company-id": True})
            company_id = cid_el["data-company-id"] if cid_el else None
            wid_el = soup.find(attrs={"data-warehouse-id": True})
            warehouse_id = wid_el["data-warehouse-id"] if wid_el else None
            sched = {}
            if company_id:
                for parent, section in [("Other Assets", "balance-sheet"), ("Fixed Assets", "balance-sheet"),
                                        ("Borrowings", "balance-sheet"), ("Other Liabilities", "balance-sheet"),
                                        ("Expenses", "profit-loss"), ("Other Income", "profit-loss"),
                                        ("Cash from Operating Activity", "cash-flow")]:
                    sched[parent] = self._schedule(company_id, parent, section, cons, url)
            fin = self._to_frame(tables, sched)
            shp = None
            sh_el = soup.find(id="quarterly-shp")
            if sh_el is not None and sh_el.find("table") is not None:
                shp = _parse_html_table(sh_el.find("table"))
            name_el = soup.find("h1")
            meta = {"name": name_el.get_text(strip=True) if name_el else symbol,
                    "view": "Consolidated" if cons else "Standalone", "url": url,
                    "path": url.replace(SCREENER_BASE, ""),
                    "classification": self.classification(soup),
                    "market_cap": self.market_cap(soup),
                    # bank / NBFC layouts use 'Revenue' and 'Financing Profit' instead of 'Sales'
                    "is_financial": "Financing Profit" in pl.index or rev_lab == "Revenue"}
            if meta["is_financial"]:
                # NPA ratios are reported on the standalone (regulated) entity; consolidated pages often omit them
                st_path = (path or f"/company/{symbol}/").replace("consolidated/", "")
                st_tables, _ = self.load_light(st_path) if cons else (tables, None)
                if st_tables is not None:
                    tables["npa_quarters"] = st_tables.get("quarters", pd.DataFrame())
            peer_list = self.peers(warehouse_id, url) if warehouse_id else []
            return {"fin": fin, "shareholding": shp, "meta": meta, "tables": tables,
                    "sched": sched, "peers": peer_list}, None
        return None, last_err

    @staticmethod
    def _to_frame(tables: dict, sched: dict) -> pd.DataFrame:
        pl, bs, cf = tables["profit-loss"], tables["balance-sheet"], tables["cash-flow"]
        periods = [p for p in pl.columns if _period_to_fy(p)[0]]
        months = pd.Series([_period_to_fy(p)[1] for p in periods])
        ye_month = int(months.mode().iloc[0]) if len(months) else 3

        def annual(df):
            cols = {}
            for p in df.columns:
                fy, m = _period_to_fy(p)
                if fy and m == ye_month:   # drops TTM and interim (e.g. Sep) balance sheets
                    cols[p] = fy
            return df[list(cols)].rename(columns=cols) if cols else pd.DataFrame()

        pl, bs, cf = annual(pl), annual(bs), annual(cf)
        cols = sorted(set(pl.columns) | set(bs.columns) | set(cf.columns))
        fin = pd.DataFrame(index=METRICS, columns=cols, dtype=float)

        def put(metric, df, *labels):
            for lab in labels:
                if df is not None and not df.empty and lab in df.index:
                    s = df.loc[lab]
                    for c in s.index:
                        if c in fin.columns:
                            fin.at[metric, c] = s[c]
                    return True
            return False

        def sch(parent, item):
            d = sched.get(parent, {}).get(item)
            if not d:
                # aligned all-missing series (an unaligned empty Series made sums like
                # Other Liabilities + Short-term borrowings come out missing for debt-free companies)
                return pd.Series(np.nan, index=cols, dtype=float)
            s = pd.Series({_period_to_fy(k)[0]: v for k, v in d.items()
                           if _period_to_fy(k)[1] == ye_month}, dtype=float)
            return s.reindex(cols)

        put("revenue", pl, "Sales", "Revenue")
        put("ebitda", pl, "Operating Profit", "Financing Profit")
        put("other_income", pl, "Other Income")
        put("interest_expense", pl, "Interest")
        put("depreciation", pl, "Depreciation")
        put("pbt", pl, "Profit before tax")
        put("net_income", pl, "Net Profit")
        if "Tax %" in pl.index:
            fin.loc["tax"] = fin.loc["pbt"] * pl.loc["Tax %"].reindex(cols) / 100
        fin.loc["ebit"] = fin.loc["pbt"] + fin.loc["interest_expense"].fillna(0)

        mat = sch("Expenses", "Material Cost %")
        if mat.notna().any():
            fin.loc["cogs"] = fin.loc["revenue"] * mat / 100
            fin.loc["gross_profit"] = fin.loc["revenue"] - fin.loc["cogs"]
        oin = sch("Other Income", "Other income normal")
        if oin.notna().any():
            fin.loc["other_income"] = oin   # excludes exceptional items

        eq = pd.Series(0.0, index=cols)
        has_eq = False
        for lab in ("Equity Capital", "Reserves"):
            if lab in bs.index:
                eq = eq + bs.loc[lab].reindex(cols).fillna(0)
                has_eq = True
        if has_eq:
            fin.loc["equity"] = eq.where(eq != 0, np.nan)
        if "Reserves" in bs.index:
            fin.loc["retained_earnings"] = bs.loc["Reserves"].reindex(cols)  # proxy
        put("total_debt", bs, "Borrowings", "Borrowing")
        put("deposits", bs, "Deposits")
        put("total_assets", bs, "Total Assets")
        put("net_ppe", bs, "Fixed Assets")
        put("cwip", bs, "CWIP")
        put("securities", bs, "Investments")
        fin.loc["total_liabilities"] = fin.loc["total_assets"] - fin.loc["equity"]
        if "Other Assets" in bs.index:
            fin.loc["current_assets"] = bs.loc["Other Assets"].reindex(cols)  # proxy

        fin.loc["receivables"] = sch("Other Assets", "Trade receivables")
        fin.loc["inventory"] = sch("Other Assets", "Inventories")
        fin.loc["cash"] = sch("Other Assets", "Cash Equivalents")
        oca = sch("Other Assets", "Loans n Advances").fillna(0) + sch("Other Assets", "Other asset items").fillna(0)
        fin.loc["other_current_assets"] = oca.where(oca != 0, np.nan)
        fin.loc["gross_ppe"] = sch("Fixed Assets", "Gross Block")
        fin.loc["intangibles"] = sch("Fixed Assets", "Intangible Assets")
        fin.loc["lt_debt"] = sch("Borrowings", "Long term Borrowings")
        ol = bs.loc["Other Liabilities"].reindex(cols) if "Other Liabilities" in bs.index else pd.Series(np.nan, index=cols)
        stb = sch("Borrowings", "Short term Borrowings")
        fin.loc["current_liabilities"] = ol + stb.fillna(0)  # proxy

        put("cfo", cf, "Cash from Operating Activity")
        put("cfi", cf, "Cash from Investing Activity")
        put("cff", cf, "Cash from Financing Activity")
        if "Free Cash Flow" in cf.index:
            fin.loc["capex"] = cf.loc["Free Cash Flow"].reindex(cols) - fin.loc["cfo"]
        dt = sch("Cash from Operating Activity", "Direct taxes")
        if dt.notna().any():
            fin.loc["tax_paid"] = -dt
        fin = fin.drop(index=[i for i in fin.index if i.startswith("_")], errors="ignore")
        keep = [c for c in fin.columns if fin.loc[["revenue", "total_assets"], c].notna().sum() >= 1]
        return fin[keep]


def normalise_screener_shareholding(shp: pd.DataFrame | None):
    """Screener quarterly shareholding -> DataFrame[date, promoter_pct, fii_pct, dii_pct, public_pct, holders]."""
    if shp is None or shp.empty:
        return None
    rows = {"Promoters": "promoter_pct", "FIIs": "fii_pct", "DIIs": "dii_pct",
            "Public": "public_pct", "No. of Shareholders": "holders"}
    out = pd.DataFrame({"date": [pd.to_datetime(c, format="%b %Y", errors="coerce") + pd.offsets.MonthEnd(0)
                                 for c in shp.columns]})
    for lab, key in rows.items():
        out[key] = shp.loc[lab].values if lab in shp.index else np.nan
    if "promoter_pct" not in out or out["promoter_pct"].isna().all():
        out["promoter_pct"] = 0.0  # widely held company with no promoter group
    out = out.dropna(subset=["date"]).sort_values("date")
    return out if len(out) else None


SCREENER_NOTE = ("Screener page: current assets/liabilities and retained earnings are approximated from "
                 "Screener's groupings, and interest income is not split from other income.")


# --------------------------------------------------------------------------
# NSE corporate filings (cookie-primed session)
# --------------------------------------------------------------------------
class NSEClient:
    BASE = "https://www.nseindia.com"

    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update(BROWSER_HEADERS)
        self.primed = False
        self.last_error = None

    def _prime(self, symbol: str):
        if self.primed:
            return
        try:
            self.s.get(self.BASE, timeout=self.timeout)
            self.s.get(f"{self.BASE}/get-quotes/equity?symbol={symbol}", timeout=self.timeout)
            self.primed = True
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"

    def get_json(self, path: str, symbol: str, retries: int = 2):
        self._prime(symbol)
        url = f"{self.BASE}{path}"
        for attempt in range(retries + 1):
            try:
                r = self.s.get(url, timeout=self.timeout,
                               headers={"Referer": f"{self.BASE}/get-quotes/equity?symbol={symbol}"})
                if r.status_code == 200 and r.text.strip()[:1] in "[{":
                    return r.json(), None
                self.last_error = f"HTTP {r.status_code}"
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
            self.primed = False
            time.sleep(1.0 + attempt)
            self._prime(symbol)
        return None, self.last_error

    @staticmethod
    def _rows(js):
        if isinstance(js, list):
            return js
        if isinstance(js, dict):
            for k in ("data", "Data", "rows"):
                if isinstance(js.get(k), list):
                    return js[k]
        return []

    def shareholding(self, symbol: str):
        js, err = self.get_json(f"/api/corporate-share-holdings-master?index=equities&symbol={symbol}", symbol)
        if js is None:
            return None, err
        df = pd.DataFrame(self._rows(js))
        return (df if len(df) else None), (None if len(df) else "empty response")

    def pledges(self, symbol: str):
        js, err = self.get_json(f"/api/corporate-pledgedata?index=equities&symbol={symbol}", symbol)
        if js is None:
            return None, err
        df = pd.DataFrame(self._rows(js))
        return (df if len(df) else None), (None if len(df) else "no pledge filings returned")

    def announcements(self, symbol: str, months: int = 24):
        to_d = date.today()
        from_d = to_d - timedelta(days=int(months * 30.5))
        path = (f"/api/corporate-announcements?index=equities&symbol={symbol}"
                f"&from_date={from_d:%d-%m-%Y}&to_date={to_d:%d-%m-%Y}")
        js, err = self.get_json(path, symbol)
        if js is None:
            return None, err
        df = pd.DataFrame(self._rows(js))
        return (df if len(df) else None), (None if len(df) else "no announcements in window")

    def insider_trades(self, symbol: str, months: int = 12):
        to_d = date.today()
        from_d = to_d - timedelta(days=int(months * 30.5))
        path = (f"/api/corporates-pit?index=equities&symbol={symbol}"
                f"&from_date={from_d:%d-%m-%Y}&to_date={to_d:%d-%m-%Y}")
        js, err = self.get_json(path, symbol)
        if js is None:
            return None, err
        df = pd.DataFrame(self._rows(js))
        return (df if len(df) else None), (None if len(df) else "no insider filings in window")


# --------------------------------------------------------------------------
# Normalisers for NSE payloads (field names are matched loosely on purpose)
# --------------------------------------------------------------------------
def _to_dt(series: pd.Series) -> pd.Series:
    """Parse NSE date strings, trying the known layouts explicitly before falling back."""
    s = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        miss = out.isna()
        if not miss.any():
            break
        out[miss] = pd.to_datetime(s[miss].str.title() if "%b" in fmt else s[miss], format=fmt, errors="coerce")
    miss = out.isna()
    if miss.any():
        out[miss] = pd.to_datetime(s[miss], errors="coerce", dayfirst=True)
    return out


def _find_col(df: pd.DataFrame, *needles, exclude=()):
    for c in df.columns:
        lc = str(c).lower()
        if all(n in lc for n in needles) and not any(x in lc for x in exclude):
            return c
    return None


def normalise_shareholding(df: pd.DataFrame | None):
    """-> DataFrame[date, promoter_pct, public_pct] ascending, or None."""
    if df is None or df.empty:
        return None
    dcol = _find_col(df, "date", exclude=("submission", "broadcast")) or _find_col(df, "date")
    pcol = _find_col(df, "pr_and_prgrp") or _find_col(df, "promoter")
    ucol = _find_col(df, "public_val") or _find_col(df, "public", exclude=("notes",))
    if dcol is None or pcol is None:
        return None
    out = pd.DataFrame({
        "date": _to_dt(df[dcol]),
        "promoter_pct": pd.to_numeric(df[pcol], errors="coerce"),
        "public_pct": pd.to_numeric(df[ucol], errors="coerce") if ucol else np.nan,
    }).dropna(subset=["date", "promoter_pct"])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out if len(out) else None


def normalise_pledges(df: pd.DataFrame | None):
    """-> DataFrame[date, pledged_pct_of_promoter, pledged_pct_of_total] ascending, or None."""
    if df is None or df.empty:
        return None
    dcol = _find_col(df, "shp") or _find_col(df, "date")
    ppc = _find_col(df, "percpromotershares") or _find_col(df, "perc", "promoter", "share", exclude=("holding",))
    ptc = _find_col(df, "perctotshares") or _find_col(df, "perc", "tot", "share")
    if dcol is None or ppc is None:
        return None
    out = pd.DataFrame({
        "date": _to_dt(df[dcol]),
        "pledged_pct_of_promoter": pd.to_numeric(df[ppc], errors="coerce"),
        "pledged_pct_of_total": pd.to_numeric(df[ptc], errors="coerce") if ptc else np.nan,
    }).dropna(subset=["date", "pledged_pct_of_promoter"])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out if len(out) else None


def normalise_announcements(df: pd.DataFrame | None):
    """-> DataFrame[date, subject, text, link] newest first, or None."""
    if df is None or df.empty:
        return None
    dcol = (_find_col(df, "sort_date") or _find_col(df, "an_dt") or
            _find_col(df, "date"))
    scol = _find_col(df, "desc") or _find_col(df, "subject")
    tcol = _find_col(df, "attchmnttext") or _find_col(df, "text")
    lcol = _find_col(df, "attchmntfile") or _find_col(df, "file")
    when = _to_dt(df[dcol]) if dcol else pd.Series(pd.NaT, index=df.index)
    if "an_dt" in df.columns:
        when = when.fillna(_to_dt(df["an_dt"]))
    out = pd.DataFrame({
        "date": when,
        "subject": df[scol].astype(str) if scol else "",
        "text": df[tcol].astype(str) if tcol else "",
        "link": df[lcol].astype(str) if lcol else "",
    })
    return out.sort_values("date", ascending=False).reset_index(drop=True)


def normalise_insider(df: pd.DataFrame | None):
    """-> DataFrame[date, person, category, txn_type, shares] or None."""
    if df is None or df.empty:
        return None
    dcol = _find_col(df, "date") or _find_col(df, "intim")
    out = pd.DataFrame({
        "date": _to_dt(df[dcol]) if dcol else pd.NaT,
        "person": df[_find_col(df, "acqname")].astype(str) if _find_col(df, "acqname") else "",
        "category": df[_find_col(df, "personcategory")].astype(str) if _find_col(df, "personcategory") else "",
        "txn_type": df[_find_col(df, "transactiontype")].astype(str) if _find_col(df, "transactiontype") else "",
        "shares": pd.to_numeric(df[_find_col(df, "secacq")], errors="coerce") if _find_col(df, "secacq") else np.nan,
    })
    return out.sort_values("date", ascending=False).reset_index(drop=True)
