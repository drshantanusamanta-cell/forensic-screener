# Forensic Red-Flag Screener (NSE)

Screens one NSE-listed company for the accounting, ownership and filing patterns that forensic short-sellers start from. It flags things worth asking about. It does not detect fraud, and a clean result does not prove the accounts are sound.

© Dr Shantanu Samanta

## What it checks

**Financial forensics** (Screener.in live by default, about 12 years; Yahoo Finance as fallback; or an uploaded Screener export)
- Operating cash flow vs EBITDA and vs net profit, over up to 5 years
- Profitable every year but free cash flow negative
- Receivable days and inventory days trends
- Beneish M-Score (8-variable), Sloan accrual ratio, Montier C-Score
- Return earned on reported cash (the "cash that earns nothing" test)
- High cash and high debt together
- Effective tax rate and cash tax paid vs book tax
- Dependence on non-operating income
- Capital work-in-progress build-up, intangibles outgrowing revenue, debt outgrowing revenue
- Altman Z″ (distress), plus Piotroski F as unscored context

**Ownership & pledges** (NSE, with Screener's quarterly shareholding pattern as fallback)
- Promoter holding change over about a year
- Promoter shares pledged
- Promoter-group buying, selling and pledge creation in the last 12 months

**Filings & events** (NSE announcements)
- Keyword scan for auditor resignation, forensic audit, fraud, search/ED, SEBI orders, defaults, insolvency, modified audit opinions, rating downgrades (red), and officer or independent-director resignations, statutory auditor changes, preferential issues/warrants, pledges, tax demands and penalties (watch). Secretarial, cost and internal auditor filings are ignored as routine.

**Company types** (from NSE's industry classification on Screener, with Yahoo as fallback; can be overridden in the sidebar)

| Type | What changes |
|---|---|
| Manufacturer | Default rules; original Altman Z using market value of equity |
| IT / software services | Stricter cash conversion and debt limits; receivables weighted up; inventory and CWIP tests dropped |
| Retail / jewellery | Looser cash-conversion and inventory thresholds; receivables weighted up |
| Construction / EPC / real estate | Looser cash, receivable and inventory thresholds; higher debt tolerance |
| Power / telecom / infra operator | Higher CWIP and debt tolerance; free-cash-flow test down-weighted |
| Pharma / biotech | Intangibles (capitalised R&D) and receivables weighted up |
| Services | Default rules; inventory test down-weighted |
| Holding company | Tax, profit cash-conversion, accrual, other-income, cash-yield and distress tests dropped |
| Stockbroker | Industrial cash tests replaced by broker checks: client-money cover (cash ÷ client liabilities — the Karvy pattern), borrowings ÷ net worth, debt-funded growth, client receivables; tax, intangibles and receivable trend still scored |
| Exchange / clearing | Cash-conversion, accrual and cash-yield tests dropped (clearing-member money distorts them); treasury income threshold looser |
| AMC / registrar / depository / ratings | Asset-light fee rules: stricter cash conversion and debt limits; treasury-income threshold looser; Sloan, cash-yield, inventory, CWIP and Altman dropped |
| Bank / NBFC | Accounting tests replaced by a lender module: gross NPA level and trend, provision coverage, loan-book growth, loan vs deposit growth (banks) or leverage (NBFCs), equity cushion, ROA trend |
| Life insurer | Insurer checks: investments ÷ policyholder liabilities, net worth ÷ liabilities, operating cash flow, reliance on investment income, margin trend; optional solvency ratio and 13th-month persistency typed in from IRDAI disclosures |
| General / health insurer | Same insurer checks plus tax; optional solvency ratio and combined ratio typed in from IRDAI disclosures |

**Sector peers.** Up to 8 companies Screener lists in the same industry are measured the same way as the company. A red or watch flag is downgraded one step when the company is no worse than the typical peer (a sector norm), and a clear core check is raised to watch when the company is worse than 85% of five or more peers. Every adjusted flag shows its original severity.

## Score

Each check is clear (0), watch (1) or red (2) with a weight. The score is the weighted share of the maximum, counting only checks that had data. **Coverage** shows how much of the model ran. Bands: under 20 low, 20–35 some questions, 35–55 elevated, above 55 high.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `forensics.py` | All formulas, flags, lender module and the score |
| `sector.py` | Company-type classification and each type's rule book |
| `peers.py` | Peer metrics (industrial, lender, broker, insurer), comparison and flag adjustment |
| `data_sources.py` | Symbol picker, Screener.in page reader, Yahoo loader, Screener export parser, NSE client |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Theme |

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository (for example `forensic-screener`).
2. Upload all files using **Add file → Upload files**. Keep the `.streamlit` folder: drag the folder itself, or create `.streamlit/config.toml` with **Add file → Create new file** and paste its contents.
3. On share.streamlit.io choose **Create app**, pick the repository, set the main file to `app.py`, and deploy.
4. Optionally add the app URL to your keep-awake workflow.

To run locally: `pip install -r requirements.txt` then `streamlit run app.py`.

## Using it

- Type an exact NSE symbol to open it directly, or part of a company name to get a list of likely scrips.
- Financials come from the public Screener.in company page (consolidated, falling back to standalone), about 12 years. Switch to Yahoo in the sidebar if Screener is unavailable; the app also falls back automatically.
- The Data tab cross-checks Screener against Yahoo for the last three years and marks gaps above 5%.
- An uploaded Screener Excel export, if given, overrides both.
- Each flag has a "Why it matters / innocent explanations" panel. Read it before drawing conclusions.

## Known limits

- Yahoo gives about four years of annual data for NSE stocks; Screener about twelve.
- From Screener's groupings, current assets, current liabilities and retained earnings are approximations, and interest income isn't split from other income. This mainly affects Altman Z″ and parts of Beneish.
- Screener's page layout can change; if it does, the app falls back to Yahoo and says so.
- NSE sometimes blocks requests from cloud servers. When that happens the ownership and filings layers show as not evaluated; retry later or run locally.
- NSE's pledge endpoint usually returns only the latest quarter, so the pledge trend is often unavailable.
- Related-party transactions, contingent liabilities and auditor remarks sit in the annual report, which this version does not read.
- Keyword matching can miss filings and can over-match. Open each filing.
- Lender checks are approximate: Screener doesn't split loans or provisions, so the loan book comes from its "Other Assets" grouping and provision coverage ignores technical write-offs. Check CRAR, slippages and the restructured book in the bank's own disclosures.
- If a whole sector shares a problem, peer comparison can make a company look normal; read the original severity on adjusted flags.
- Broker client-money cover uses Screener's 'Other Liabilities' as client liabilities; money parked as exchange margin or fixed deposits can sit outside 'cash', understating cover.
- Insurer claims, commission and reserve changes aren't split on Screener, so solvency, combined ratio and persistency come only from figures you enter.
- Screener rate-limits rapid requests; the app backs off and retries, but heavy use can still slow it down. Some companies (e.g. ICICI Prudential Life) aren't listed on Screener at all.
- The first run for a company takes 15–25 seconds (company plus peers); results are cached afterwards.
- Beneish and similar models were built on US data and give false positives, especially for fast-growing companies.
- The Screener export parser was tested on a mock workbook built to the export's layout, not on a downloaded file.

## Disclaimer

For research and education only. Not investment advice and not an allegation of wrongdoing against any company. Public data may be incomplete or delayed. Keep any concerns the app raises as private research notes rather than public claims.
