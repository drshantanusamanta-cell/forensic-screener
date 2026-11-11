"""
Company archetypes and the per-type rule book.

A forensic test that is sharp for a manufacturer can be meaningless for a bank
and misleading for a builder (inventory = land) or a jeweller (inventory = gold).
Each archetype adjusts thresholds, weights, or switches tests off.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

# Default thresholds used by forensics.financial_flags
DEFAULT_TH = {
    "cfo_ebitda": {"green": 0.70, "red": 0.50},
    "cfo_ni": {"green": 0.80, "red": 0.50},
    "dso": {"amber_rel": 0.15, "amber_abs": 7, "red_rel": 0.30, "red_abs": 15},
    "dio": {"amber_rel": 0.15, "amber_abs": 10, "red_rel": 0.30, "red_abs": 20},
    "cash_yield": {"red": 0.02, "amber": 0.04},
    "cash_debt": {"amber": 0.15, "red": 0.25},
    "tax": {"red": 0.12, "amber": 0.18},
    "other_income": {"amber": 0.20, "red": 0.40},
    "cwip": {"amber": 0.50, "red": 1.00},
    "debt": {"amber": 1.0, "red": 2.0},
    "altman": {"model": "z2"},
}


@dataclass
class Profile:
    key: str
    label: str
    note: str
    th: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_TH))
    weights: dict = field(default_factory=dict)     # flag key -> new weight
    skip: dict = field(default_factory=dict)        # flag key -> reason it doesn't apply
    financial_layer: bool = True                     # run the accounting tests at all
    lender_layer: bool = False                       # run the bank / NBFC tests
    broker_layer: bool = False                       # run the stockbroker tests
    insurer_layer: str = ""                          # "life" / "general" -> run the insurer tests
    keep_only: tuple = ()                            # if set, only these accounting tests are scored

    @property
    def peer_mode(self) -> str:
        if self.lender_layer:
            return "lender"
        if self.broker_layer:
            return "broker"
        if self.insurer_layer:
            return "insurer"
        return "industrial" if self.financial_layer else ""


def _p(key, label, note, th=None, weights=None, skip=None, financial_layer=True, lender_layer=False,
       broker_layer=False, insurer_layer="", keep_only=()):
    pr = Profile(key, label, note, weights=weights or {}, skip=skip or {},
                 financial_layer=financial_layer, lender_layer=lender_layer,
                 broker_layer=broker_layer, insurer_layer=insurer_layer, keep_only=tuple(keep_only))
    for k, v in (th or {}).items():
        pr.th[k].update(v)
    return pr


PROFILES = {
    "manufacturer": _p(
        "manufacturer", "Manufacturer",
        "Standard rule book. Altman uses the original manufacturing Z-score with market value of equity.",
        th={"altman": {"model": "z_orig"}}),
    "it_services": _p(
        "it_services", "IT / software services",
        "Asset-light: cash conversion should be high, debt low, and receivables (incl. unbilled) are the main "
        "place earnings can be stretched. Inventory and CWIP tests are dropped.",
        th={"cfo_ebitda": {"green": 0.75, "red": 0.55}, "debt": {"amber": 0.5, "red": 1.0}},
        weights={"dso": 3, "intangibles": 1.5, "cash_yield": 3},
        skip={"dio": "IT services carry little or no inventory", "cwip": "not material for IT services"}),
    "retail": _p(
        "retail", "Retail / jewellery / consumer distribution",
        "Large inventory and working-capital swings are part of the model (gold, fashion, store stock), so "
        "cash-conversion and inventory thresholds are looser; receivables should be tiny.",
        th={"cfo_ebitda": {"green": 0.60, "red": 0.35}, "cfo_ni": {"green": 0.70, "red": 0.40},
            "dio": {"amber_rel": 0.25, "amber_abs": 25, "red_rel": 0.45, "red_abs": 50}},
        weights={"dio": 1, "dso": 2.5}),
    "epc_realty": _p(
        "epc_realty", "Construction / EPC / real estate",
        "Long project cycles: receivables, contract assets and inventory (land, work-in-progress) are naturally "
        "large and cash lags profit. Thresholds are looser; debt tolerance is higher.",
        th={"cfo_ebitda": {"green": 0.50, "red": 0.30}, "cfo_ni": {"green": 0.60, "red": 0.30},
            "dso": {"amber_rel": 0.25, "amber_abs": 20, "red_rel": 0.50, "red_abs": 40},
            "dio": {"amber_rel": 0.30, "amber_abs": 60, "red_rel": 0.60, "red_abs": 120},
            "debt": {"amber": 1.5, "red": 3.0}},
        weights={"fcf": 1, "cwip": 0.5}),
    "utilities": _p(
        "utilities", "Power / telecom / infrastructure operator",
        "Regulated, capital-heavy businesses: high debt and multi-year CWIP are normal, free cash flow is often "
        "negative during build-out. Debt and CWIP thresholds are higher.",
        th={"cwip": {"amber": 0.60, "red": 1.20}, "debt": {"amber": 2.0, "red": 3.5},
            "cash_debt": {"amber": 0.20, "red": 0.30}},
        weights={"fcf": 0.5, "altman": 1, "cwip": 1.5}),
    "pharma": _p(
        "pharma", "Pharmaceuticals / biotech",
        "Capitalised R&D and acquired intangibles deserve more weight; receivables (distributor stuffing, "
        "US generics price erosion) matter.",
        weights={"intangibles": 2, "dso": 2.5}),
    "services": _p(
        "services", "Services (hospitals, media, hospitality, logistics…)",
        "Standard rule book; inventory test dropped when inventory is immaterial.",
        weights={"dio": 0.5}),
    "holding": _p(
        "holding", "Holding / investment company",
        "Profit is mostly the share of associates' earnings plus dividends and investment income — already taxed "
        "elsewhere and not received as cash — so tax, cash-conversion-of-profit, accrual and other-income tests "
        "don't apply. The remaining tests are lightly weighted.",
        weights={"cfo_ebitda": 1, "beneish": 1},
        skip={"other_income": "investment income is the business", "cash_yield": "treasury is the business",
              "altman": "not meaningful for holding companies", "dio": "no operating inventory",
              "tax": "associates' profits are booked after their own tax",
              "cfo_ni": "equity-accounted profits are non-cash",
              "sloan": "equity-accounted profits inflate accruals mechanically"}),
    "broker": _p(
        "broker", "Stockbroker",
        "Client money, exchange margins and margin-trading loans run through the balance sheet and cash flow, so "
        "the industrial cash tests mislead. Broker checks are used instead: client-money cover, leverage, "
        "debt-funded growth and receivables; tax and intangibles are still scored.",
        broker_layer=True, keep_only=("tax", "intangibles", "dso"),
        th={"dso": {"amber_rel": 0.30, "amber_abs": 10, "red_rel": 0.60, "red_abs": 25}}),
    "exchange": _p(
        "exchange", "Exchange / clearing",
        "Clearing members' margins and settlement funds sit in the exchange group's cash and operating cash flow, "
        "so cash-conversion, accrual and cash-yield tests are dropped. Treasury income is a normal, large part "
        "of profit.",
        th={"other_income": {"amber": 0.35, "red": 0.60}},
        skip={k: "clearing-member money distorts it for exchanges"
              for k in ("cfo_ebitda", "cfo_ni", "fcf", "sloan", "beneish", "montier", "cash_yield", "cash_debt",
                        "altman", "dio")}),
    "amc_rta": _p(
        "amc_rta", "AMC / registrar / depository / ratings",
        "Asset-light fee businesses with no client money on the balance sheet: cash conversion should be high. "
        "They invest their own surplus, so treasury income is a larger, normal share of profit.",
        th={"cfo_ebitda": {"green": 0.75, "red": 0.55}, "other_income": {"amber": 0.30, "red": 0.50},
            "debt": {"amber": 0.5, "red": 1.0}},
        weights={"dso": 2.5},
        skip={"dio": "no inventory", "cwip": "not material", "cash_yield": "surplus sits in investments, not cash",
              "altman": "not meaningful for asset-light fee businesses",
              "sloan": "investing surplus cash inflates the cash-flow accrual ratio mechanically"}),
    "bank": _p(
        "bank", "Bank",
        "Lending balance sheet: industrial accounting tests are switched off and replaced by asset-quality, "
        "provision-coverage, growth, funding and capital checks.",
        financial_layer=False, lender_layer=True),
    "nbfc": _p(
        "nbfc", "NBFC / housing finance",
        "Lending balance sheet: industrial tests are off; asset-quality, provision-coverage, growth and leverage "
        "checks are used instead. Negative operating cash flow is normal while the loan book grows.",
        financial_layer=False, lender_layer=True),
    "insurer_life": _p(
        "insurer_life", "Life insurer",
        "Industrial tests don't fit insurance accounts. Insurer checks use what Screener shows (premium income, "
        "total outgo, investments, policyholder liabilities, cash flow); solvency ratio and 13th-month persistency "
        "can be typed in from the latest IRDAI public disclosure.",
        financial_layer=False, insurer_layer="life"),
    "insurer_general": _p(
        "insurer_general", "General / health insurer",
        "Industrial tests don't fit insurance accounts. Insurer checks use what Screener shows; solvency ratio and "
        "combined ratio can be typed in from the latest IRDAI public disclosure.",
        financial_layer=False, insurer_layer="general"),
}


def _has(text: str, *words) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


def classify(screener_cls: dict | None, yf_info: dict | None, is_financial_layout: bool = False) -> tuple[str, str]:
    """
    -> (profile key, how it was decided). Uses Screener's NSE classification first, Yahoo as fallback.
    """
    c = screener_cls or {}
    bsec, sec, bind, ind = (c.get("Broad Sector", ""), c.get("Sector", ""),
                            c.get("Broad Industry", ""), c.get("Industry", ""))
    allc = " | ".join([bsec, sec, bind, ind])
    src = f"NSE classification: {ind or bind or sec}" if allc.strip(" |") else ""

    if allc.strip(" |"):
        if _has(bsec, "financial"):
            if _has(ind, "non banking", "nbfc", "housing finance"):
                return "nbfc", src
            if _has(bind, "bank") or _has(ind, "sector bank", "small finance bank", "payments bank"):
                return "bank", src
            if _has(bind, "insurance") or _has(ind, "insurance"):
                return ("insurer_life" if _has(ind, "life") else "insurer_general"), src
            if _has(ind, "holding company", "investment company"):
                return "holding", src
            if _has(ind, "stockbroking", "broking"):
                return "broker", src
            if _has(ind, "exchange"):
                return "exchange", src
            if _has(bind, "capital markets") or _has(ind, "asset management", "depositor", "registrar", "wealth",
                                                      "ratings", "distributor", "other capital market",
                                                      "other financial services"):
                return "amc_rta", src
            if _has(ind, "non banking", "nbfc", "housing finance", "microfinance", "financial institution",
                    "finance company", "leasing"):
                return "nbfc", src
            return "amc_rta", src  # unknown financial-services label: fee-business rules, not lender rules
        if _has(ind, "holding company", "investment company"):
            return "holding", src
        if _has(bsec, "information technology") or _has(ind, "software", "it enabled", "computers -"):
            return "it_services", src
        if _has(sec, "realty", "construction") or _has(ind, "civil construction", "residential", "real estate",
                                                        "construction"):
            return "epc_realty", src
        if _has(sec, "power", "telecommunication") or _has(ind, "power", "telecom", "gas distribution",
                                                            "port", "airport", "road assets", "transport related"):
            return "utilities", src
        if _has(ind, "pharmaceutical", "biotech"):
            return "pharma", src
        if _has(ind, "retail", "jewel", "gems", "e-retail", "distributor", "trading"):
            return "retail", src
        if _has(bsec, "services", "healthcare", "media", "consumer services") and not _has(ind, "pharma"):
            return "services", src
        if _has(ind, "hospital", "healthcare service", "hotel", "restaurant", "media", "film", "broadcast",
                "logistics", "airline", "shipping", "education", "advertis"):
            return "services", src
        return "manufacturer", src

    # Fallback: Yahoo sector / industry
    info = yf_info or {}
    s, i = str(info.get("sector", "")), str(info.get("industry", ""))
    src = f"Yahoo classification: {i or s}" if (s or i) else "no classification found — default rules"
    if is_financial_layout or _has(s, "financial"):
        if _has(i, "bank"):
            return "bank", src
        if _has(i, "insurance"):
            return ("insurer_life" if _has(i, "life") else "insurer_general"), src
        if _has(i, "capital markets"):
            return "broker", src
        if _has(i, "exchange"):
            return "exchange", src
        if _has(i, "asset management"):
            return "amc_rta", src
        return "nbfc", src
    if _has(s, "technology") and _has(i, "software", "information technology", "it services"):
        return "it_services", src
    if _has(s, "real estate") or _has(i, "engineering & construction", "construction"):
        return "epc_realty", src
    if _has(s, "utilities", "communication services") and _has(i, "utilit", "telecom", "power"):
        return "utilities", src
    if _has(i, "drug", "biotech", "pharma"):
        return "pharma", src
    if _has(i, "retail", "luxury", "department", "distribution"):
        return "retail", src
    if _has(i, "conglomerate", "holding"):
        return "holding", src
    if _has(s, "healthcare", "communication services") or _has(i, "lodging", "airline", "media", "hospital"):
        return "services", src
    return "manufacturer", src
