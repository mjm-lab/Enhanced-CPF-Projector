from datetime import date, datetime
import pandas as pd
import streamlit as st
import altair as alt

# ==============================
# Aesthetics & Page Config
# ==============================
st.set_page_config(
    page_title="CPF Projector",
    page_icon="🧮",
    layout="wide"
)

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()

def altair_to_png_bytes(chart, width=900, height=360, scale=1):
    # Ensure charts have fixed size for consistent PDFs
    ch = chart.properties(width=width, height=height)
    buf = BytesIO()
    ch.save(buf, format="png", scale=scale)  # requires vl-convert-python
    return buf.getvalue()

def df_to_table(df, cols, col_widths=None, max_rows=30, number_format=None):
    data = [cols]
    _df = df.loc[:, cols].copy()
    if number_format:
        for c, fmt in number_format.items():
            if c in _df.columns:
                _df[c] = _df[c].map(lambda x: fmt.format(x) if isinstance(x, (int, float)) else x)
    if len(_df) > max_rows:
        _df = _df.head(max_rows)
    data += _df.astype(str).values.tolist()
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
    ]))
    return tbl

def make_pdf_report(
    monthly_df, yearly_df, cohort_frs, cpf_life_df, bequest_df,
    include_cpf_life: bool, cpf_life_plan: str, payout_start_age: int,
    notes_html_list: list[str] | None = None,
):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    story = []

    # --- Title page ---
    story += [
        Paragraph("CPF Projector — Report", styles["Title"]),
        Spacer(1, 6),
        Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"Cohort FRS: ${cohort_frs:,.0f}", styles["Normal"]),
        Spacer(1, 18),
    ]

    # ========== 1) Account Balances ==========
    story += [Paragraph("1) Account Balances", styles["Heading2"]), Spacer(1, 6)]

    # KPI mini-table
    end_row = yearly_df.sort_values("Year").iloc[-1]
    kpi_cols = ["Account", "Final Balance (S$)"]
    kpi_df = (
        pd.DataFrame({
            "Account": ["OA","SA","MA","RA"],
            "Final Balance (S$)": [end_row["End_OA"], end_row["End_SA"], end_row["End_MA"], end_row["End_RA"]],
        })
    )
    story += [df_to_table(kpi_df, kpi_cols, number_format={"Final Balance (S$)": "{:,.0f}"}), Spacer(1, 10)]

    # Rebuild stacked yearly Altair chart and insert as PNG
    yearly_long = yearly_df.melt(
        id_vars=['Year','Age_end'],
        value_vars=['End_OA','End_SA','End_MA','End_RA'],
        var_name='Account', value_name='Balance'
    ).replace({'End_OA':'OA','End_SA':'SA','End_MA':'MA','End_RA':'RA'})
    yearly_long['YearAge'] = yearly_long.apply(lambda r: f"{int(r['Year'])} (Age {int(r['Age_end'])})", axis=1)
    stacked = (
        alt.Chart(yearly_long)
           .mark_bar()
           .encode(
              x=alt.X('YearAge:O', title='Year (Age)', sort=None),
              y=alt.Y('sum(Balance):Q', title='Balance (S$)', axis=alt.Axis(format=',.0f')),
              color=alt.Color('Account:N', legend=alt.Legend(title='Account'))
           )
           .properties(width=900, height=320)
    )
    frs_rule = alt.Chart(pd.DataFrame({'y': [cohort_frs]})).mark_rule(strokeDash=[6,3]).encode(y='y:Q')
    png = altair_to_png_bytes(stacked + frs_rule)
    story += [Image(BytesIO(png), width=6.5*inch, height= (6.5*inch)*320/900), Spacer(1, 12)]

    # Yearly summary table (pick the important columns; keep it short)
    cols = ['Year','Age_end','End_OA','End_SA','End_MA','End_RA','RA_capital_end','Prevailing_ERS']
    story += [df_to_table(yearly_df, cols, number_format={c: "{:,.0f}" for c in cols if c != "Year" and c != "Age_end"}), Spacer(1, 16)]

    story += [PageBreak()]

    # ========== 2) CPF LIFE & Bequest ==========
    story += [Paragraph("2) CPF LIFE & Bequest", styles["Heading2"]), Spacer(1, 6)]
    if include_cpf_life and (cpf_life_df is not None) and (bequest_df is not None):
        # Dual-axis style: draw lines separately and layer
        beq_plot = bequest_df.copy()
        beq_plot["YearAge"] = beq_plot["Year"].map(lambda y: f"{int(y)}")
        bequest_long = beq_plot.melt(
            id_vars=["Year","YearAge"],
            value_vars=["Bequest_Remaining","RA_Savings_Remaining","Unused_Premium"],
            var_name="Component", value_name="Amount"
        )
        beq_lines = (
            alt.Chart(bequest_long)
               .mark_line(point=True)
               .encode(
                   x=alt.X("Year:O", title="Year", sort=None),
                   y=alt.Y("Amount:Q", title="Bequest (S$)", axis=alt.Axis(format=',.0f')),
                   color=alt.Color("Component:N", title="Component")
               )
               .properties(width=900, height=300)
        )
        payout_plot = cpf_life_df.copy()
        payout_plot["YearAge"] = payout_plot["Year"].map(lambda y: f"{int(y)}")
        payout_line = (
            alt.Chart(payout_plot)
               .mark_line(point=True, strokeDash=[4,2])
               .encode(
                   x=alt.X("Year:O", title="Year", sort=None),
                   y=alt.Y("Monthly_Payout:Q", title="Monthly Payout (S$)", axis=alt.Axis(format=',.0f')),
                   color=alt.value("#6b7280")
               )
               .properties(width=900, height=300)
        )
        png = altair_to_png_bytes(beq_lines + payout_line)
        story += [Image(BytesIO(png), width=6.5*inch, height=(6.5*inch)*300/900), Spacer(1, 10)]

        # Small payout schedule table
        story += [Paragraph(f"Plan: {cpf_life_plan} · Start age: {payout_start_age}", styles["Normal"]), Spacer(1, 6)]
        story += [df_to_table(cpf_life_df, ["Year","Monthly_Payout","Annual_Payout"],
                              number_format={"Monthly_Payout": "{:,.0f}", "Annual_Payout": "{:,.0f}"},
                              max_rows=25),
                  Spacer(1, 16)]
    else:
        story += [Paragraph("CPF LIFE charts/tables not available for this run.", styles["Italic"]), Spacer(1, 16)]

    story += [PageBreak()]

    # ========== 3) Cashflows ==========
    story += [Paragraph("3) Cashflows", styles["Heading2"]), Spacer(1, 6)]

    # Rebuild the nominal & real lines exactly like the app’s logic (abbrev.)
    _snap = (monthly_df.sort_values(["Year","Month"]).groupby("Year").tail(1)
             [["Year","CPF_LIFE_monthly_payout","OA_Withdrawal1_Paid","OA_Withdrawal2_Paid"]].copy())
    _snap["YearAge"] = _snap["Year"].map(lambda y: f"{int(y)}")
    _nom = _snap.copy()
    _nom["Total"] = _nom[["CPF_LIFE_monthly_payout","OA_Withdrawal1_Paid","OA_Withdrawal2_Paid"]].sum(axis=1)
    nom_long = _nom.melt(id_vars=["Year","YearAge"],
                         value_vars=["Total","CPF_LIFE_monthly_payout","OA_Withdrawal1_Paid","OA_Withdrawal2_Paid"],
                         var_name="Series", value_name="Amount")
    nom_chart = (
        alt.Chart(nom_long).mark_line(point=True)
           .encode(x=alt.X("Year:O", title="Year", sort=None),
                   y=alt.Y("Amount:Q", title="Amount (S$)", axis=alt.Axis(format=',.0f')),
                   color=alt.Color("Series:N", title="Component"))
           .properties(width=900, height=300)
    )
    story += [Image(BytesIO(altair_to_png_bytes(nom_chart)), width=6.5*inch, height=(6.5*inch)*300/900),
              Spacer(1, 12)]

    # Real chart
    _today_year = date.today().year
    _snap["Deflator"] = (1.0 + float(meta["inflation_pct"])) ** (_snap["Year"] - _today_year)
    _real = _snap.copy()
    _real["CPF"]  = _real["CPF_LIFE_monthly_payout"] / _real["Deflator"]
    _real["OA_A"] = _real["OA_Withdrawal1_Paid"] / _real["Deflator"]
    _real["OA_B"] = _real["OA_Withdrawal2_Paid"] / _real["Deflator"]
    _real["Total"] = _real[["CPF","OA_A","OA_B"]].sum(axis=1)
    real_long = _real.melt(id_vars=["Year","YearAge"], value_vars=["Total","CPF","OA_A","OA_B"],
                           var_name="Series", value_name="Amount")
    real_chart = (
        alt.Chart(real_long).mark_line(point=True)
           .encode(x=alt.X("Year:O", title="Year", sort=None),
                   y=alt.Y("Amount:Q", title="Amount (S$, today’s $)", axis=alt.Axis(format=',.0f')),
                   color=alt.Color("Series:N", title="Component"))
           .properties(width=900, height=300)
    )
    story += [Image(BytesIO(altair_to_png_bytes(real_chart)), width=6.5*inch, height=(6.5*inch)*300/900),
              Spacer(1, 16)]

    story += [PageBreak()]

    # ========== 4) Health Insurance ==========
    story += [Paragraph("4) Health Insurance", styles["Heading2"]), Spacer(1, 6)]
    has_hi = all(c in yearly_df.columns for c in
                 ["MSHL_Nominal","IP_Base_MA_Nominal_Annual","IP_Base_Cash_Nominal_Annual","IP_Rider_Cash_Nominal_Annual"])
    if has_hi and yearly_df[["MSHL_Nominal","IP_Base_MA_Nominal_Annual","IP_Base_Cash_Nominal_Annual","IP_Rider_Cash_Nominal_Annual"]].sum().sum() > 0:
        plot_df = yearly_df[["Year","Age_end","MSHL_Nominal","IP_Base_MA_Nominal_Annual","IP_Base_Cash_Nominal_Annual","IP_Rider_Cash_Nominal_Annual"]].copy()
        plot_df["YearAge"] = plot_df["Year"].map(str)
        hi_long = plot_df.melt(id_vars=["Year","YearAge"],
                               value_vars=["MSHL_Nominal","IP_Base_MA_Nominal_Annual","IP_Base_Cash_Nominal_Annual","IP_Rider_Cash_Nominal_Annual"],
                               var_name="Component", value_name="Amount")
        ip_chart = (
            alt.Chart(hi_long).mark_bar()
               .encode(x=alt.X("YearAge:O", title="Year", sort=None),
                       y=alt.Y("sum(Amount):Q", title="Premium (S$)", axis=alt.Axis(format=',.0f')),
                       color=alt.Color("Component:N", title="Component"))
               .properties(width=900, height=300)
        )
        story += [Image(BytesIO(altair_to_png_bytes(ip_chart)), width=6.5*inch, height=(6.5*inch)*300/900), Spacer(1, 12)]
    else:
        story += [Paragraph("No health insurance premiums in this run.", styles["Italic"]), Spacer(1, 12)]

    # ========== Notes ==========
    if notes_html_list:
        story += [Paragraph("Notes", styles["Heading3"])]
        for n in notes_html_list:
            # strip simple HTML tags (or use a proper cleaner if needed)
            story += [Paragraph(n.replace("<b>","").replace("</b>","").replace("&times;","x"), styles["Normal"])]

    doc.build(story)
    return buf.getvalue()



import hashlib
import json

# Keep results across reruns
if "proj_results" not in st.session_state:
    st.session_state.proj_results = None  # tuple: (monthly_df, yearly_df, cohort_frs, cohort_ers, meta, cpf_life_df, bequest_df)
if "params_hash" not in st.session_state:
    st.session_state.params_hash = None
if "ran_once" not in st.session_state:
    st.session_state.ran_once = False

st.markdown(
    """
    <style>
    .metric-card {
        border-radius: 16px;
        padding: 16px 18px;
        border: 1px solid rgba(0,0,0,0.08);
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .small-muted { color: #6b7280; font-size: 12px; }
    .section-title { font-weight: 700; font-size: 20px; margin-top: 8px; }
    .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: #ecfeff;
        color: #0369a1;
        font-size: 12px;
        margin-left: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================
# Assumptions / Policy Tables
# ==============================
BASE_INT = {"OA": 0.025, "SA": 0.04, "MA": 0.04, "RA": 0.04}
ERS_FACTOR_DEFAULT = 2.0  # default factor: 2×FRS

EXTRA_BELOW_55 = {"pool": 60000.0, "oa_cap": 20000.0, "tier1_rate": 0.01}
EXTRA_55_PLUS = {
    "tier1_amount": 30000.0, "tier1_rate": 0.02,
    "tier2_amount": 30000.0, "tier2_rate": 0.01,
    "oa_cap": 20000.0
}


def ow_ceiling_monthly(year: int):
    if year <= 2024:
        return 6800.0
    elif year == 2025:
        return 7400.0
    else:  # 2026+
        return 8000.0

ANNUAL_TW_CEILING = 102000.0


# Full Retirement Sum by cohort year (age 55 in <year>)
FRS_KNOWN = {
    1996:  40000.0, 1997:  45000.0, 1998:  50000.0,
    1999:  55000.0, 2000:  60000.0, 2001:  65000.0,
    2002:  70000.0, 2003:  75000.0, 2004:  80000.0,
    2005:  84500.0, 2006:  90000.0, 2007:  94600.0,
    2008:  99600.0, 2009: 106000.0, 2010: 117000.0,
    2011: 123000.0, 2012: 131000.0, 2013: 139000.0,
    2014: 148000.0, 2015: 155000.0, 2016: 161000.0,
    2017: 166000.0, 2018: 171000.0, 2019: 176000.0,
    2020: 181000.0, 2021: 186000.0, 2022: 192000.0,
    2023: 198800.0, 2024: 205800.0, 2025: 213000.0,
    2026: 220400.0, 2027: 228200.0,
}

FRS_growth_pct_default = 3.5

BHS_KNOWN = {2025: 75500.0}
BHS_growth_pct_default = 5.0

# ---- Cohort BHS (fixed at 65) ----
COHORT_BHS_BY_YEAR65 = {
    2025: 75500.0, 2024: 71500.0, 2023: 68500.0, 2022: 66000.0,
    2021: 63000.0, 2020: 60000.0, 2019: 57200.0, 2018: 54500.0,
    2017: 52000.0, 2016: 49800.0
}

# ==============================
# CPF LIFE payout coefficients (monthly)
# payout = a + b * RA_at_65 (when start age = 65)
# ==============================
CPF_LIFE_COEFFS = {
    "M": {
        "Standard_65":   {"a": 72.29412442, "b": 0.005182694},
        "Escalating_65": {"a": 58.33903251, "b": 0.004085393},
        "Basic_65":      {"a": 66.1522067,  "b": 0.004734576},
    },
    "F": {
        "Standard_65":   {"a": 68.51952457, "b": 0.004825065},
        "Escalating_65": {"a": 54.50151582, "b": 0.003718627},
        "Basic_65":      {"a": 66.44720836, "b": 0.004566059},
    },
}
CPF_LIFE_DEFERRAL_PER_YEAR = 0.07  # ~7% more payout per year of deferral beyond 65
ESCALATING_RATE = 0.02             # +2% per year escalation after start
BASIC_PREMIUM_FRAC = 0.10          # fraction of RA paid as premium at start for Basic (approx)


# ==============================
# Helper functions
# ==============================
def _label_year_age(y, yearly_df):
    try:
        age = int(yearly_df.set_index("Year").loc[y, "Age_end"])
        return f"{y} (Age {age})"
    except KeyError:
        return str(y)

def age_at(dob: date, year: int, month: int):
    d = date(year, month, 28)  # approx month-end
    return d.year - dob.year - ((d.month, d.day) < (dob.month, d.day))

def get_alloc_for_age(age):
    for row in [
        {"min_age": 0,  "max_age": 35, "total": 0.37,  "OA": 0.23,  "SA_RA": 0.06,  "MA": 0.08},
        {"min_age": 35, "max_age": 45, "total": 0.37,  "OA": 0.21,  "SA_RA": 0.07,  "MA": 0.09},
        {"min_age": 45, "max_age": 50, "total": 0.37,  "OA": 0.19,  "SA_RA": 0.08,  "MA": 0.10},
        {"min_age": 50, "max_age": 55, "total": 0.37,  "OA": 0.15,  "SA_RA": 0.115, "MA": 0.105},
        {"min_age": 55, "max_age": 60, "total": 0.325, "OA": 0.12,  "SA_RA": 0.10,  "MA": 0.105},
        {"min_age": 60, "max_age": 65, "total": 0.235, "OA": 0.035, "SA_RA": 0.095, "MA": 0.105},
        {"min_age": 65, "max_age": 70, "total": 0.165, "OA": 0.01,  "SA_RA": 0.085, "MA": 0.07},
        {"min_age": 70, "max_age": 200,"total": 0.125, "OA": 0.005, "SA_RA": 0.07,  "MA": 0.05},
    ]:
        if (age >= row["min_age"]) and (age < row["max_age"]):
            return row
    return {"total":0.0,"OA":0.0,"SA_RA":0.0,"MA":0.0}

def get_frs_for_cohort(year55: int, frs_growth_pct: float):
    if year55 in FRS_KNOWN:
        return FRS_KNOWN[year55]
    last_year = max(FRS_KNOWN.keys())
    frs = FRS_KNOWN[last_year]
    for _ in range(last_year+1, year55+1):
        frs *= (1 + frs_growth_pct)
    return round(frs, 0)

def get_frs_for_year(year: int, frs_growth_pct: float) -> float:
    if year in FRS_KNOWN:
        return FRS_KNOWN[year]
    last_year = max(FRS_KNOWN.keys())
    frs = FRS_KNOWN[last_year]
    for _ in range(last_year + 1, year + 1):
        frs *= (1 + frs_growth_pct)
    return round(frs, 0)

def get_bhs_for_year_with_cohort(dob: date, year: int, bhs_growth_pct: float):
    year65 = dob.year + 65
    if year < year65:
        # Prevailing national BHS BEFORE 65
        if year in BHS_KNOWN:
            return BHS_KNOWN[year]
        # Use your historical prevailing BHS for pre-2025 years:
        if year <= 2024:
            return float(COHORT_BHS_BY_YEAR65.get(year, COHORT_BHS_BY_YEAR65[2016]))
        # For 2026+ extrapolate from the last known (2025)
        bhs = BHS_KNOWN[2025]
        for _ in range(2026, year + 1):
            bhs *= (1 + bhs_growth_pct)
        return round(bhs, 0)
    else:
        # Cohort BHS fixed at age 65 (your current logic is fine)
        if year65 in BHS_KNOWN:
            cohort_bhs = BHS_KNOWN[year65]
        elif year65 >= 2026:
            cohort_bhs = BHS_KNOWN[2025]
            for _ in range(2026, year65 + 1):
                cohort_bhs *= (1 + bhs_growth_pct)
        else:
            # Historical cohort table for <=2024 cohorts
            for k in [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]:
                if year65 >= k:
                    cohort_bhs = COHORT_BHS_BY_YEAR65[k]
                    break
        return round(cohort_bhs, 0)


# ---- MediShield Life premium (GST inc.) by ANB ----
def get_mshl_premium_by_anb(anb: int) -> float:
    bands = [
        ((1,20), 200), ((21,30), 295), ((31,40), 503), ((41,50), 637),
        ((51,60), 903), ((61,65), 1131), ((66,70), 1326), ((71,73), 1643),
        ((74,75), 1816), ((76,78), 2027), ((79,80), 2187), ((81,83), 2303),
        ((84,85), 2616), ((86,88), 2785), ((89,90), 2785), ((91,200), 2826)
    ]
    for (lo, hi), prem in bands:
        if lo <= anb <= hi:
            return float(prem)
    return 0.0

# -------- Extra interest (single rule; same before/after LIFE accrual) --------
def compute_extra_interest_distribution(age, oa, sa, ma, ra):
    """Return ANNUAL extra-interest (not /12) allocation by source account."""
    ei = {"OA": 0.0, "SA": 0.0, "MA": 0.0, "RA": 0.0}
    if age < 55:
        remaining = EXTRA_BELOW_55["pool"]
        take_oa = min(remaining, min(oa, EXTRA_BELOW_55["oa_cap"]))
        if take_oa > 0:
            ei["OA"] += take_oa * EXTRA_BELOW_55["tier1_rate"]
            remaining -= take_oa
        take_sa = min(remaining, sa)
        if take_sa > 0:
            ei["SA"] += take_sa * EXTRA_BELOW_55["tier1_rate"]
            remaining -= take_sa
        take_ma = min(remaining, ma)
        if take_ma > 0:
            ei["MA"] += take_ma * EXTRA_BELOW_55["tier1_rate"]
            remaining -= take_ma
    else:
        t1 = 30000.0; t2 = 30000.0
        oa_cap = 20000.0
        r1 = min(t1, ra); t1 -= r1
        o1 = min(t1, min(oa, oa_cap)); t1 -= o1
        s1 = min(t1, sa); t1 -= s1
        m1 = min(t1, ma); t1 -= m1
        ei["RA"] += r1 * 0.02; ei["OA"] += o1 * 0.02; ei["SA"] += s1 * 0.02; ei["MA"] += m1 * 0.02

        r2 = min(t2, max(0.0, ra - r1)); t2 -= r2
        o2 = min(t2, max(0.0, min(oa, oa_cap) - o1)); t2 -= o2
        s2 = min(t2, max(0.0, sa - s1)); t2 -= s2
        m2 = min(t2, max(0.0, ma - m1)); t2 -= m2
        ei["RA"] += r2 * 0.01; ei["OA"] += o2 * 0.01; ei["SA"] += s2 * 0.01; ei["MA"] += m2 * 0.01
    return ei

def spill_from_ma(age, ma_end, bhs, sa, oa, ra, frs_for_cohort, ra_capital):
    """
    Enforce BHS with spillovers:
    <55: MA -> SA (up to cohort FRS), then OA
    ≥55: MA -> RA (only until RA capital reaches cohort FRS), then OA
    """
    if ma_end <= bhs:
        return ma_end, sa, oa, ra
    excess = ma_end - bhs
    ma_end = bhs
    if age < 55:
        space_sa = max(0.0, frs_for_cohort - sa)
        to_sa = min(excess, space_sa); sa += to_sa; excess -= to_sa
        oa += excess; excess = 0.0
    else:
        space_ra_cap = max(0.0, frs_for_cohort - ra_capital)
        to_ra = min(excess, space_ra_cap); ra += to_ra; excess -= to_ra
        oa += excess; excess = 0.0
    return ma_end, sa, oa, ra

def transfer_to_ra_at_55(age_this_month, sa, oa, ra, transfer_target, cohort_frs):
    if age_this_month < 55:
        return sa, oa, ra, 0.0

    moved_capital = 0.0
    total_pre = oa + sa

    # Case 1: below FRS
    if total_pre < cohort_frs:
        keep = min(5000.0, total_pre)
        if oa < keep:
            need = keep - oa
            take = min(sa, need)
            sa -= take
            oa += take
        if sa > 0:
            ra += sa
            moved_capital += sa
            sa = 0.0
        excess_oa = max(0.0, oa - keep)
        if excess_oa > 0:
            ra += excess_oa
            moved_capital += excess_oa
            oa -= excess_oa
        return sa, oa, ra, moved_capital

    # Case 2: at/above FRS
    needed = max(0.0, transfer_target - ra)
    if needed <= 0:
        oa += sa
        sa = 0.0
        return sa, oa, ra, moved_capital

    take_sa = min(needed, sa)
    ra += take_sa
    sa -= take_sa
    moved_capital += take_sa
    needed -= take_sa

    if needed > 0:
        take_oa = min(needed, oa)
        ra += take_oa
        oa -= take_oa
        moved_capital += take_oa

    oa += sa
    sa = 0.0
    return sa, oa, ra, moved_capital


# ==============================
# IP helpers (CSV schema)
# ==============================
REQUIRED_IP_COLS = ["Insurer", "Ward Class", "Plan Type", "Plan Name", "Age", "Premium in MA", "Premium in Cash"]

def try_load_ip_csv(uploaded_file):
    """
    Try to load from uploaded file first, else from local ./ip_premiums.csv, else /mnt/data/ip_premiums.csv.
    Returns (df or None, error_message or None)
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df, None
        except Exception as e:
            return None, f"Could not read the uploaded CSV: {e}"
    for p in ["ip_premiums.csv", "/mnt/data/ip_premiums.csv"]:
        try:
            df = pd.read_csv(p)
            return df, None
        except Exception:
            continue
    return None, "ip_premiums.csv not found (search paths: app folder or /mnt/data). Upload it to enable IP."

def validate_ip_df(df):
    missing = [c for c in REQUIRED_IP_COLS if c not in df.columns]
    if missing:
        return f"ip_premiums.csv is missing required columns: {missing}"
    return None

def _ip_lookup_amounts(ip_df, insurer, ward_class, plan_name, plan_type, anb):
    """Return (ma_amount, cash_amount) for given selection and ANB. 0 if not found."""
    if ip_df is None or any(x in (None, "", "(None)") for x in [insurer, ward_class, plan_name]):
        return 0.0, 0.0
    df = ip_df[
        (ip_df["Insurer"] == insurer) &
        (ip_df["Ward Class"] == ward_class) &
        (ip_df["Plan Type"] == plan_type) &
        (ip_df["Plan Name"] == plan_name) &
        (ip_df["Age"].astype(int) == int(anb))
    ]
    if df.empty:
        return 0.0, 0.0
    row = df.iloc[0]
    ma = float(row.get("Premium in MA", 0.0))
    cash = float(row.get("Premium in Cash", 0.0))
    return ma, cash


# ==============================
# Core projection
# ==============================
def project(
    name: str,
    dob_str: str,
    gender: str,
    start_year: int,
    years: int,
    monthly_income: float,
    annual_bonus: float,
    salary_growth_pct: float,
    bonus_growth_pct: float,
    opening_balances: dict,
    frs_growth_pct: float,
    bhs_growth_pct: float,
    ers_factor: float = ERS_FACTOR_DEFAULT,
    retirement_age: int = 65,
    # CPF LIFE controls
    include_cpf_life: bool = True,
    cpf_life_plan: str = "Standard",  # "Standard","Escalating","Basic"
    payout_start_age: int = 65,
    m_topup_month: int = 1,
    topup_OA: float = 0.0,
    topup_SA_RA: float = 0.0,
    topup_MA: float = 0.0,
    
    # Top-up stop controls
    topup_stop_option: str = "No limit",
    topup_years_limit: int = 0,
    topup_stop_age: int = 120,
    # Lump-sum top-up (one-time)
    lump_enabled: bool = False,
    lump_year: int = 0,
    lump_month: int = 0,
    lump_OA: float = 0.0,
    lump_SA_RA: float = 0.0,
    lump_MA: float = 0.0,

    # Long-term care insurance (deduct from MA)
    include_ltci: bool = False,
    ltci_ma_premium: float = 0.0,
    ltci_pay_until_age: int = 67,
    ltci_month: int = 1,
    # Integrated Shield Plan
    ip_enabled: bool = False,
    ip_df: pd.DataFrame | None = None,
    ip_insurer: str | None = None,
    ip_ward: str | None = None,
    ip_base_plan: str | None = "(None)",
    ip_rider: str | None = "(None)",
    insurance_month: int = 1,   # <-- single month for MSHL & IP
    # OA withdrawal (A)
    withdraw_oa_enabled: bool = False,
    withdraw_oa_monthly_amount: float = 0.0,
    withdraw_oa_start_age: int = 55,
    withdraw_oa_end_age: int = 120,
    inflation_pct: float = 0.02,
    withdraw_oa_inflate: bool = False,
    withdraw_oa_monthly_today: float = 0.0,
    oa_withdrawal_fv_at_start_year: float = 0.0,

    # OA withdrawal (B)  
    withdraw_oa2_enabled: bool = False,
    withdraw_oa2_monthly_amount: float = 0.0,
    withdraw_oa2_start_age: int = 55,
    withdraw_oa2_end_age: int = 120,
    withdraw_oa2_inflate: bool = False,
    withdraw_oa2_monthly_today: float = 0.0,
    oa_withdrawal2_fv_at_start_year: float = 0.0,

    # Housing loan (OA monthly)
    house_enabled: bool = False,
    house_monthly_amount: float = 0.0,
    house_end_age: int = 120,
    opening_ra_capital: float | None = None,
):
    dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
    bal = opening_balances.copy()
    # Use the FV computed in the UI (at first withdrawal birthday)
    oa_wd_base_at_start_year  = float(oa_withdrawal_fv_at_start_year or 0.0)
    oa_wd2_base_at_start_year = float(oa_withdrawal2_fv_at_start_year or 0.0) 


    monthly_rows = []
    year55 = dob.year + 55
    cohort_frs = get_frs_for_cohort(year55, frs_growth_pct)
    cohort_ers = cohort_frs * 2
    desired_ra_opening_multiple = ers_factor
    ra_transfer_target = min(cohort_frs * desired_ra_opening_multiple, cohort_ers)

    # Track capital and CPF LIFE
    # If user provided an explicit RA capital (for users already ≥55), use it; else infer from opening RA
    ra_capital = (
        float(opening_ra_capital)
        if (opening_ra_capital is not None and opening_ra_capital > 0)
        else float(bal.get("RA", 0.0))
    )

    prev_bal_for_interest = bal.copy()

    # CPF LIFE derived values
    ra_at_65_value = None
    premium_pool = 0.0
    ra_savings_for_basic = 0.0
    cpf_life_started = False
    monthly_start_payout = None
    psa_month = dob.month  # CPF LIFE start in birth month of chosen age

    # OA run-out trackers
    oa_runs_out_age = None
    oa_runs_out_year = None
    oa_runs_out_month = None

    house_runs_out_age = None
    house_runs_out_year = None
    house_runs_out_month = None

    start_year_sched = None

    # Sweep control (perform once, right after LIFE starts)
    did_ra_interest_sweep_at_start = False

    for year in range(start_year, start_year + years):
        yr_index = year - start_year
        monthly_income_y = monthly_income * ((1 + salary_growth_pct) ** yr_index)
        annual_bonus_y = annual_bonus * ((1 + bonus_growth_pct) ** yr_index)

        bhs_this_year = get_bhs_for_year_with_cohort(dob, year, bhs_growth_pct)
        ow_cap = ow_ceiling_monthly(year)

        prevailing_frs_year = get_frs_for_year(year, frs_growth_pct)
        prevailing_ers_year = prevailing_frs_year * 2

        ow_subject_per_mo = min(monthly_income_y, ow_cap)
        ow_used_ytd = 0.0  # for bonus cap
        
        # ---- Interest accrual buckets (reset each calendar year) ----
        accr_base_to_OA = 0.0
        accr_base_to_SA = 0.0
        accr_base_to_MA = 0.0

        # Split RA-directed credits so we can route correctly post-LIFE
        accr_base_to_RA_from_RA = 0.0   # base interest earned by RA itself
        accr_base_to_RA_from_SA = 0.0   # base interest earned by SA (routed to RA ≥55)

        accr_extra_to_SA = 0.0
        accr_extra_to_MA = 0.0

        # extra interest routed to RA, by source (≥55)
        accr_extra_to_RA_from_RA = 0.0
        accr_extra_to_RA_from_OA = 0.0
        accr_extra_to_RA_from_SA = 0.0

        for month in range(1, 13):
            # --- init per-month tracking vars (top-ups & lump-sum) ---
            # Regular top-ups
            topup_oa_applied = topup_sa_applied = topup_ra_applied = topup_ma_applied = 0.0
            topup_oa_requested = topup_sa_requested = topup_ra_requested = topup_ma_requested = 0.0  # (kept for symmetry)

            # Lump-sum top-ups
            lump_oa_applied = lump_sa_applied = lump_ra_applied = lump_ma_applied = 0.0
            lump_oa_requested = lump_sa_requested = lump_ra_requested = lump_ma_requested = 0.0
            lump_attempted_this_month = False

            # “room” snapshots (so they always exist even when not the lump month)
            lump_ma_room = 0.0
            lump_sa_room = 0.0
            lump_ra_room = 0.0
            age = age_at(dob, year, month)
            alloc = get_alloc_for_age(age)

            # Snapshot RA at exact age-65 birth month (before any CPF LIFE premium deduction)
            if (age == 65) and (month == dob.month) and (ra_at_65_value is None):
                ra_at_65_value = bal["RA"]

            # RA transfer at 55 birth month
            if age >= 55 and bal["SA"] > 0:
                if (year > year55) or (year == year55 and month >= dob.month):
                    bal["SA"], bal["OA"], bal["RA"], moved = transfer_to_ra_at_55(
                        age, bal["SA"], bal["OA"], bal["RA"], ra_transfer_target, cohort_frs
                    )
                    ra_capital += moved

            # CPF LIFE: initialise payouts & deduct premium at start month
            if include_cpf_life and (age == payout_start_age) and (month == psa_month) and not cpf_life_started:
                if ra_at_65_value is None:
                    ra_at_65_value = bal["RA"]
                coeff_key = "M" if gender == "M" else "F"
                plan_map = {"Standard":"Standard_65","Escalating":"Escalating_65","Basic":"Basic_65"}
                coeff = CPF_LIFE_COEFFS[coeff_key][plan_map[cpf_life_plan]]
                payout_65 = coeff["a"] + coeff["b"] * ra_at_65_value
                monthly_start_payout = payout_65 * ((1 + CPF_LIFE_DEFERRAL_PER_YEAR) ** (payout_start_age - 65))

                # Premium deduction (moves RA to annuity pool; ra_capital stays unchanged)
                if cpf_life_plan in ("Standard","Escalating"):
                    premium = bal["RA"]
                    premium_pool += premium
                    bal["RA"] = 0.0
                else:  # Basic
                    premium = bal["RA"] * BASIC_PREMIUM_FRAC
                    premium_pool += premium
                    bal["RA"] -= premium
                    ra_savings_for_basic = bal["RA"]

                cpf_life_started = True
                start_year_sched = year

                # --------- RA interest sweep RIGHT AFTER start (Std/Esc only) ---------
                if cpf_life_plan in ("Standard","Escalating") and not did_ra_interest_sweep_at_start:
                    # Pending RA-directed interest accrued YTD
                    ra_due_from_buckets = (
                        accr_base_to_RA_from_RA
                        + accr_base_to_RA_from_SA
                        + accr_extra_to_RA_from_RA
                        + accr_extra_to_RA_from_OA
                        + accr_extra_to_RA_from_SA
                    )

                    # Pending MA interest YTD that would spill to RA if credited now
                    pending_ma_interest = accr_base_to_MA + accr_extra_to_MA
                    ma0, sa0, oa0, ra0 = bal["MA"], bal["SA"], bal["OA"], bal["RA"]
                    ma_sim_end = ma0 + pending_ma_interest
                    _ma, _sa, _oa, ra_after = spill_from_ma(
                        age=age,
                        ma_end=ma_sim_end,
                        bhs=bhs_this_year,
                        sa=sa0,
                        oa=oa0,
                        ra=ra0,
                        frs_for_cohort=cohort_frs,
                        ra_capital=ra_capital
                    )
                    ra_due_from_ma_spill = max(0.0, ra_after - ra0)

                    sweep_amount = ra_due_from_buckets + ra_due_from_ma_spill
                    if sweep_amount > 0:
                        premium_pool += sweep_amount

                    # Clear RA-directed buckets so they won't later credit to member
                    accr_base_to_RA_from_RA = 0.0
                    accr_base_to_RA_from_SA = 0.0
                    accr_extra_to_RA_from_RA = 0.0
                    accr_extra_to_RA_from_OA = 0.0
                    accr_extra_to_RA_from_SA = 0.0

                    did_ra_interest_sweep_at_start = True
                # ---------------------------------------------------------------------

            # --- Monthly OW contributions ---
            working = age < retirement_age
            if working:
                to_MA = ow_subject_per_mo * alloc["MA"]
                to_SA_RA = ow_subject_per_mo * alloc["SA_RA"]
                to_OA = ow_subject_per_mo * alloc["OA"]
            else:
                to_MA = to_SA_RA = to_OA = 0.0

            if age >= 55:
                space_ra_cap = max(0.0, cohort_frs - ra_capital)
                to_RA = min(to_SA_RA, space_ra_cap)
                to_OA += (to_SA_RA - to_RA)
                to_SA = 0.0
            else:
                to_SA = to_SA_RA
                to_RA = 0.0

            bal["MA"] += to_MA
            bal["OA"] += to_OA
            bal["SA"] += to_SA
            bal["RA"] += to_RA
            if age >= 55:
                ra_capital += to_RA  # contributions to RA are capital

            income_used_this_month = ow_subject_per_mo if working else 0.0
            ow_used_ytd += income_used_this_month

            # --- Annual bonus in December (AW) ---
            aw_used_this_dec = 0.0
            if month == 12 and working and (annual_bonus_y > 0):
                aw_ceiling_rem = max(0.0, ANNUAL_TW_CEILING - ow_used_ytd)
                aw_subject = min(annual_bonus_y, aw_ceiling_rem)
                aw_used_this_dec = aw_subject
                to_MA_aw = aw_subject * alloc["MA"]
                to_SA_RA_aw = aw_subject * alloc["SA_RA"]
                to_OA_aw = aw_subject * alloc["OA"]
                if age >= 55:
                    space_ra2_cap = max(0.0, cohort_frs - ra_capital)
                    to_RA_aw = min(to_SA_RA_aw, space_ra2_cap)
                    to_OA_aw += (to_SA_RA_aw - to_RA_aw)
                    to_SA_aw = 0.0
                else:
                    to_SA_aw = to_SA_RA_aw
                    to_RA_aw = 0.0
                bal["MA"] += to_MA_aw
                bal["OA"] += to_OA_aw
                bal["SA"] += to_SA_aw
                bal["RA"] += to_RA_aw
                if age >= 55:
                    ra_capital += to_RA_aw

            # --- MediShield Life premium (Age Next Birthday) ---
            mshl_paid_this_month = 0.0
            mshl_nominal_this_month = 0.0
            if month == insurance_month:
                anb = age + 1
                prem = get_mshl_premium_by_anb(anb)
                mshl_nominal_this_month = prem
                pay = min(bal["MA"], prem)
                bal["MA"] -= pay
                mshl_paid_this_month = pay

            # --- Long-term care insurance premium (from MA) ---
            ltci_paid_this_month = 0.0
            if include_ltci and (month == ltci_month) and (age <= ltci_pay_until_age):
                ltci_paid_this_month = min(bal["MA"], float(ltci_ma_premium))
                bal["MA"] -= ltci_paid_this_month

            # --- Integrated Shield Plan premiums (CSV) ---
            ip_base_ma_paid = 0.0
            ip_base_cash = 0.0
            ip_rider_cash = 0.0
            ip_base_ma_nominal = 0.0
            ip_base_cash_nominal = 0.0
            ip_rider_cash_nominal = 0.0

            if ip_enabled and (ip_df is not None) and (month == insurance_month):
                anb_ip = age + 1

                # Base plan (MA + Cash)
                if ip_base_plan and ip_base_plan != "(None)":
                    base_ma, base_cash = _ip_lookup_amounts(
                        ip_df, ip_insurer, ip_ward, ip_base_plan, "Base", anb_ip
                    )
                    ip_base_ma_nominal = base_ma
                    ip_base_cash_nominal = base_cash

                    if base_ma > 0:
                        pay_ma = min(bal["MA"], base_ma)
                        bal["MA"] -= pay_ma
                        ip_base_ma_paid = pay_ma
                        ip_base_cash += max(0.0, base_ma - pay_ma)  # MA shortfall paid in cash
                    ip_base_cash += base_cash  # plan's cash part

                # Rider (cash only)
                if ip_rider and ip_rider != "(None)":
                    _, rider_cash = _ip_lookup_amounts(
                        ip_df, ip_insurer, ip_ward, ip_rider, "Rider", anb_ip
                    )
                    ip_rider_cash_nominal = rider_cash
                    ip_rider_cash += rider_cash


            if month == m_topup_month:
                allow_topup = True
                if topup_stop_option == "After N years":
                    if (year - start_year) >= int(topup_years_limit):
                        allow_topup = False
                elif topup_stop_option == "After age X":
                    if age > int(topup_stop_age):
                        allow_topup = False

                if allow_topup:
                    if float(topup_OA) > 0.0:
                        add = float(topup_OA)
                        bal["OA"] += add
                        topup_oa_applied = add

                    if float(topup_MA) > 0.0:
                        room_ma = max(0.0, bhs_this_year - bal["MA"])
                        add = min(float(topup_MA), room_ma)
                        if add > 0:
                            bal["MA"] += add
                            topup_ma_applied = add

                    if float(topup_SA_RA) > 0.0:
                        if age < 55:
                            room_sa = max(0.0, cohort_frs - bal["SA"])
                            add = min(float(topup_SA_RA), room_sa)
                            if add > 0:
                                bal["SA"] += add
                                topup_sa_applied = add
                        else:
                            room_ra_capital = max(0.0, prevailing_ers_year - ra_capital)
                            add = min(float(topup_SA_RA), room_ra_capital)
                            if add > 0:
                                bal["RA"] += add
                                ra_capital += add
                                topup_ra_applied = add

                                
                                

            if lump_enabled and (year == int(lump_year)) and (month == int(lump_month)):
                lump_attempted_this_month = True

                # Compute "room" first (pre-application)
                lump_ma_room = max(0.0, bhs_this_year - bal["MA"])
                if age < 55:
                    lump_sa_room = max(0.0, cohort_frs - bal["SA"])
                else:
                    lump_ra_room = max(0.0, prevailing_ers_year - ra_capital)

                # OA: no cap
                if float(lump_OA) > 0.0:
                    lump_oa_requested = float(lump_OA)
                    bal["OA"] += lump_oa_requested
                    lump_oa_applied = lump_oa_requested

                # MA: cap by BHS
                if float(lump_MA) > 0.0:
                    lump_ma_requested = float(lump_MA)
                    add = min(lump_ma_requested, lump_ma_room)
                    if add > 0:
                        bal["MA"] += add
                        lump_ma_applied = add

                # SA/RA: SA <55 up to cohort FRS; RA ≥55 up to prevailing ERS by *capital*
                if float(lump_SA_RA) > 0.0:
                    req = float(lump_SA_RA)
                    if age < 55:
                        lump_sa_requested = req
                        add = min(req, lump_sa_room)
                        if add > 0:
                            bal["SA"] += add
                            lump_sa_applied = add
                    else:
                        lump_ra_requested = req
                        add = min(req, lump_ra_room)
                        if add > 0:
                            bal["RA"] += add
                            ra_capital += add
                            lump_ra_applied = add

            # --- Monthly OA withdrawal (55+) : TWO schedules ---
            oa_withdrawal1_paid = 0.0
            oa_withdrawal2_paid = 0.0

            # Helper to compute the *this-month* amount for an inflation-adjusted schedule
            def _inflated_amt_at(year, month, start_age, base_fv_at_start):
                # Birthday-stepped escalation. First withdrawal month occurs at DOB month of start-age.
                withdraw_start_year = dob.year + int(start_age)
                years_since_first_withdraw_bday = (year - withdraw_start_year) - (1 if month < dob.month else 0)
                years_since_first_withdraw_bday = max(0, years_since_first_withdraw_bday)
                return base_fv_at_start * ((1 + inflation_pct) ** years_since_first_withdraw_bday)

            # Schedule A
            if withdraw_oa_enabled and (age >= max(55, int(withdraw_oa_start_age))) and (age <= int(withdraw_oa_end_age)):
                if withdraw_oa_inflate and (oa_wd_base_at_start_year > 0):
                    amt1 = _inflated_amt_at(year, month, withdraw_oa_start_age, oa_wd_base_at_start_year)
                else:
                    amt1 = float(withdraw_oa_monthly_amount)
                if amt1 > 0.0:
                    pay1 = min(bal["OA"], amt1)
                    bal["OA"] -= pay1
                    oa_withdrawal1_paid = pay1
                    if (oa_runs_out_age is None) and (bal["OA"] <= 1e-6):
                        oa_runs_out_age = int(age)
                        oa_runs_out_year = int(year)
                        oa_runs_out_month = int(month)


            # Schedule B (applies on the post-Schedule-A OA balance)
            if withdraw_oa2_enabled and (age >= max(55, int(withdraw_oa2_start_age))) and (age <= int(withdraw_oa2_end_age)):
                if withdraw_oa2_inflate and (oa_wd2_base_at_start_year > 0):
                    amt2 = _inflated_amt_at(year, month, withdraw_oa2_start_age, oa_wd2_base_at_start_year)
                else:
                    amt2 = float(withdraw_oa2_monthly_amount)
                if amt2 > 0.0:
                    pay2 = min(bal["OA"], amt2)
                    bal["OA"] -= pay2
                    oa_withdrawal2_paid = pay2
                    if (oa_runs_out_age is None) and (bal["OA"] <= 1e-6):
                        oa_runs_out_age = int(age)
                        oa_runs_out_year = int(year)
                        oa_runs_out_month = int(month)

            # Total OA withdrawal this month (for charts/tables)
            oa_withdrawal_paid = oa_withdrawal1_paid + oa_withdrawal2_paid


            # --- Monthly housing loan deduction from OA (simple, up to end age) ---
            house_paid_this_month = 0.0
            if (
                house_enabled
                and (age <= int(house_end_age))
                and float(house_monthly_amount) > 0.0
            ):
                pay = min(bal["OA"], float(house_monthly_amount))
                bal["OA"] -= pay
                house_paid_this_month = pay

                # First time OA hits zero during housing period
                if (house_runs_out_age is None) and (bal["OA"] <= 1e-6):
                    house_runs_out_age = int(age)
                    house_runs_out_year = int(year)
                    house_runs_out_month = int(month)

            # --- Monthly interest ACCRUAL (computed on previous month-end balances) ---
            base_int_OA = prev_bal_for_interest.get("OA", 0.0) * (BASE_INT["OA"] / 12.0)
            base_int_SA_raw = prev_bal_for_interest.get("SA", 0.0) * (BASE_INT["SA"] / 12.0)
            base_int_MA = prev_bal_for_interest.get("MA", 0.0) * (BASE_INT["MA"] / 12.0)
            base_int_RA = prev_bal_for_interest.get("RA", 0.0) * (BASE_INT["RA"] / 12.0)  # always accrue RA interest notionally

            # Route base-interest ACCRUAL to buckets (not credited yet)
            accr_base_to_OA += base_int_OA
            if age < 55:
                accr_base_to_SA += base_int_SA_raw
            else:
                # After 55, SA base interest is treated as RA-directed
                accr_base_to_RA_from_SA += base_int_SA_raw
            accr_base_to_MA += base_int_MA
            # RA base interest is RA-directed
            accr_base_to_RA_from_RA += base_int_RA

            # Extra-interest ACCRUAL on prev month-end balances (single rule)
            ei = compute_extra_interest_distribution(
                age,
                prev_bal_for_interest.get("OA", 0.0),
                prev_bal_for_interest.get("SA", 0.0),
                prev_bal_for_interest.get("MA", 0.0),
                prev_bal_for_interest.get("RA", 0.0),
            )

            if age < 55:
                # OA+SA extra -> SA; MA extra -> MA
                accr_extra_to_SA += (ei["OA"] / 12.0) + (ei["SA"] / 12.0)
                accr_extra_to_MA += (ei["MA"] / 12.0)
                # (Any RA extra when <55 is zero in allocator)
            else:
                post_life = (include_cpf_life and cpf_life_started)
                # OA extra: before LIFE -> RA; after LIFE (any plan) -> pool
                if post_life:
                    premium_pool += (ei["OA"] / 12.0)
                else:
                    accr_extra_to_RA_from_OA += (ei["OA"] / 12.0)
                # SA & RA extra are RA-directed
                accr_extra_to_RA_from_SA += (ei["SA"] / 12.0)
                accr_extra_to_RA_from_RA += (ei["RA"] / 12.0)
                # MA extra stays with MA
                accr_extra_to_MA += (ei["MA"] / 12.0)

            # --- CPF LIFE payouts (Basic only draws from RA savings) ---
            monthly_cpf_payout = 0.0
            if include_cpf_life and cpf_life_started:
                years_since_start = (year - start_year_sched) if start_year_sched is not None else 0
                current_monthly_payout = monthly_start_payout
                if cpf_life_plan == "Escalating" and years_since_start > 0:
                    current_monthly_payout *= ((1 + ESCALATING_RATE) ** years_since_start)
                monthly_cpf_payout = current_monthly_payout
                if cpf_life_plan == "Basic":
                    draw = min(bal["RA"], monthly_cpf_payout)
                    bal["RA"] -= draw
                    # (ra_capital unchanged by draw)

            # --- Enforce BHS and spillovers ---
            ra_before = bal["RA"]
            bal["MA"], bal["SA"], bal["OA"], bal["RA"] = spill_from_ma(
                age, bal["MA"], bhs_this_year, bal["SA"], bal["OA"], bal["RA"], cohort_frs, ra_capital
            )
            ra_spill = max(0.0, bal["RA"] - ra_before)
            ra_capital += ra_spill  # MA->RA spill counts as capital

            # Ensure SA closed after 55
            if age >= 55 and bal["SA"] > 0:
                bal["RA"] += bal["SA"]; bal["SA"] = 0.0
                # (moving SA->RA here does not add to ra_capital; only transfers/contri/spills do)


            # --- Save monthly row OR (in December) save it AFTER year-end interest credit ---

            # We will credit interest on 31 Dec so Dec ending balances include the year's interest.
            is_december = (month == 12)

            if is_december:
                # ---- CREDIT all accrued interest ON 31 DEC (so Dec ending balances include interest) ----
                # 1) Credit interest for THIS calendar year into member/pool as per rules
                bal["OA"] += accr_base_to_OA
                bal["SA"] += accr_base_to_SA + accr_extra_to_SA
                bal["MA"] += accr_base_to_MA + accr_extra_to_MA

                ra_dir_interest = (
                    accr_base_to_RA_from_RA
                    + accr_base_to_RA_from_SA
                    + accr_extra_to_RA_from_RA
                    + accr_extra_to_RA_from_SA
                    + accr_extra_to_RA_from_OA
                )

                post_life_std_esc = (
                    include_cpf_life and cpf_life_started and (cpf_life_plan in ("Standard", "Escalating"))
                )
                if post_life_std_esc:
                    # After LIFE (Std/Esc): ALL RA-directed interest goes to pool (not to member's RA)
                    premium_pool += ra_dir_interest
                else:
                    # Before LIFE or Basic plan: credit to RA
                    bal["RA"] += ra_dir_interest

                # 2) Apply spillovers USING THIS YEAR'S LIMITS (credit belongs to this year)
                ra_before_dec = bal["RA"]
                bal["MA"], bal["SA"], bal["OA"], bal["RA"] = spill_from_ma(
                    age, bal["MA"], bhs_this_year, bal["SA"], bal["OA"], bal["RA"], cohort_frs, ra_capital
                )
                ra_spill_dec = max(0.0, bal["RA"] - ra_before_dec)
                ra_capital += ra_spill_dec  # MA->RA spill counts as capital

                # Ensure SA is closed after 55 (in case credit landed anything in SA)
                if age >= 55 and bal["SA"] > 0:
                    bal["RA"] += bal["SA"]; bal["SA"] = 0.0
                    # (SA->RA here is not capital)

                # 3) Reset accrual buckets for the new year
                accr_base_to_OA = accr_base_to_SA = accr_base_to_MA = 0.0
                accr_base_to_RA_from_RA = accr_base_to_RA_from_SA = 0.0
                accr_extra_to_SA = accr_extra_to_MA = 0.0
                accr_extra_to_RA_from_RA = accr_extra_to_RA_from_OA = accr_extra_to_RA_from_SA = 0.0

            # Now record the month (Dec row will already include the credited interest)
            monthly_rows.append({
                "Year": year, "Month": month, "Age": age,
                "BHS": bhs_this_year, "FRS_cohort": cohort_frs, "ERS_cohort": cohort_ers,
                "RA_target55_multiple": ers_factor, "OW_cap": ow_cap,
                "Income_used": income_used_this_month, "Bonus_used_dec": aw_used_this_dec,
                "OA": bal["OA"], "SA": bal["SA"], "MA": bal["MA"], "RA": bal["RA"],
                "BaseInt_OA": base_int_OA, "BaseInt_SA": base_int_SA_raw, "BaseInt_MA": base_int_MA, "BaseInt_RA": base_int_RA,
                "ExtraInt_OA": ei["OA"]/12.0, "ExtraInt_SA": ei["SA"]/12.0, "ExtraInt_MA": ei["MA"]/12.0, "ExtraInt_RA": ei["RA"]/12.0,
                "RA_capital": ra_capital, "Prevailing_ERS": prevailing_ers_year,
                "CPF_LIFE_started": int(cpf_life_started), "CPF_LIFE_monthly_payout": monthly_cpf_payout,

                # Insurance flows
                "MSHL_Premium_Paid": mshl_paid_this_month,
                "MSHL_Premium_Nominal": mshl_nominal_this_month,

                # LTC from MA (deducted from MA but NOT plotted in the health insurance chart)
                "LTCI_MA_Premium_Paid": ltci_paid_this_month,

                # IP flows
                "IP_Base_MA_Paid": ip_base_ma_paid,
                "IP_Base_Cash": ip_base_cash,
                "IP_Rider_Cash": ip_rider_cash,

                # Nominal (for stacked chart by intended source — LTC intentionally excluded)
                "IP_Base_MA_Nominal": ip_base_ma_nominal,
                "IP_Base_Cash_Nominal": ip_base_cash_nominal,
                "IP_Rider_Cash_Nominal": ip_rider_cash_nominal,

                # Top-ups actually applied this month
                "Topup_OA_Applied": topup_oa_applied,
                "Topup_SA_Applied": topup_sa_applied,
                "Topup_RA_Applied": topup_ra_applied,
                "Topup_MA_Applied": topup_ma_applied,
                
                # Lump-sum actually applied this month
                "Lump_OA_Applied": lump_oa_applied,
                "Lump_SA_Applied": lump_sa_applied,
                "Lump_RA_Applied": lump_ra_applied,
                "Lump_MA_Applied": lump_ma_applied,
                "Lump_Attempted": int(lump_attempted_this_month),
                
                # Requested vs room (for precise warnings)
                "Lump_OA_Requested": lump_oa_requested,
                "Lump_SA_Requested": lump_sa_requested,
                "Lump_RA_Requested": lump_ra_requested,
                "Lump_MA_Requested": lump_ma_requested,
                "Lump_SA_Room": 0.0 if lump_sa_room is None else float(lump_sa_room),
                "Lump_RA_Room": 0.0 if lump_ra_room is None else float(lump_ra_room),
                "Lump_MA_Room": 0.0 if lump_ma_room is None else float(lump_ma_room),


                # Withdrawals / Housing 
                "OA_Withdrawal_Paid": oa_withdrawal_paid,
                "OA_Withdrawal1_Paid": oa_withdrawal1_paid,
                "OA_Withdrawal2_Paid": oa_withdrawal2_paid,
                "Housing_OA_Paid": house_paid_this_month,

            })

            # Set next month's accrual base to month-end balances
            # (post-credit balances for December)
            prev_bal_for_interest = {
                "OA": bal["OA"], "SA": bal["SA"], "MA": bal["MA"], "RA": bal["RA"]
            }


            
           

    monthly_df = pd.DataFrame(monthly_rows)

    # ----- Build yearly roll-up -----
    yearly = []
    for y, grp in monthly_df.groupby("Year"):
        end_row = grp.sort_values("Month").iloc[-1]
        total_base_int = grp[["BaseInt_OA", "BaseInt_SA", "BaseInt_MA", "BaseInt_RA"]].sum().sum()
        total_extra_int = grp[["ExtraInt_OA", "ExtraInt_SA", "ExtraInt_MA", "ExtraInt_RA"]].sum().sum()
        yearly.append({
            "Year": y, "Age_end": int(end_row["Age"]),
            "End_OA": end_row["OA"], "End_SA": end_row["SA"], "End_MA": end_row["MA"], "End_RA": end_row["RA"],
            "RA_capital_end": float(end_row["RA_capital"]), "Prevailing_ERS": float(end_row["Prevailing_ERS"]),
            "Total_Base_Interest": total_base_int, "Total_Extra_Interest": total_extra_int,
            "OW_subject_total": grp["Income_used"].sum(), "AW_subject_total": grp["Bonus_used_dec"].sum(),
            "CPF_LIFE_Annual_Payout": grp["CPF_LIFE_monthly_payout"].sum(),

            # Insurance annuals (actual paid)
            "MSHL_Annual": grp["MSHL_Premium_Paid"].sum(),
            "LTCI_Premium_Annual_MA": grp["LTCI_MA_Premium_Paid"].sum(),
            "IP_Base_MA_Annual": grp["IP_Base_MA_Paid"].sum(),
            "IP_Base_Cash_Annual": grp["IP_Base_Cash"].sum(),
            "IP_Rider_Cash_Annual": grp["IP_Rider_Cash"].sum(),

            # Nominal (for chart)
            "MSHL_Nominal": grp["MSHL_Premium_Nominal"].sum(),
            "IP_Base_MA_Nominal_Annual": grp["IP_Base_MA_Nominal"].sum(),
            "IP_Base_Cash_Nominal_Annual": grp["IP_Base_Cash_Nominal"].sum(),
            "IP_Rider_Cash_Nominal_Annual": grp["IP_Rider_Cash_Nominal"].sum(),

            # Top-up annual totals
            "Topup_OA_Annual": grp["Topup_OA_Applied"].sum(),
            "Topup_SA_Annual": grp["Topup_SA_Applied"].sum(),
            "Topup_RA_Annual": grp["Topup_RA_Applied"].sum(),
            "Topup_MA_Annual": grp["Topup_MA_Applied"].sum(),
            
            # Lump-sum annual totals
            "Lump_OA_Annual": grp["Lump_OA_Applied"].sum(),
            "Lump_SA_Annual": grp["Lump_SA_Applied"].sum(),
            "Lump_RA_Annual": grp["Lump_RA_Applied"].sum(),
            "Lump_MA_Annual": grp["Lump_MA_Applied"].sum(),


            # OA withdrawal + Housing annual totals
            "OA_Withdrawal_Annual": (grp["OA_Withdrawal1_Paid"].sum() + grp["OA_Withdrawal2_Paid"].sum()),
            "OA_Withdrawal_A_Annual": grp["OA_Withdrawal1_Paid"].sum(),
            "OA_Withdrawal_B_Annual": grp["OA_Withdrawal2_Paid"].sum(),

            "Housing_OA_Annual": grp["Housing_OA_Paid"].sum(),
        })
    yearly_df = pd.DataFrame(yearly)

    # ----- CPF LIFE payout schedule & bequest track -----
    cpf_life_df = None
    bequest_df = None
    if include_cpf_life and ('monthly_start_payout' in locals()) and (monthly_start_payout is not None):
        # Determine start year calendar
        psa_rows = monthly_df[(monthly_df["Age"] == payout_start_age) & (monthly_df["Month"] == psa_month)]
        if not psa_rows.empty:
            start_year_sched = int(psa_rows.iloc[0]["Year"])
        else:
            tmp = yearly_df[yearly_df["Age_end"] >= payout_start_age]
            start_year_sched = int(tmp.iloc[0]["Year"]) if not tmp.empty else int(yearly_df["Year"].min())

        sched = []
        beq_rows = []

        beq_premium = float(premium_pool)  # premium (no credited interest)
        beq_ra_savings = float(ra_savings_for_basic if cpf_life_plan == "Basic" else 0.0)

        def _annual_ra_interest(balance: float) -> float:
            if balance <= 0:
                return 0.0
            base = balance * 0.04
            tier1 = min(balance, 30000.0) * 0.02
            tier2 = min(max(balance - 30000.0, 0.0), 30000.0) * 0.01
            return base + tier1 + tier2

        last_year = int(yearly_df["Year"].max())

        for y in range(start_year_sched, last_year + 1):
            years_since = y - start_year_sched
            monthly = monthly_start_payout
            if cpf_life_plan == "Escalating":
                monthly *= ((1 + ESCALATING_RATE) ** max(0, years_since))
            annual = monthly * 12.0

            sched.append({"Year": y, "Monthly_Payout": monthly, "Annual_Payout": annual})

            if cpf_life_plan == "Basic":
                # Basic: RA savings keep earning to member
                beq_ra_savings += _annual_ra_interest(beq_ra_savings)
                draw_from_ra = min(beq_ra_savings, annual)
                beq_ra_savings -= draw_from_ra
                from_pool = max(0.0, annual - draw_from_ra)
                beq_premium = max(0.0, beq_premium - from_pool)
            else:
                # Std/Esc: payouts entirely from pooled premium (no interest added)
                beq_premium = max(0.0, beq_premium - annual)

            beq_rows.append({
                "Year": y,
                "Bequest_Remaining": beq_premium + beq_ra_savings,
                "Unused_Premium": beq_premium,
                "RA_Savings_Remaining": beq_ra_savings
            })

        cpf_life_df = pd.DataFrame(sched)
        bequest_df = pd.DataFrame(beq_rows)

    # meta & OA/housing run-out info
    meta = {
        "monthly_start_payout": monthly_start_payout,
        # OA withdrawals (summary for ribbons)
        "oa_withdrawal_enabled": bool(
            (withdraw_oa_enabled and (
                withdraw_oa_monthly_amount > 0.0 or
                (withdraw_oa_inflate and (oa_wd_base_at_start_year or 0.0) > 0.0)
             ))
            or
            (withdraw_oa2_enabled and (
                withdraw_oa2_monthly_amount > 0.0 or
                (withdraw_oa2_inflate and (oa_wd2_base_at_start_year or 0.0) > 0.0)
             ))
        ),

        # A
        "oaA_enabled": bool(withdraw_oa_enabled),
        "oaA_inflate": bool(withdraw_oa_inflate),
        "oaA_amount": float(withdraw_oa_monthly_amount),
        "oaA_start_age": int(withdraw_oa_start_age),
        "oaA_end_age": int(withdraw_oa_end_age),
        "oaA_fv_start_year": float(oa_wd_base_at_start_year),

        # B
        "oaB_enabled": bool(withdraw_oa2_enabled),
        "oaB_inflate": bool(withdraw_oa2_inflate),
        "oaB_amount": float(withdraw_oa2_monthly_amount),
        "oaB_start_age": int(withdraw_oa2_start_age),
        "oaB_end_age": int(withdraw_oa2_end_age),
        "oaB_fv_start_year": float(oa_wd2_base_at_start_year),
        
        "oa_runs_out_age": oa_runs_out_age,
        "oa_runs_out_year": oa_runs_out_year,
        "oa_runs_out_month": oa_runs_out_month,



        "house_enabled": bool(house_enabled and house_monthly_amount > 0),
        "house_end_age": int(house_end_age),
        "house_runs_out_age": house_runs_out_age,
        "house_runs_out_year": house_runs_out_year,
        "house_runs_out_month": house_runs_out_month,
        "house_amount": float(house_monthly_amount),
        
        "inflation_pct": float(inflation_pct),
    }
    return monthly_df, yearly_df, cohort_frs, cohort_ers, meta, cpf_life_df, bequest_df

# ==== Helpers for milestones, PV, and planning ====
def find_first_hit(df: pd.DataFrame, col: str, target_series_or_value) -> tuple | None:
    """
    If target_series_or_value is a scalar -> compare df[col] >= scalar.
    If it's a Series in df (e.g., 'BHS') -> compare row-wise: df[col] >= df[target].
    Returns (year, month, age) or None.
    """
    tol = 1e-6
    if isinstance(target_series_or_value, (int, float)):
        mask = df[col] >= (float(target_series_or_value) - tol)
    else:
        # assume column name
        tcol = str(target_series_or_value)
        mask = df[col] >= (df[tcol] - tol)
    hit = df[mask].sort_values(["Year", "Month"]).head(1)
    if hit.empty:
        return None
    r = hit.iloc[0]
    return int(r["Year"]), int(r["Month"]), int(r["Age"])

def years_months_from_start(start_year: int, y: int, m: int) -> int:
    """Months from projection start to (y,m). Month is 1-based."""
    return (y - start_year) * 12 + (m - 1)


# ==============================
# Sidebar Inputs
# ==============================
with st.sidebar:
    st.header("Inputs")
    name = st.text_input("Name", value="Member")
    dob = st.date_input("Date of birth", value=date(1980,1,1), min_value=date(1960,1,1), max_value=date.today(), format="DD-MM-YYYY")
    gender = st.selectbox("Gender", ["M", "F"], index=1)

    start_year = st.number_input("Start year", min_value=2000, max_value=2100, value=date.today().year, step=1)
    years = st.slider("Number of years to project from start year", min_value=5, max_value=100, value=60, step=1)

    monthly_income = st.number_input("Monthly income (gross)", min_value=0.0, value=6000.0, step=100.0, format="%.2f")
    annual_bonus = st.number_input("Annual bonus (gross)", min_value=0.0, value=6000.0, step=500.0, format="%.2f")
    salary_growth_pct = st.number_input("Salary growth % p.a.", min_value=0.0, max_value=20.0, value=3.0, step=1.0, format="%.1f")/100
    bonus_growth_pct = st.number_input("Bonus growth % p.a.", min_value=0.0, max_value=20.0, value=3.0, step=1.0, format="%.1f")/100
    retirement_age = st.number_input("Retirement age (stop working contributions)", min_value=40, max_value=80, value=60, step=1, help="From this birthday onward, monthly salary and bonus contributions stop.")

    st.subheader("Opening balances")
    col1, col2 = st.columns(2)
    with col1:
        opening_OA = st.number_input("OA", min_value=0.0, value=30000.0, step=100.0, format="%.2f")
        opening_SA = st.number_input("SA", min_value=0.0, value=100000.0, step=100.0, format="%.2f")
    with col2:
        opening_MA = st.number_input("MA", min_value=0.0, value=75000.0, step=100.0, format="%.2f")
        opening_RA = st.number_input("RA", min_value=0.0, value=0.0, step=100.0, format="%.2f")

    # --- RA Savings (Capital) at Start (for users already ≥55) ---
    st.subheader("RA Savings (capital) at start (optional)")
    st.caption("If you’re already 55+, enter your RA *capital* (excluding interest) as shown in your CPF portal. "
               "Leave as 0 if you’re below 55 or you’re unsure.")

    opening_ra_capital = st.number_input(
        "RA capital at start (S$)", min_value=0.0, value=0.0, step=100.0, format="%.2f"
    )
    # Compute current age (year-based; no birthday precision)
    _today = date.today()
    current_age_years = _today.year - dob.year

    # Friendly hints / validations
    if opening_ra_capital > 0 and current_age_years < 55:
        st.info(
            "You entered an RA **capital** value but you appear to be **below 55**. "
            "RA capital matters only once RA is formed at 55."
        )

    # If the user is already 55+ but left capital at 0 (and has an RA balance), nudge them
    if current_age_years >= 55 and opening_ra_capital == 0.0 and opening_RA > 0.0:
        st.info(
            "You’re **55+** with a non-zero RA opening balance. "
            "Consider entering your **RA capital** (from CPF portal) so caps & limits use the correct figure."
        )

    # If capital exceeds the RA balance, warn about a likely mismatch
    if opening_ra_capital > opening_RA + 1e-6:
        st.warning(
            "Your **RA capital at start** is **greater** than your RA opening balance. "
            "Please double-check both values."
        )

    # If a capital was entered but RA balance is zero, flag it
    if opening_ra_capital > 0.0 and opening_RA == 0.0 and current_age_years >= 55:
        st.warning(
            "You entered a positive **RA capital** but your **RA opening balance** is zero. "
            "If RA is already formed, you probably need to fill in the RA opening balance too."
        )

    with st.expander("Advanced assumptions", expanded=False):
        frs_growth_pct = st.number_input("FRS growth after last known year (default 3.5%)", min_value=0.0, max_value=10.0, value=FRS_growth_pct_default, step=0.5, format="%.1f")/100
        bhs_growth_pct = st.number_input("BHS growth after 2025 (default 5%)", min_value=0.0, max_value=10.0, value=BHS_growth_pct_default, step=0.5, format="%.1f")/100
        st.caption("You can adjust FRS/BHS growth to reflect future policy changes.")
        ers_factor = st.number_input("Desired RA opening amount (×FRS)", min_value=1.0, max_value=2.0, value=1.0, step=0.05, help="Target RA amount at age 55 as a multiple of FRS (1× to 2×).")
        st.caption("Choose your desired RA opening target between 1× and 2× FRS.")

    st.subheader("Global Assumptions")
    inflation_pct = st.number_input(
    "Inflation % p.a.",
    min_value=0.0, max_value=10.0, value=2.0, step=0.5, format="%.1f"
) / 100
    
    # --- FV Calculator---
    st.markdown("---")
    st.subheader("FV Calculator")
    st.caption("Enter today’s amount and the age you want it at. Uses current year vs the year you turn that age.")

    _today = date.today()
    # Year-based age (ignores birthday)
    _age_now_years = _today.year - dob.year

    fv_present_amount = st.number_input(
        "Present amount (S$)", min_value=0.0, value=0.0, step=50.0, format="%.2f"
    )
    fv_target_age = st.number_input(
        "Target age", min_value=int(_age_now_years), max_value=120,
        value=max(60, int(_age_now_years)), step=1
    )

    # Year you turn the target age, minus current calendar year
    _target_year = dob.year + int(fv_target_age)
    _years_diff = max(0, _target_year - _today.year)

    fv_result = fv_present_amount * ((1 + float(inflation_pct)) ** _years_diff)

    st.metric("Future value (S$)", f"{fv_result:,.2f}",
              help=f"Inflated for {_years_diff} year(s) (from {_today.year} to {_target_year}).")

    
    
with st.expander("Regular top-ups", expanded=False):
    st.caption("Amounts apply once every calendar year in the selected month. SA top-ups only when SA < FRS (cohort). RA top-ups allowed up to prevailing ERS, based on capital (excludes RA interest).")
    colA, colB = st.columns(2)
    with colA:
        m_topup_month = st.selectbox("Month to apply top-ups", list(range(1,13)), index=0, help="1=Jan ... 12=Dec")
        topup_OA = st.number_input("Top-up to OA (yearly)", min_value=0.0, value=0.0, step=100.0)
        topup_SA_RA = st.number_input("Top-up to SA (if <55) / RA (if ≥55) (yearly)", min_value=0.0, value=0.0, step=100.0)
        topup_MA = st.number_input("Top-up to MA (yearly)", min_value=0.0, value=0.0, step=100.0)

    st.markdown("---")
    topup_stop_option = st.selectbox("Top-up stop option", ["No limit", "After N years", "After age X"], index=0)
    colY, colZ = st.columns(2)
    with colY:
        if topup_stop_option == "After N years":
            topup_years_limit = st.number_input("Number of years to top-up", min_value=1, max_value=60, value=10, step=1)
        else:
            topup_years_limit = 0
    with colZ:
        if topup_stop_option == "After age X":
            topup_stop_age = st.number_input("Stop top-ups from this age (no top-ups when age > this)", min_value=30, max_value=100, value=65, step=1)
        else:
            topup_stop_age = 120
            
with st.expander("Lump sum top-ups (one-time)", expanded=False):
    lump_enabled = st.checkbox("Enable a one-time lump-sum top-up", value=False)
    colL1, colL2 = st.columns(2)
    with colL1:
        lump_year = st.number_input(
            "Top-up year", min_value=2000, max_value=2100, value=int(start_year), step=1,
            disabled=not lump_enabled
        )
    with colL2:
        lump_month = st.selectbox(
            "Top-up month (1–12)", list(range(1, 13)), index=0, disabled=not lump_enabled
        )
    st.caption("Amounts below will be attempted only once at the selected year & month, "
               "and are still subject to FRS/ERS/BHS limits at that time.")
    colL3, colL4, colL5 = st.columns(3)
    with colL3:
        lump_OA = st.number_input("Lump-sum to OA", min_value=0.0, value=0.0, step=100.0, disabled=not lump_enabled)
    with colL4:
        lump_SA_RA = st.number_input("Lump-sum to SA (<55) / RA (≥55)", min_value=0.0, value=0.0, step=100.0, disabled=not lump_enabled)
    with colL5:
        lump_MA = st.number_input("Lump-sum to MA", min_value=0.0, value=0.0, step=100.0, disabled=not lump_enabled)


with st.expander("Long-term care insurance (paid from MA)", expanded=False):
    include_ltci = st.checkbox("I have long-term care insurance and pay premiums from MA", value=False)
    col_lt1, col_lt2, col_lt3 = st.columns(3)
    with col_lt1:
        ltci_month = st.selectbox("Month to deduct", list(range(1,13)), index=0, disabled=not include_ltci)
    with col_lt2:
        ltci_ma_premium = st.number_input("MA used per year (S$)", min_value=0.0, value=0.0, step=10.0, disabled=not include_ltci)
    with col_lt3:
        ltci_pay_until_age = st.number_input("Premium payable up to age", min_value=30, max_value=120, value=67, step=1, disabled=not include_ltci)

# ---- Integrated Shield Plan (IP) + MSHL month unified ----
with st.expander("Health Insurance (MSHL + Integrated Shield)", expanded=False):
    insurance_month = st.selectbox("Month to deduct health insurance premiums (MSHL & IP)", list(range(1,13)), index=0)

    ip_enabled = st.checkbox("Include Integrated Shield Plan premiums", value=False)
    ip_upload = st.file_uploader("Upload IP premiums CSV", type=["csv"], help="Headers required: Insurer, Ward Class, Plan Type, Plan Name, Age, Premium in MA, Premium in Cash")
    ip_df, ip_load_msg = (None, None)
    ip_insurer = ip_ward = ip_base_plan = ip_rider = None

    if ip_enabled:
        ip_df, ip_load_msg = try_load_ip_csv(ip_upload)
        if ip_df is None:
            st.warning(ip_load_msg)
        else:
            err = validate_ip_df(ip_df)
            if err:
                st.error(err)
                ip_df = None
            else:
                try:
                    ip_df["Age"] = ip_df["Age"].astype(int)
                except Exception:
                    st.error("Column 'Age' must be integer values.")
                    ip_df = None

        if ip_df is not None:
            insurers = sorted(ip_df["Insurer"].dropna().unique().tolist())
            ip_insurer = st.selectbox("Insurer", insurers, index=0 if insurers else 0)

            wards = sorted(ip_df[ip_df["Insurer"] == ip_insurer]["Ward Class"].dropna().unique().tolist())
            ip_ward = st.selectbox("Ward Class", wards, index=0 if wards else 0)

            base_opts = sorted(
                ip_df[
                    (ip_df["Insurer"] == ip_insurer) &
                    (ip_df["Ward Class"] == ip_ward) &
                    (ip_df["Plan Type"] == "Base")
                ]["Plan Name"].dropna().unique().tolist()
            )
            base_opts = ["(None)"] + base_opts
            ip_base_plan = st.selectbox("Base Plan", base_opts, index=0)

            rider_opts = sorted(
                ip_df[
                    (ip_df["Insurer"] == ip_insurer) &
                    (ip_df["Ward Class"] == ip_ward) &
                    (ip_df["Plan Type"] == "Rider")
                ]["Plan Name"].dropna().unique().tolist()
            )
            rider_opts = ["(None)"] + rider_opts
            ip_rider = st.selectbox("Rider", rider_opts, index=0)

            
# ---- Housing loan (OA monthly) ----
with st.expander("Housing loan (OA monthly)", expanded=False):
    house_enabled = st.checkbox("Deduct housing repayment from OA every month", value=False)
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        house_monthly_amount = st.number_input("Monthly OA deduction (S$)", min_value=0.0, value=0.0, step=50.0, format="%.2f", disabled=not house_enabled)
    with col_h2:
        house_end_age = st.number_input("End age (stop housing deduction after this age)", min_value=18, max_value=100, value=60, step=1, disabled=not house_enabled)

with st.expander("OA Withdrawal (55+)", expanded=False):
    st.caption("Set up to two separate monthly withdrawals. Both deduct from OA each month")

    # --- Schedule A ---
    st.markdown("**Schedule A**")
    withdraw_oa_enabled = st.checkbox("Enable A", value=False)

    withdraw_oa_inflate = st.checkbox(
        "Inflation-adjust A",
        value=False, disabled=not withdraw_oa_enabled
    )

    colA1, colA2, colA3 = st.columns(3)
    with colA1:
        withdraw_oa_monthly_amount = st.number_input(
            "A: Monthly amount (S$)",
            min_value=0.0, value=0.0, step=50.0, format="%.2f",
            disabled=not withdraw_oa_enabled
        )
    with colA2:
        withdraw_oa_start_age = st.number_input(
            "A: Start age (≥55)", min_value=55, max_value=120, value=60, step=1,
            disabled=not withdraw_oa_enabled
        )
    with colA3:
        withdraw_oa_end_age = st.number_input(
            "A: End age", min_value=55, max_value=120, value=90, step=1,
            disabled=not withdraw_oa_enabled
        )

    st.markdown("---")

    # --- Schedule B ---
    st.markdown("**Schedule B**")
    withdraw_oa2_enabled = st.checkbox("Enable B", value=False)

    withdraw_oa2_inflate = st.checkbox(
        "Inflation-adjust B",
        value=False, disabled=not withdraw_oa2_enabled
    )

    colB1, colB2, colB3 = st.columns(3)
    with colB1:
        withdraw_oa2_monthly_amount = st.number_input(
            "B: Monthly amount (S$)",
            min_value=0.0, value=0.0, step=50.0, format="%.2f",
            disabled=not withdraw_oa2_enabled
        )
    with colB2:
        withdraw_oa2_start_age = st.number_input(
            "B: Start age (≥55)", min_value=55, max_value=120, value=65, step=1,
            disabled=not withdraw_oa2_enabled
        )
    with colB3:
        withdraw_oa2_end_age = st.number_input(
            "B: End age", min_value=55, max_value=120, value=95, step=1,
            disabled=not withdraw_oa2_enabled
        )

# Map UI to projection parameters:

# Schedule A
if withdraw_oa_enabled and withdraw_oa_inflate:
    oa_withdrawal_fv_at_start_year = float(withdraw_oa_monthly_amount)
    withdraw_oa_monthly_amount_to_pass = 0.0
else:
    oa_withdrawal_fv_at_start_year = 0.0
    withdraw_oa_monthly_amount_to_pass = float(withdraw_oa_monthly_amount)

withdraw_oa_monthly_today = 0.0  # (not used in current calc)

# Schedule B (NEW)
if withdraw_oa2_enabled and withdraw_oa2_inflate:
    oa_withdrawal2_fv_at_start_year = float(withdraw_oa2_monthly_amount)
    withdraw_oa2_monthly_amount_to_pass = 0.0
else:
    oa_withdrawal2_fv_at_start_year = 0.0
    withdraw_oa2_monthly_amount_to_pass = float(withdraw_oa2_monthly_amount)

withdraw_oa2_monthly_today = 0.0




with st.expander("CPF LIFE (payouts)", expanded=False):
    include_cpf_life = st.checkbox("Include CPF LIFE payouts in projection", value=True)
    cpf_life_plan = st.selectbox("Plan", ["Standard","Escalating","Basic"], index=0)
    payout_start_age = st.number_input("Payout start age (65–70)", min_value=65, max_value=70, value=65, step=1)

def _fingerprint_ip_df(ip_df):
    # Tiny, stable fingerprint for cache key; safe if ip_df is None
    if ip_df is None:
        return None
    try:
        return int(pd.util.hash_pandas_object(ip_df, index=False).sum())
    except Exception:
        return None

# bundle everything that actually affects the projection
_input_key = {
    "name": name,
    "dob": dob.isoformat(),
    "gender": gender,
    "start_year": int(start_year),
    "years": int(years),
    "monthly_income": float(monthly_income),
    "annual_bonus": float(annual_bonus),
    "salary_growth_pct": float(salary_growth_pct),
    "bonus_growth_pct": float(bonus_growth_pct),
    "opening": {
        "OA": float(opening_OA), "SA": float(opening_SA),
        "MA": float(opening_MA), "RA": float(opening_RA),
        "RA_capital": float(opening_ra_capital),
    },
    "frs_growth_pct": float(frs_growth_pct),
    "bhs_growth_pct": float(bhs_growth_pct),
    "ers_factor": float(ers_factor),
    "retirement_age": int(retirement_age),

    # top-ups (regular + stop rules)
    "m_topup_month": int(m_topup_month),
    "topup_OA": float(topup_OA),
    "topup_SA_RA": float(topup_SA_RA),
    "topup_MA": float(topup_MA),
    "topup_stop_option": str(topup_stop_option),
    "topup_years_limit": int(topup_years_limit),
    "topup_stop_age": int(topup_stop_age),

    # lump
    "lump_enabled": bool(lump_enabled),
    "lump_year": int(lump_year),
    "lump_month": int(lump_month),
    "lump_OA": float(lump_OA),
    "lump_SA_RA": float(lump_SA_RA),
    "lump_MA": float(lump_MA),

    # insurance
    "include_ltci": bool(include_ltci),
    "ltci_ma_premium": float(ltci_ma_premium),
    "ltci_pay_until_age": int(ltci_pay_until_age),
    "ltci_month": int(ltci_month),

    "ip_enabled": bool(ip_enabled),
    "ip_sig": _fingerprint_ip_df(ip_df if ip_enabled else None),
    "ip_insurer": None if not ip_enabled else str(ip_insurer),
    "ip_ward": None if not ip_enabled else str(ip_ward),
    "ip_base_plan": None if not ip_enabled else str(ip_base_plan),
    "ip_rider": None if not ip_enabled else str(ip_rider),
    "insurance_month": int(insurance_month),

    # OA withdrawal
    "withdraw_oa_enabled": bool(withdraw_oa_enabled),
    "withdraw_oa_monthly_amount_to_pass": float(withdraw_oa_monthly_amount_to_pass),
    "withdraw_oa_start_age": int(withdraw_oa_start_age),
    "withdraw_oa_end_age": int(withdraw_oa_end_age),
    "withdraw_oa_inflate": bool(withdraw_oa_inflate),
    "inflation_pct": float(inflation_pct),
    "oa_withdrawal_fv_at_start_year": float(oa_withdrawal_fv_at_start_year),

    # OA withdrawal B
    "withdraw_oa2_enabled": bool(withdraw_oa2_enabled),
    "withdraw_oa2_monthly_amount_to_pass": float(withdraw_oa2_monthly_amount_to_pass),
    "withdraw_oa2_start_age": int(withdraw_oa2_start_age),
    "withdraw_oa2_end_age": int(withdraw_oa2_end_age),
    "withdraw_oa2_inflate": bool(withdraw_oa2_inflate),
    "oa_withdrawal2_fv_at_start_year": float(oa_withdrawal2_fv_at_start_year),

    # housing
    "house_enabled": bool(house_enabled),
    "house_monthly_amount": float(house_monthly_amount),
    "house_end_age": int(house_end_age),

    # CPF LIFE
    "include_cpf_life": bool(include_cpf_life),
    "cpf_life_plan": str(cpf_life_plan),
    "payout_start_age": int(payout_start_age),
}

_params_hash = hashlib.sha1(
    json.dumps(_input_key, sort_keys=True, default=str).encode("utf-8")
).hexdigest()
    
run_btn = st.button("Run Projection", type="primary", use_container_width=True)

# ==============================
# Main
# ==============================
st.title("CPF Projector")
st.markdown("Project CPF balances year-by-year with clear charts and downloadable tables.")
st.markdown('<div class="small-muted">Assumes current CPF rules in 2025; edit assumptions in the sidebar.</div>', unsafe_allow_html=True)

# recompute only if the user clicks the button
if run_btn:
    opening_balances = {"OA": opening_OA, "SA": opening_SA, "MA": opening_MA, "RA": opening_RA}
    st.session_state.proj_results = project(
        name=name,
        dob_str=dob.strftime("%Y-%m-%d"),
        gender=gender,
        start_year=int(start_year),
        years=int(years),
        monthly_income=float(monthly_income),
        annual_bonus=float(annual_bonus),
        salary_growth_pct=float(salary_growth_pct),
        bonus_growth_pct=float(bonus_growth_pct),
        opening_balances=opening_balances,
        frs_growth_pct=float(frs_growth_pct),
        bhs_growth_pct=float(bhs_growth_pct),
        ers_factor=float(ers_factor),
        retirement_age=int(retirement_age),

        include_cpf_life=bool(include_cpf_life),
        cpf_life_plan=str(cpf_life_plan),
        payout_start_age=int(payout_start_age),

        # top-ups (explicit params)
        m_topup_month=int(m_topup_month),
        topup_OA=float(topup_OA),
        topup_SA_RA=float(topup_SA_RA),
        topup_MA=float(topup_MA),

        topup_stop_option=str(topup_stop_option),
        topup_years_limit=int(topup_years_limit),
        topup_stop_age=int(topup_stop_age),

        # lump
        lump_enabled=bool(lump_enabled),
        lump_year=int(lump_year),
        lump_month=int(lump_month),
        lump_OA=float(lump_OA),
        lump_SA_RA=float(lump_SA_RA),
        lump_MA=float(lump_MA),

        # insurance
        include_ltci=bool(include_ltci),
        ltci_ma_premium=float(ltci_ma_premium),
        ltci_pay_until_age=int(ltci_pay_until_age),
        ltci_month=int(ltci_month),

        ip_enabled=bool(ip_enabled),
        ip_df=ip_df if ip_enabled else None,
        ip_insurer=ip_insurer,
        ip_ward=ip_ward,
        ip_base_plan=ip_base_plan,
        ip_rider=ip_rider,
        insurance_month=int(insurance_month),

        # OA withdrawal (A)
        withdraw_oa_enabled=bool(withdraw_oa_enabled),
        withdraw_oa_monthly_amount=float(withdraw_oa_monthly_amount_to_pass),
        withdraw_oa_start_age=int(withdraw_oa_start_age),
        withdraw_oa_end_age=int(withdraw_oa_end_age),
        inflation_pct=float(inflation_pct),
        withdraw_oa_inflate=bool(withdraw_oa_inflate),
        withdraw_oa_monthly_today=float(withdraw_oa_monthly_today),
        oa_withdrawal_fv_at_start_year=float(oa_withdrawal_fv_at_start_year),

        # OA withdrawal (B)
        withdraw_oa2_enabled=bool(withdraw_oa2_enabled),
        withdraw_oa2_monthly_amount=float(withdraw_oa2_monthly_amount_to_pass),
        withdraw_oa2_start_age=int(withdraw_oa2_start_age),
        withdraw_oa2_end_age=int(withdraw_oa2_end_age),
        withdraw_oa2_inflate=bool(withdraw_oa2_inflate),
        withdraw_oa2_monthly_today=float(withdraw_oa2_monthly_today),
        oa_withdrawal2_fv_at_start_year=float(oa_withdrawal2_fv_at_start_year),


        # housing
        house_enabled=bool(house_enabled),
        house_monthly_amount=float(house_monthly_amount),
        house_end_age=int(house_end_age),

        opening_ra_capital=float(opening_ra_capital),
    )
    st.session_state.params_hash = _params_hash
    st.session_state.ran_once = True

# ---- After the `if run_btn:` block, add this ----
settings_changed = (
    st.session_state.get("ran_once", False)
    and st.session_state.get("proj_results") is not None
    and st.session_state.get("params_hash") is not None
    and _params_hash != st.session_state.params_hash
)

if settings_changed:
    # Nice little badge + explicit warning
    st.markdown(
        "<div class='pill' style='background:#fff7ed;color:#9a3412;"
        "border:1px solid #fed7aa;'>Inputs changed since last run</div>",
        unsafe_allow_html=True
    )
    st.warning(
        "You’ve changed inputs since the last run. Click **Run Projection** to refresh the charts & tables."
    )
    
    
# use cached results if available
if not st.session_state.ran_once or st.session_state.proj_results is None:
    st.info("Set your inputs in the sidebar and click **Run Projection**.")
    st.stop()

# unpack cached results for the rest of the page
monthly_df, yearly_df, cohort_frs, cohort_ers, meta, cpf_life_df, bequest_df = st.session_state.proj_results

# ---------- (1) Milestones + (2) Time-to-target ----------
# SA reaches cohort FRS (scalar)
sa_hit = find_first_hit(monthly_df, "SA", cohort_frs)


chips = []
months_to_sa = None

if sa_hit:
    y, m, a = sa_hit
    chips.append(f"SA reached cohort FRS: <b>{y}-{m:02d}</b> (Age {a})")
    months_to_sa = years_months_from_start(start_year, y, m)

if chips:
    st.markdown(
        "<div class='ribbon-row'>" + " ".join([f"<span class='pill'>{c}</span>" for c in chips]) + "</div>",
        unsafe_allow_html=True
    )

col_tt1 = st.columns(1)[0]
with col_tt1:
    if months_to_sa is None:
        st.metric("Time to SA = FRS", "Not within horizon")
    else:
        st.metric("Time to SA = FRS", f"{months_to_sa} months", delta=("Achieved" if months_to_sa == 0 else None))


# ---------- (11) Policy-limit warnings ----------
warns = []

# Helper: unique int years as CSV string
def _years_csv(series):
    yrs = sorted(set(int(y) for y in series.astype(int).tolist()))
    return ", ".join(str(y) for y in yrs)

# A) SA top-up blocked (attempted before 55 but SA already ≥ cohort FRS at the top-up month)
if float(topup_SA_RA) > 0:
    is_topup_month = monthly_df["Month"] == int(m_topup_month)
    pre55 = monthly_df["Age"] < 55
    sa_roomless = (monthly_df["SA"] >= (monthly_df["FRS_cohort"] - 1e-6))
    sa_topup_attempts = monthly_df[is_topup_month & pre55]

    if not sa_topup_attempts.empty:
        sa_blocked = sa_topup_attempts[
            (sa_topup_attempts["Topup_SA_Applied"] <= 1e-6) & sa_roomless
        ]
        if not sa_blocked.empty:
            yrs = _years_csv(sa_blocked["Year"])
            warns.append(f"SA top-up blocked in: {yrs} (SA ≥ cohort FRS at your top-up month).")

# B) RA top-up blocked (attempted after 55 but RA capital already ≥ Prevailing ERS at the top-up month)
if float(topup_SA_RA) > 0:
    is_topup_month = monthly_df["Month"] == int(m_topup_month)
    post55 = monthly_df["Age"] >= 55
    ra_roomless = (monthly_df["RA_capital"] >= (monthly_df["Prevailing_ERS"] - 1e-6))
    ra_topup_attempts = monthly_df[is_topup_month & post55]

    if not ra_topup_attempts.empty:
        ra_blocked = ra_topup_attempts[
            (ra_topup_attempts["Topup_RA_Applied"] <= 1e-6) & ra_roomless
        ]
        if not ra_blocked.empty:
            yrs = _years_csv(ra_blocked["Year"])
            warns.append(f"RA top-up blocked in: {yrs} (RA capital ≥ Prevailing ERS at your top-up month).")

# C) Lump-sum exceeds allowed space (warn even if partially applied)
tol = 1e-6
if lump_enabled and (float(lump_OA) > 0 or float(lump_SA_RA) > 0 or float(lump_MA) > 0):
    lump_mask = (monthly_df["Year"] == int(lump_year)) & (monthly_df["Month"] == int(lump_month))
    if not monthly_df[lump_mask].empty:
        r = monthly_df[lump_mask].iloc[0]

        # Option A: keep the symbol, escape it so Markdown won’t start LaTeX
        def _money(x): return f"S\\${x:,.0f}"

        # MA vs BHS
        ma_req  = float(r.get("Lump_MA_Requested", float(lump_MA)))
        # try stored room first; fallback computes from row values if not present
        ma_room = float(r.get("Lump_MA_Room", max(0.0, float(r["BHS"]) - float(r["MA"]))))
        ma_app  = float(r.get("Lump_MA_Applied", 0.0))
        if ma_req > ma_room + tol:
            warns.append(
                f"Lump-sum MA top-up exceeds BHS space in {int(lump_year)}-{int(lump_month):02d}: "
                f"requested {_money(ma_req)}, applied {_money(ma_app)}."
            )

        if r["Age"] < 55:
            # SA vs cohort FRS
            sa_req  = float(r.get("Lump_SA_Requested", float(lump_SA_RA)))
            sa_room = float(r.get("Lump_SA_Room", max(0.0, float(r["FRS_cohort"]) - float(r["SA"]))))
            sa_app  = float(r.get("Lump_SA_Applied", 0.0))
            if sa_req > sa_room + tol:
                warns.append(
                    f"Lump-sum SA top-up exceeds SA room (cohort FRS) in {int(lump_year)}-{int(lump_month):02d}: "
                    f"requested {_money(sa_req)}, applied {_money(sa_app)}."
                )
        else:
            # RA vs Prevailing ERS (by capital)
            ra_req  = float(r.get("Lump_RA_Requested", float(lump_SA_RA)))
            ra_room = float(r.get("Lump_RA_Room", max(0.0, float(r["Prevailing_ERS"]) - float(r["RA_capital"]))))
            ra_app  = float(r.get("Lump_RA_Applied", 0.0))
            if ra_req > ra_room + tol:
                warns.append(
                    f"Lump-sum RA top-up exceeds RA capital room (Prevailing ERS) in {int(lump_year)}-{int(lump_month):02d}: "
                    f"requested {_money(ra_req)}, applied {_money(ra_app)}."
                )

for w in warns:
    st.warning(w)

# Banner ribbons
ribbons = []
if meta.get("house_enabled"):
    ribbons.append(
        f"Housing deduction ACTIVE — ${meta['house_amount']:,.0f}/mo, "
        f"till Age {meta['house_end_age']}"
    )

if meta.get("oaA_enabled"):
    if meta.get("oaA_inflate"):
        ribbons.append(
            f"OA withdrawal A - ${meta['oaA_fv_start_year']:,.0f}/mo, "
            f"Age {meta['oaA_start_age']}–{meta['oaA_end_age']} @ {meta['inflation_pct']*100:.1f}% pa"
        )
    else:
        ribbons.append(
            f"OA withdrawal A — ${meta['oaA_amount']:,.0f}/mo, "
            f"Age {meta['oaA_start_age']}–{meta['oaA_end_age']}"
        )

if meta.get("oaB_enabled"):
    if meta.get("oaB_inflate"):
        ribbons.append(
            f"OA withdrawal B - ${meta['oaB_fv_start_year']:,.0f}/mo, "
            f"Age {meta['oaB_start_age']}–{meta['oaB_end_age']} @ {meta['inflation_pct']*100:.1f}% pa"
        )
    else:
        ribbons.append(
            f"OA withdrawal B — ${meta['oaB_amount']:,.0f}/mo, "
            f"Age {meta['oaB_start_age']}–{meta['oaB_end_age']}"
        )


if ribbons:
    html_ribbons = " ".join([f"<span class='pill'>{r}</span>" for r in ribbons])
    st.markdown(f"<div class='ribbon-row'>{html_ribbons}</div>", unsafe_allow_html=True)



# Warnings
if meta.get("house_enabled") and (meta.get("house_runs_out_age") is not None) and (meta["house_runs_out_age"] < meta["house_end_age"]):
    st.warning(
        f"OA runs out at age {meta['house_runs_out_age']} "
        f"(Year {meta['house_runs_out_year']}) before your housing end age {meta['house_end_age']}."
    )

if meta.get("oa_withdrawal_enabled") and (meta.get("oa_runs_out_age") is not None):
    st.warning(
        f"OA runs out at age {meta['oa_runs_out_age']} "
        f"(Year {meta['oa_runs_out_year']}) under your OA withdrawal settings."
    )



# ─────────────────────────────────────────────────────────
# Helper: Top-up Allowance Preview (SA / BHS / ERS)
# ─────────────────────────────────────────────────────────
st.markdown("### Top-up Allowance Preview (SA / BHS / ERS)")
with st.expander("Preview allowance at a specific month", expanded=False):
    years_available = sorted(monthly_df["Year"].unique().tolist())
    _today = date.today()

    # Clamp today's year into projection range
    if _today.year < years_available[0]:
        default_year = years_available[0]
    elif _today.year > years_available[-1]:
        default_year = years_available[-1]
    else:
        default_year = _today.year
    default_month = _today.month if default_year == _today.year else 1

    # Always show selectors (no toggle)
    sel_year = st.selectbox("Year", years_available, index=years_available.index(default_year))
    sel_month = st.selectbox("Month", list(range(1, 13)), index=default_month - 1)

    # Row for selected Year+Month
    _row = monthly_df[(monthly_df["Year"] == int(sel_year)) & (monthly_df["Month"] == int(sel_month))].iloc[0]
    _age = int(_row["Age"])

    # Compute rooms
    sa_room = max(0.0, float(_row["FRS_cohort"]) - float(_row["SA"])) if _age < 55 else None
    ma_room = max(0.0, float(_row["BHS"]) - float(_row["MA"]))  # BHS room for MA
    ers_room = max(0.0, float(_row["Prevailing_ERS"]) - float(_row["RA_capital"])) if _age >= 55 else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "SA room (to cohort FRS)" if _age < 55 else "SA room (n/a ≥55)",
            f"${sa_room:,.0f}" if sa_room is not None else "—"
        )
    with c2:
        st.metric("MA room (to BHS)", f"${ma_room:,.0f}")
    with c3:
        st.metric(
            "RA capital room (to ERS)" if _age >= 55 else "ERS room (n/a <55)",
            f"${ers_room:,.0f}" if ers_room is not None else "—"
        )

    # Context / policy snapshot
    st.markdown(
        f"<div class='small-muted'>"
        f"Selected: <b>{sel_year}-{sel_month:02d}</b> (Age {_age}) · "
        f"FRS (cohort): <b>${float(_row['FRS_cohort']):,.0f}</b> · "
        f"Prevailing ERS: <b>${float(_row['Prevailing_ERS']):,.0f}</b> · "
        f"BHS for year: <b>${float(_row['BHS']):,.0f}</b>"
        f"</div>",
        unsafe_allow_html=True
    )

    # If this is exactly the configured lump month, also show recorded rooms
    if bool(lump_enabled) and (int(sel_year) == int(lump_year)) and (int(sel_month) == int(lump_month)):
        rec_sa = float(_row.get("Lump_SA_Room", 0.0))
        rec_ra = float(_row.get("Lump_RA_Room", 0.0))
        rec_ma = float(_row.get("Lump_MA_Room", 0.0))
        st.caption(
            f"Recorded room at your lump month → "
            f"SA: ${rec_sa:,.0f} · RA (ERS): ${rec_ra:,.0f} · MA (BHS): ${rec_ma:,.0f}"
        )

# ---- Tabs layout ----
tab_bal, tab_cpf, tab_cash, tab_health = st.tabs([
    "Account Balances",
    "CPF LIFE & Bequest",
    "Cashflows",
    "Health Insurance"
])

# ============= TAB 1: Account Balances =============
with tab_bal:

    # KPI Cards
    end_row = yearly_df.sort_values("Year").iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "OA (final)", end_row["End_OA"]),
        (c2, "SA (final)", end_row["End_SA"]),
        (c3, "MA (final)", end_row["End_MA"]),
        (c4, "RA (final)", end_row["End_RA"]),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="small-muted">{label}</div><div style="font-size:24px;font-weight:700;">${val:,.0f}</div></div>', unsafe_allow_html=True)

    # Yearly stacked balances with FRS line
    st.markdown("### Yearly Balances (Stacked)")
    yearly_long = yearly_df.melt(id_vars=['Year','Age_end'], value_vars=['End_OA','End_SA','End_MA','End_RA'], var_name='Account', value_name='Balance')
    yearly_long['Account'] = yearly_long['Account'].replace({'End_OA':'OA','End_SA':'SA','End_MA':'MA','End_RA':'RA'})
    yearly_long['YearAge'] = yearly_long.apply(lambda r: f"{int(r['Year'])} (Age {int(r['Age_end'])})", axis=1)
#    stacked = alt.Chart(yearly_long).mark_bar().encode(
#        x=alt.X('YearAge:O', title='Year (Age)', sort=None),
#        y=alt.Y('sum(Balance):Q', title='Balance (S$)'),
#        color=alt.Color('Account:N', legend=alt.Legend(title='Account')),
#        tooltip=['Year','Age_end','Account','Balance']
#    ).properties(height=360)

    stacked = alt.Chart(yearly_long).mark_bar().encode(
        x=alt.X('YearAge:O', title='Year (Age)', sort=None),
        y=alt.Y('sum(Balance):Q', title='Balance (S$)', axis=alt.Axis(format=',.0f')),  # y-axis no decimals
        color=alt.Color('Account:N', legend=alt.Legend(title='Account')),
        tooltip=[
            alt.Tooltip('Year:O', title='Year'),
            alt.Tooltip('Age_end:Q', title='Age_end'),
            alt.Tooltip('Account:N', title='Account'),
            alt.Tooltip('Balance:Q', title='Balance', format=',.0f')  # tooltip no decimals
        ]
    ).properties(height=360)


    frs_rule = alt.Chart(pd.DataFrame({'y': [cohort_frs]})).mark_rule(strokeDash=[6,3]).encode(
        y='y:Q', tooltip=[alt.Tooltip('y:Q', title='Cohort FRS')]
    )
    st.altair_chart(stacked + frs_rule, use_container_width=True)

    # Yearly table
    st.markdown("### Yearly Summary Table")
    yearly_df['Year (Age)'] = yearly_df.apply(lambda r: f"{int(r['Year'])} (Age {int(r['Age_end'])})", axis=1)
    cols = ['Year (Age)', 'Year', 'Age_end', 'End_OA','End_SA','End_MA','End_RA',
            'RA_capital_end','Prevailing_ERS','Total_Base_Interest','Total_Extra_Interest',
            'OW_subject_total','AW_subject_total','CPF_LIFE_Annual_Payout',
            'MSHL_Annual','LTCI_Premium_Annual_MA','IP_Base_MA_Annual','IP_Base_Cash_Annual','IP_Rider_Cash_Annual',
            'Housing_OA_Annual',
            'Topup_OA_Annual','Topup_SA_Annual','Topup_RA_Annual','Topup_MA_Annual',            'Lump_OA_Annual','Lump_SA_Annual','Lump_RA_Annual','Lump_MA_Annual',
            'OA_Withdrawal_A_Annual','OA_Withdrawal_B_Annual','OA_Withdrawal_Annual']
    yearly_display = yearly_df[[c for c in cols if c in yearly_df.columns]].copy()
    st.dataframe(
        yearly_display.style.format({
            'End_OA':'{:,.0f}', 'End_SA':'{:,.0f}', 'End_MA':'{:,.0f}', 'End_RA':'{:,.0f}',
            'RA_capital_end':'{:,.0f}', 'Prevailing_ERS':'{:,.0f}',
            'Total_Base_Interest':'{:,.0f}', 'Total_Extra_Interest':'{:,.0f}',
            'OW_subject_total':'{:,.0f}', 'AW_subject_total':'{:,.0f}',
            'CPF_LIFE_Annual_Payout':'{:,.0f}',
            'MSHL_Annual':'{:,.0f}', 'LTCI_Premium_Annual_MA':'{:,.0f}',
            'IP_Base_MA_Annual':'{:,.0f}', 'IP_Base_Cash_Annual':'{:,.0f}', 'IP_Rider_Cash_Annual':'{:,.0f}',
            'Housing_OA_Annual':'{:,.0f}',
            'Topup_OA_Annual':'{:,.0f}', 'Topup_SA_Annual':'{:,.0f}', 'Topup_RA_Annual':'{:,.0f}', 'Topup_MA_Annual':'{:,.0f}',
            'Lump_OA_Annual':'{:,.0f}', 'Lump_SA_Annual':'{:,.0f}', 'Lump_RA_Annual':'{:,.0f}', 'Lump_MA_Annual':'{:,.0f}',
            'OA_Withdrawal_Annual':'{:,.0f}','OA_Withdrawal_A_Annual':'{:,.0f}','OA_Withdrawal_B_Annual':'{:,.0f}',

        }),
        use_container_width=True, height=360
    )

    # Monthly detail
    st.markdown("### Monthly Detail (optional)")
    with st.expander("Show monthly breakdown"):
        st.dataframe(monthly_df, use_container_width=True, height=420)

# ============= TAB 2: CPF LIFE & Bequest =============
with tab_cpf:
    # CPF LIFE payouts & bequest
    if include_cpf_life and cpf_life_df is not None:
         # Bequest + Monthly payout chart
        st.markdown("### Bequest & Monthly Payout Over Time")
        _bequest_plot = bequest_df.copy()
        _bequest_plot["YearAge"] = _bequest_plot["Year"].map(lambda y: _label_year_age(y, yearly_df))
        bequest_long = _bequest_plot.melt(
            id_vars=["Year", "YearAge"],
            value_vars=["Bequest_Remaining", "RA_Savings_Remaining", "Unused_Premium"],
            var_name="Component",
            value_name="Amount",
        )
        _payout_plot = cpf_life_df.copy()
        _payout_plot["YearAge"] = _payout_plot["Year"].map(lambda y: _label_year_age(y, yearly_df))
        _payout_plot["Component"] = "Monthly_Payout"

        legend_domain = ["Bequest_Remaining", "RA_Savings_Remaining", "Unused_Premium", "Monthly_Payout"]
        legend_range  = ["#1f77b4", "#9ecae1", "#d62728", "#6b7280"]

        base_x = alt.X("YearAge:O", title="Year (Age)", sort=None, axis=alt.Axis(labelAngle=-40))
        shared_color = alt.Color(
            "Component:N",
            title="Component",
            scale=alt.Scale(domain=legend_domain, range=legend_range),
        )

        bequest_lines = alt.Chart(bequest_long).mark_line(point=True).encode(
            x=base_x,
            y=alt.Y("Amount:Q", title="Bequest (S$)"),
            color=shared_color,
            tooltip=[
                alt.Tooltip("Year:Q", title="Year"),
                alt.Tooltip("YearAge:N", title="Year (Age)"),
                alt.Tooltip("Component:N", title="Component"),
                alt.Tooltip("Amount:Q", title="Bequest", format=",.0f"),
            ],
        )

        payout_line = alt.Chart(_payout_plot).mark_line(point=True, strokeDash=[4, 2]).encode(
            x=base_x,
            y=alt.Y("Monthly_Payout:Q", axis=alt.Axis(title="Monthly Payout (S$)", orient="right")),
            color=shared_color,
            tooltip=[
                alt.Tooltip("Year:Q", title="Year"),
                alt.Tooltip("YearAge:N", title="Year (Age)"),
                alt.Tooltip("Monthly_Payout:Q", title="Monthly payout", format=",.0f"),
                alt.Tooltip("Annual_Payout:Q", title="Annual payout", format=",.0f"),
            ],
        )

        dual_axis_chart = alt.layer(bequest_lines, payout_line).resolve_scale(y='independent').properties(height=320)
        st.altair_chart(dual_axis_chart, use_container_width=True)
        st.markdown("### CPF LIFE Payouts")
        start_monthly = meta.get("monthly_start_payout", None)
        if start_monthly is not None:
            st.markdown(
                f"**Plan:** {cpf_life_plan} &nbsp;&nbsp; "
                f"**Start age:** {payout_start_age} &nbsp;&nbsp; "
                f"**Monthly payout at start:** ${start_monthly:,.0f} "
            )

        _cpf_life_table = cpf_life_df.copy()
        _cpf_life_table["Year (Age)"] = _cpf_life_table["Year"].map(lambda y: _label_year_age(y, yearly_df))
        st.dataframe(
            _cpf_life_table[["Year (Age)", "Monthly_Payout", "Annual_Payout"]]
                .style.format({"Monthly_Payout": "{:,.0f}", "Annual_Payout": "{:,.0f}"}),
            use_container_width=True, height=260
        )

    if include_cpf_life and bequest_df is not None:
        st.markdown("### CPF LIFE Bequest (Estimated)")
        _bequest_table = bequest_df.copy()
        _bequest_table["Year (Age)"] = _bequest_table["Year"].map(lambda y: _label_year_age(y, yearly_df))
        st.dataframe(
            _bequest_table[["Year (Age)", "Bequest_Remaining", "Unused_Premium", "RA_Savings_Remaining"]]
                .style.format({
                    "Bequest_Remaining": "{:,.0f}",
                    "Unused_Premium": "{:,.0f}",
                    "RA_Savings_Remaining": "{:,.0f}",
                }),
            use_container_width=True, height=260
        )


# ============= TAB 3: Cashflows =============
with tab_cash:
    # ===== CASHFLOW CHARTS: CPF LIFE + OA A + OA B, plus TOTAL =====
    st.markdown("### Cashflow: CPF LIFE + OA Withdrawal (Nominal)")

    # Year-end (Dec) snapshot per year
    _snap = (
        monthly_df.sort_values(["Year", "Month"])
                  .groupby("Year", as_index=False)
                  .tail(1)[["Year", "CPF_LIFE_monthly_payout", "OA_Withdrawal1_Paid", "OA_Withdrawal2_Paid"]]
                  .copy()
    )

    # Ensure columns exist even if zeros
    for col in ["CPF_LIFE_monthly_payout", "OA_Withdrawal1_Paid", "OA_Withdrawal2_Paid"]:
        if col not in _snap.columns:
            _snap[col] = 0.0

    # X-axis label "YYYY (Age A)"
    _age_by_year = yearly_df.set_index("Year")["Age_end"].to_dict()
    _snap["YearAge"] = _snap["Year"].map(lambda y: f"{int(y)} (Age {int(_age_by_year.get(int(y), 0))})")

    # Palette (consistent with your other charts)
    COLOR_CPF   = "#6b7280"  # CPF LIFE (dashed)
    COLOR_OA_A  = "#1f77b4"  # OA A
    COLOR_OA_B  = "#9ecae1"  # OA B
    COLOR_TOTAL = "#d62728"  # Total
    LINE_W = 2

    base_x = alt.X("YearAge:O", title="Year (Age)", sort=None, axis=alt.Axis(labelAngle=-40))

    # ---------- NOMINAL ----------
    _nom = _snap.copy()
    _nom["CPF"]  = _nom["CPF_LIFE_monthly_payout"]
    _nom["OA_A"] = _nom["OA_Withdrawal1_Paid"]
    _nom["OA_B"] = _nom["OA_Withdrawal2_Paid"]
    _nom["Total"] = _nom[["CPF", "OA_A", "OA_B"]].sum(axis=1)

    oaA_present = float(_nom["OA_A"].abs().sum()) > 1e-9
    oaB_present = float(_nom["OA_B"].abs().sum()) > 1e-9

    value_vars_nom = ["Total", "CPF"] + (["OA_A"] if oaA_present else []) + (["OA_B"] if oaB_present else [])
    _nom_long = pd.melt(
        _nom,
        id_vars=["Year", "YearAge"],
        value_vars=value_vars_nom,
        var_name="Series",
        value_name="Amount",
    )

    label_map_nom = {
        "Total": "Total (monthly)",
        "CPF":   "CPF LIFE (monthly)",
        "OA_A":  "OA A (monthly)",
        "OA_B":  "OA B (monthly)",
    }
    _nom_long["Label"] = _nom_long["Series"].map(label_map_nom)

    legend_domain_nom = [label_map_nom[s] for s in value_vars_nom]
    color_map_nom = {"Total": COLOR_TOTAL, "CPF": COLOR_CPF, "OA_A": COLOR_OA_A, "OA_B": COLOR_OA_B}
    legend_range_nom = [color_map_nom[s] for s in value_vars_nom]

    # Area under Total ONLY
    nom_total_area = (
        alt.Chart(_nom_long[_nom_long["Series"] == "Total"])
          .mark_area(opacity=0.12)
          .encode(
              x=base_x,
              y=alt.Y("Amount:Q", title="Amount (S$)", axis=alt.Axis(format=",.0f")),
              color=alt.value(COLOR_TOTAL),
          )
    )

    # Total line (solid)
    nom_total_line = (
        alt.Chart(_nom_long[_nom_long["Series"] == "Total"])
          .mark_line(point=True)
          .encode(
              x=base_x,
              y=alt.Y("Amount:Q", title="Amount (S$)", axis=alt.Axis(format=",.0f")),
              color=alt.Color("Label:N", title="Component",
                              scale=alt.Scale(domain=legend_domain_nom, range=legend_range_nom)),
              strokeWidth=alt.value(LINE_W),
          )
    )

    # CPF LIFE line (dashed)
    nom_cpf_line = (
        alt.Chart(_nom_long[_nom_long["Series"] == "CPF"])
          .mark_line(point=True, strokeDash=[4, 2])
          .encode(
              x=base_x,
              y=alt.Y("Amount:Q", title="Amount (S$)", axis=alt.Axis(format=",.0f")),
              color=alt.Color("Label:N", title="Component",
                              scale=alt.Scale(domain=legend_domain_nom, range=legend_range_nom)),
              strokeWidth=alt.value(LINE_W),
          )
    )

    layers = nom_total_area + nom_total_line + nom_cpf_line

    if oaA_present:
        layers = layers + (
            alt.Chart(_nom_long[_nom_long["Series"] == "OA_A"])
              .mark_line(point=True, strokeDash=[4, 2])
              .encode(
                  x=base_x,
                  y=alt.Y("Amount:Q", title="Amount (S$)", axis=alt.Axis(format=",.0f")),
                  color=alt.Color("Label:N", title="Component",
                                  scale=alt.Scale(domain=legend_domain_nom, range=legend_range_nom)),
                  strokeWidth=alt.value(LINE_W),
              )
        )

    if oaB_present:
        layers = layers + (
            alt.Chart(_nom_long[_nom_long["Series"] == "OA_B"])
              .mark_line(point=True, strokeDash=[4, 2])
              .encode(
                  x=base_x,
                  y=alt.Y("Amount:Q", title="Amount (S$)", axis=alt.Axis(format=",.0f")),
                  color=alt.Color("Label:N", title="Component",
                                  scale=alt.Scale(domain=legend_domain_nom, range=legend_range_nom)),
                  strokeWidth=alt.value(LINE_W),
              )
        )

    st.altair_chart(layers.properties(height=320, title="Nominal"), use_container_width=True)

    # ---------- REAL (today’s dollars) ----------
    st.markdown("### Cashflow: CPF LIFE + OA Withdrawal (Real, in Today’s Dollars)")

    _today_year = date.today().year
    _snap["Deflator"] = (1.0 + float(inflation_pct)) ** (_snap["Year"] - _today_year).clip(lower=0)

    _real = _snap.copy()
    _real["CPF"]  = _real["CPF_LIFE_monthly_payout"] / _real["Deflator"]
    _real["OA_A"] = _real["OA_Withdrawal1_Paid"]      / _real["Deflator"]
    _real["OA_B"] = _real["OA_Withdrawal2_Paid"]      / _real["Deflator"]
    _real["Total"] = _real[["CPF", "OA_A", "OA_B"]].sum(axis=1)

    value_vars_real = ["Total", "CPF"] + (["OA_A"] if oaA_present else []) + (["OA_B"] if oaB_present else [])
    _real_long = pd.melt(
        _real,
        id_vars=["Year", "YearAge"],
        value_vars=value_vars_real,
        var_name="Series",
        value_name="Amount",
    )

    label_map_real = {
        "Total": "Total (monthly, real)",
        "CPF":   "CPF LIFE (monthly, real)",
        "OA_A":  "OA A (monthly, real)",
        "OA_B":  "OA B (monthly, real)",
    }
    _real_long["Label"] = _real_long["Series"].map(label_map_real)

    legend_domain_real = [label_map_real[s] for s in value_vars_real]
    color_map_real = {"Total": COLOR_TOTAL, "CPF": COLOR_CPF, "OA_A": COLOR_OA_A, "OA_B": COLOR_OA_B}
    legend_range_real = [color_map_real[s] for s in value_vars_real]

    real_total_area = (
        alt.Chart(_real_long[_real_long["Series"] == "Total"])
          .mark_area(opacity=0.12)
          .encode(
              x=base_x,
              y=alt.Y("Amount:Q", title="Amount (S$, today’s $)", axis=alt.Axis(format=",.0f")),
              color=alt.value(COLOR_TOTAL),
          )
    )

    real_total_line = (
        alt.Chart(_real_long[_real_long["Series"] == "Total"])
          .mark_line(point=True)
          .encode(
              x=base_x,
              y=alt.Y("Amount:Q", title="Amount (S$, today’s $)", axis=alt.Axis(format=",.0f")),
              color=alt.Color("Label:N", title="Component",
                              scale=alt.Scale(domain=legend_domain_real, range=legend_range_real)),
              strokeWidth=alt.value(LINE_W),
          )
    )

    real_cpf_line = (
        alt.Chart(_real_long[_real_long["Series"] == "CPF"])
          .mark_line(point=True, strokeDash=[4, 2])
          .encode(
              x=base_x,
              y=alt.Y("Amount:Q", title="Amount (S$, today’s $)", axis=alt.Axis(format=",.0f")),
              color=alt.Color("Label:N", title="Component",
                              scale=alt.Scale(domain=legend_domain_real, range=legend_range_real)),
              strokeWidth=alt.value(LINE_W),
          )
    )

    real_layers = real_total_area + real_total_line + real_cpf_line

    if oaA_present:
        real_layers = real_layers + (
            alt.Chart(_real_long[_real_long["Series"] == "OA_A"])
              .mark_line(point=True, strokeDash=[4, 2])
              .encode(
                  x=base_x,
                  y=alt.Y("Amount:Q", title="Amount (S$, today’s $)", axis=alt.Axis(format=",.0f")),
                  color=alt.Color("Label:N", title="Component",
                                  scale=alt.Scale(domain=legend_domain_real, range=legend_range_real)),
                  strokeWidth=alt.value(LINE_W),
              )
        )

    if oaB_present:
        real_layers = real_layers + (
            alt.Chart(_real_long[_real_long["Series"] == "OA_B"])
              .mark_line(point=True, strokeDash=[4, 2])
              .encode(
                  x=base_x,
                  y=alt.Y("Amount:Q", title="Amount (S$, today’s $)", axis=alt.Axis(format=",.0f")),
                  color=alt.Color("Label:N", title="Component",
                                  scale=alt.Scale(domain=legend_domain_real, range=legend_range_real)),
                  strokeWidth=alt.value(LINE_W),
              )
        )

    st.altair_chart(real_layers.properties(height=320, title="Real"), use_container_width=True)

# ============= TAB 4: Health Insurance =============
with tab_health:

    # Health Insurance Premiums (stacked intended sources)
    if ("MSHL_Nominal" in yearly_df.columns) and (
        yearly_df[["MSHL_Nominal",
                   "IP_Base_MA_Nominal_Annual",
                   "IP_Base_Cash_Nominal_Annual",
                   "IP_Rider_Cash_Nominal_Annual"]].sum(numeric_only=True).sum() > 0
    ):
        st.markdown("### Health Insurance Premiums (By Intended Funding Source)")
        plot_df = yearly_df[["Year", "Age_end",
                             "MSHL_Nominal",
                             "IP_Base_MA_Nominal_Annual",
                             "IP_Base_Cash_Nominal_Annual",
                             "IP_Rider_Cash_Nominal_Annual"]].copy()
        plot_df["YearAge"] = plot_df["Year"].map(lambda y: _label_year_age(y, yearly_df))

        hi_long = plot_df.melt(
            id_vars=["Year", "Age_end", "YearAge"],
            value_vars=["MSHL_Nominal", "IP_Base_MA_Nominal_Annual", "IP_Base_Cash_Nominal_Annual", "IP_Rider_Cash_Nominal_Annual"],
            var_name="HI_Component", value_name="Amount"
        )

        name_map = {
            "MSHL_Nominal": "MediShield Life (MA)",
            "IP_Base_MA_Nominal_Annual": "IP Base (MA)",
            "IP_Base_Cash_Nominal_Annual": "IP Base (Cash)",
            "IP_Rider_Cash_Nominal_Annual": "IP Rider (Cash)",
        }
        order_map = {
            "MediShield Life (MA)": 0,
            "IP Base (MA)": 1,
            "IP Base (Cash)": 2,
            "IP Rider (Cash)": 3,
        }
        hi_long["HI_Component"] = hi_long["HI_Component"].map(name_map)
        hi_long["HI_Order"] = hi_long["HI_Component"].map(order_map)

        ip_chart = alt.Chart(hi_long).mark_bar().encode(
            x=alt.X("YearAge:O", title="Year (Age)", sort=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("sum(Amount):Q", title="Premium (S$)", axis=alt.Axis(format=',.0f')),
            color=alt.Color("HI_Component:N", title="Component",
                            scale=alt.Scale(domain=list(order_map.keys()))),
            order=alt.Order("HI_Order:Q"),
            tooltip=[
                alt.Tooltip("Year:Q", title="Year"),
                alt.Tooltip("YearAge:N", title="Year (Age)"),
                alt.Tooltip("HI_Component:N", title="Component"),
                alt.Tooltip("Amount:Q", title="Amount", format=",.0f"),
            ],
        ).properties(height=300)
        st.altair_chart(ip_chart, use_container_width=True)


        st.markdown("### Healthcare Inflation & IP Premium Calculator")

        col_inf1, col_inf2 = st.columns(2)
        with col_inf1:
            healthcare_inflation_pct = st.number_input(
                "Healthcare inflation % p.a.",
                min_value=0.0, max_value=20.0, value=3.0, step=0.5, format="%.1f"
            ) / 100
        with col_inf2:
            base_year_for_health_inf = st.number_input(
                "Inflation base year (compounding starts here)",
                min_value=int(yearly_df["Year"].min()),
                max_value=int(yearly_df["Year"].max()),
                value=int(start_year),
                step=1
            )

        plot_df_inf = yearly_df[[
            "Year", "Age_end",
            "IP_Base_MA_Nominal_Annual",
            "IP_Base_Cash_Nominal_Annual",
            "IP_Rider_Cash_Nominal_Annual"
        ]].copy()

        # Combine IP Base components and rename rider
        plot_df_inf["IP_Base_Total_Nominal"] = (
            plot_df_inf["IP_Base_MA_Nominal_Annual"].fillna(0.0) +
            plot_df_inf["IP_Base_Cash_Nominal_Annual"].fillna(0.0)
        )
        plot_df_inf["IP_Rider_Nominal"] = plot_df_inf["IP_Rider_Cash_Nominal_Annual"].fillna(0.0)
        plot_df_inf["YearAge"] = plot_df_inf["Year"].map(lambda y: _label_year_age(y, yearly_df))

        hi_long_inf = plot_df_inf.melt(
            id_vars=["Year", "Age_end", "YearAge"],
            value_vars=["IP_Base_Total_Nominal", "IP_Rider_Nominal"],
            var_name="HI_ComponentRaw", value_name="Amount_Nominal"
        )
        name_map = {"IP_Base_Total_Nominal": "IP Base", "IP_Rider_Nominal": "IP Rider"}
        order_map = {"IP Base": 0, "IP Rider": 1}
        hi_long_inf["HI_Component"] = hi_long_inf["HI_ComponentRaw"].map(name_map)
        hi_long_inf["HI_Order"] = hi_long_inf["HI_Component"].map(order_map)

        # Inflation math (defensive)
        hi_long_inf["Amount_Nominal"] = hi_long_inf["Amount_Nominal"].fillna(0.0)
        hi_long_inf["YearsFromBase"] = (hi_long_inf["Year"].astype(int) - int(base_year_for_health_inf)).clip(lower=0)
        hi_long_inf["InflFactor"] = (1.0 + float(healthcare_inflation_pct)) ** hi_long_inf["YearsFromBase"]
        hi_long_inf["Amount_Inflated"] = hi_long_inf["Amount_Nominal"] * hi_long_inf["InflFactor"]

        if float(hi_long_inf["Amount_Nominal"].sum()) <= 1e-9:
            st.info("No IP premiums found to inflate in the current projection.")
        else:
            ip_chart_infl = alt.Chart(hi_long_inf).mark_bar().encode(
                x=alt.X("YearAge:O", title="Year (Age)", sort=None, axis=alt.Axis(labelAngle=-40)),
                y=alt.Y("sum(Amount_Inflated):Q", title="Premium (S$)", axis=alt.Axis(format=',.0f')),
                color=alt.Color("HI_Component:N", title="Component",
                                scale=alt.Scale(domain=list(order_map.keys()))),
                order=alt.Order("HI_Order:Q"),
                tooltip=[
                    alt.Tooltip("Year:Q", title="Year"),
                    alt.Tooltip("YearAge:N", title="Year (Age)"),
                    alt.Tooltip("HI_Component:N", title="Component"),
                    alt.Tooltip("Amount_Nominal:Q", title="Nominal", format=",.0f"),
                    alt.Tooltip("Amount_Inflated:Q", title="Inflation-adjusted", format=",.0f"),
                ],
            ).properties(height=300, title="IP Premiums (Inflation-adjusted)")
            st.altair_chart(ip_chart_infl, use_container_width=True)

            # ANB-range totals
            st.markdown("### Totals Between ANB Range (Inflation-adjusted)")
            def _anb_for_year(y: int) -> int:
                return age_at(dob, int(y), int(insurance_month)) + 1
            hi_long_inf["ANB"] = hi_long_inf["Year"].apply(_anb_for_year)

            min_anb = int(hi_long_inf["ANB"].min()); max_anb = int(hi_long_inf["ANB"].max())
            col_anb1, col_anb2, col_anb3 = st.columns([1,1,2])
            with col_anb1:
                anb_start = st.number_input("Start ANB", min_value=min_anb, max_value=max_anb, value=min_anb, step=1)
            with col_anb2:
                anb_end = st.number_input("End ANB", min_value=min_anb, max_value=max_anb, value=max_anb, step=1)

            if anb_end < anb_start:
                st.warning("End ANB is less than Start ANB. Please adjust the range.")
            else:
                sel = hi_long_inf[(hi_long_inf["ANB"] >= int(anb_start)) & (hi_long_inf["ANB"] <= int(anb_end))].copy()
                total_infl = float(sel["Amount_Inflated"].sum())
                by_component = (sel.groupby("HI_Component", as_index=False)["Amount_Inflated"]
                                  .sum()
                                  .sort_values("Amount_Inflated", ascending=False))
                st.metric(
                    f"Total inflation-adjusted IP premiums (ANB {anb_start}–{anb_end})",
                    f"${total_infl:,.0f}"
                )
                st.dataframe(
                    by_component.rename(columns={"Amount_Inflated": "Total (Inflation-adjusted)"}).style.format({"Total (Inflation-adjusted)": "{:,.0f}"}),
                    use_container_width=True, height=160
                )


# Notes
notes_html = []
cohort_bhs = get_bhs_for_year_with_cohort(dob, dob.year + 65, bhs_growth_pct)
notes_html.append(f"  Cohort FRS (fixed at age 55): <b>${cohort_frs:,.0f}</b>.")
notes_html.append(f"  Desired RA opening balance at 55 years old (&times;{ers_factor:.2f}): <b>${cohort_frs*ers_factor:,.0f}</b>.")
notes_html.append(f"  Cohort BHS (fixed at age 65): <b>${cohort_bhs:,.0f}</b>.")

# Note when RA capital first reaches cohort FRS
tol = 1e-6
hit = (
    monthly_df.loc[monthly_df["RA_capital"] >= (cohort_frs - tol)]
              .sort_values(["Year", "Month"], ascending=[True, True])
)
if not hit.empty:
    first = hit.iloc[0]
    notes_html.append(
        f"  Your RA savings (capital) first reached your cohort FRS in "
        f"<b>{int(first['Year'])}-{int(first['Month']):02d} (Age {int(first['Age'])})</b>."
    )

if meta.get("monthly_start_payout"):
    start_monthly = meta["monthly_start_payout"]
    notes_html.append(
        f"  CPF LIFE monthly payout at start (nominal): <b>${start_monthly:,.0f}</b>."
    )

if include_cpf_life and (cpf_life_df is not None) and meta.get("monthly_start_payout"):
    start_monthly = meta["monthly_start_payout"]
    inflation_assumed = float(inflation_pct)
    years_until_start = max(0, (dob.year + int(payout_start_age)) - int(start_year))
    start_monthly_today = start_monthly / ((1 + inflation_assumed) ** years_until_start)
    notes_html.append(
        f"  Starting CPF LIFE monthly payout in today's value @{inflation_assumed*100:.1f}% inflation: "
        f"<b>${start_monthly_today:,.0f}</b>"
    )


st.markdown("---")
st.markdown(
    "<div style='font-size:16px; line-height:1.6; color:#111;'>" +
    "<br/>".join(notes_html) +
    "</div>",
    unsafe_allow_html=True
)

# Build and download the PDF
if st.button("📥 Download PDF report", use_container_width=True):
    pdf_bytes = make_pdf_report(
        monthly_df=monthly_df,
        yearly_df=yearly_df,
        cohort_frs=cohort_frs,
        cpf_life_df=cpf_life_df,
        bequest_df=bequest_df,
        include_cpf_life=bool(include_cpf_life),
        cpf_life_plan=cpf_life_plan,
        payout_start_age=int(payout_start_age),
        notes_html_list=notes_html,   # you already compute this list
    )
    st.download_button(
        "Save report.pdf",
        data=pdf_bytes,
        file_name="cpf_projector_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
