"""Main Streamlit application shell for Personal GL."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import io
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.workspace_apps.GLA import (
    JournalEntryLine,
    compute_balance_time_series,
    compute_mom_financials_month_end,
    compute_period_close_preview,
    create_manual_txn_template,
    create_month_end_checklist,
    create_user_upload_mapping,
    delete_txn_keyword_rule,
    get_account_balance_as_of,
    get_accounts,
    get_app_access_password,
    get_bs_category_gl_numbers,
    get_connection,
    get_financials_start_date,
    get_last_closed_period_end,
    get_manual_txn_templates,
    get_month_end_checklist_account_ids,
    get_month_end_checklists,
    get_month_end_closing_checklist_enabled,
    get_override_pin,
    get_retained_earnings_gl_number,
    get_streamlined_manual_txns_enabled,
    get_suspense_gl_number,
    get_transfers_clearing_gl_number,
    get_txn_keyword_rules,
    get_unbalanced_journal_entries,
    get_user_upload_mapping_by_name,
    get_user_upload_mappings,
    has_period_close_entry,
    init_db,
    insert_journal_entry,
    load_user_defined_file,
    post_journal_entry_with_period_lock,
    post_period_close_entry,
    set_app_access_password,
    set_bs_category_gl_numbers,
    set_financials_start_date,
    set_last_closed_period_end,
    set_manual_txn_template_active,
    set_month_end_checklist_active,
    set_month_end_closing_checklist_enabled,
    set_override_pin,
    set_retained_earnings_gl_number,
    set_streamlined_manual_txns_enabled,
    set_suspense_gl_number,
    set_transfers_clearing_gl_number,
    set_user_upload_mapping_active,
    update_month_end_checklist_accounts,
    upsert_txn_keyword_rule,
)
from app.workspace_apps.personal_gl.constants import (
    BS_CATEGORIES,
    DATASET_BULK_JE_STAGE,
    DATASET_FINANCIAL_REPORT,
    DATASET_SQLITE_DB,
    DATASET_UPLOAD_STAGE,
    TAB_LABELS,
)
from app.workspace_apps.personal_gl.runtime import build_runtime
from app.workspace_apps.personal_gl.state import UploadStageMeta
from app.workspace_apps.personal_gl.ui.auth import load_stored_password, render_global_controls, render_lock_gate
from app.workspace_apps.personal_gl.ui.formatting import format_money, format_ratio, safe_ratio
from app.workspace_apps.personal_gl.ui.tabs.account_history import render_account_history_tab
from app.workspace_apps.personal_gl.ui.tabs.chart_of_accounts import render_chart_of_accounts_tab
from app.workspace_apps.personal_gl.ui.tabs.documentation import render_documentation_tab
from app.workspace_apps.personal_gl.ui.tabs.logs import render_logs_tab
from app.workspace_apps.personal_gl.ui.tabs.notes import render_notes_tab
from app.workspace_apps.personal_gl.ui.tabs.search import render_search_tab
from app.workspace_apps.personal_gl.ui.tabs.warnings import render_warnings_tab


def _is_month_end(value: date) -> bool:
    return (value + timedelta(days=1)).month != value.month


def _last_day_of_month(value: date) -> date:
    first_next = (value.replace(day=1) + timedelta(days=32)).replace(day=1)
    return first_next - timedelta(days=1)


def _shift_month_end(month_end: date, months: int) -> date:
    year = month_end.year
    month = month_end.month + months
    year += (month - 1) // 12
    month = ((month - 1) % 12) + 1
    return _last_day_of_month(date(year, month, 1))


def _robust_outlier_flags(values: pd.Series, z_thresh: float = 3.5, fallback_top_n: int = 3) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").fillna(0.0)
    median = float(series.median())
    mad = float((series - median).abs().median())
    if mad <= 0 or pd.isna(mad):
        absolute = series.abs()
        flags = pd.Series(False, index=series.index)
        if len(absolute) <= 1:
            return flags
        top_index = absolute.sort_values(ascending=False).head(max(0, int(fallback_top_n))).index
        flags.loc[top_index] = True
        return flags
    z_scores = 0.6745 * (series - median) / mad
    return (z_scores.abs() > float(z_thresh)).astype(bool)


def _chart_balance_and_variance(df_plot: pd.DataFrame, title_prefix: str) -> None:
    if df_plot.empty:
        st.info("No data to chart.")
        return
    dfp = df_plot.copy()
    dfp["date"] = pd.to_datetime(dfp["date"])

    line = (
        alt.Chart(dfp)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("balance:Q", title="Balance ($)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("balance:Q", title="Balance ($)", format=",.2f"),
            ],
        )
        .properties(title=f"{title_prefix} — Balance")
    )
    points = (
        alt.Chart(dfp)
        .mark_circle(size=35)
        .encode(
            x="date:T",
            y="balance:Q",
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("balance:Q", title="Balance ($)", format=",.2f"),
            ],
        )
    )
    st.altair_chart((line + points).interactive(), use_container_width=True)

    bars = (
        alt.Chart(dfp)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("delta:Q", title="Period Change ($)"),
            color=alt.condition("datum.is_outlier", alt.value("#d62728"), alt.value("#1f77b4")),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("delta:Q", title="Change ($)", format=",.2f"),
                alt.Tooltip("is_outlier:N", title="Outlier"),
            ],
        )
        .properties(title=f"{title_prefix} — Period-to-Period Change")
    )
    zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule().encode(y="y:Q")
    st.altair_chart((bars + zero_rule).interactive(), use_container_width=True)


def _format_money_df_from_cents(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        out[column] = out[column].apply(
            lambda value: "" if value is None or (isinstance(value, float) and pd.isna(value)) else format_money(float(value) / 100.0)
        )
    return out


def _format_ratio_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        out[column] = out[column].apply(
            lambda value: "" if value is None or (isinstance(value, float) and pd.isna(value)) else format_ratio(float(value))
        )
    return out


def render_financial_statements_tab(runtime) -> None:
    st.subheader("Financial Statements")

    as_of = st.date_input("As of date", value=date.today(), key="financials_as_of")
    as_of_str = as_of.isoformat()
    report_type = st.selectbox(
        "Report Type",
        options=["GL Summary View", "Balance Sheet", "Income Statement", "Historical (Month-End)", "Trends & Outliers"],
        index=0,
        key="financials_report_type",
    )

    conn = get_connection()
    try:
        accounts = get_accounts(conn, include_inactive=False)
        balances_cents = {}
        for account in accounts:
            bal_dc = get_account_balance_as_of(conn, account.id, as_of_str)
            balances_cents[account.gl_number] = bal_dc if account.type.value in ("ASSET", "EXPENSE") else -bal_dc

        def sum_gl(gl_numbers: list[int]) -> int:
            return sum(balances_cents.get(gl, 0) for gl in gl_numbers)

        def get_bs_gls(category_key: str) -> list[int]:
            return get_bs_category_gl_numbers(conn, category_key) or []

        liquid_cash_c = sum_gl(get_bs_gls("LIQUID_CASH"))
        certificates_c = sum_gl(get_bs_gls("CERTIFICATES"))
        equities_c = sum_gl(get_bs_gls("EQUITIES"))
        bonds_c = sum_gl(get_bs_gls("BONDS"))
        cryptocurrency_c = sum_gl(get_bs_gls("CRYPTOCURRENCY"))
        retirement_invest_c = sum_gl(get_bs_gls("RETIREMENT_INVEST"))
        liquid_invest_c = sum_gl(get_bs_gls("LIQUID_INVEST"))
        illiquid_invest_c = sum_gl(get_bs_gls("ILLIQUID_INVEST"))
        inventory_c = sum_gl(get_bs_gls("INVENTORY"))
        accrued_c = sum_gl(get_bs_gls("ACCRUED"))
        allowance_doubtful_c = sum_gl(get_bs_gls("ALLOWANCE_DOUBTFUL"))
        net_accrued_c = sum_gl(get_bs_gls("NET_ACCRUED"))
        real_estate_c = sum_gl(get_bs_gls("REAL_ESTATE"))
        fixed_assets_c = sum_gl(get_bs_gls("FIXED_ASSETS"))
        accumulated_depreciation_c = sum_gl(get_bs_gls("ACCUMULATED DEPRECIATION"))
        net_fixed_assets_c = sum_gl(get_bs_gls("NET FIXED ASSETS"))
        other_assets_c = sum_gl(get_bs_gls("OTHER_ASSETS"))
        liquid_assets_c = sum_gl(get_bs_gls("LIQUID_ASSETS"))
        illiquid_assets_c = sum_gl(get_bs_gls("ILLIQUID_ASSETS"))
        revolving_debt_c = sum_gl(get_bs_gls("REVOLVING_DEBT"))
        installment_debt_c = sum_gl(get_bs_gls("INSTALLMENT_DEBT"))
        education_debt_c = sum_gl(get_bs_gls("EDUCATION_DEBT"))
        real_estate_debt_c = sum_gl(get_bs_gls("REAL_ESTATE_DEBT"))
        payables_c = sum_gl(get_bs_gls("PAYABLES"))
        other_liabilities_c = sum_gl(get_bs_gls("OTHER_LIABILITIES"))

        total_cash_c = liquid_cash_c + certificates_c
        total_invest_c = liquid_invest_c + retirement_invest_c + illiquid_invest_c + equities_c + bonds_c + cryptocurrency_c
        total_cash_invest_c = total_cash_c + total_invest_c
        total_assets_c = sum(balances_cents.get(account.gl_number, 0) for account in accounts if account.type.value == "ASSET")
        total_debt_c = revolving_debt_c + installment_debt_c + real_estate_debt_c
        total_liabilities_c = sum(balances_cents.get(account.gl_number, 0) for account in accounts if account.type.value == "LIABILITY")
        net_worth_c = total_assets_c - total_liabilities_c

        transfers_gl_number = get_transfers_clearing_gl_number(conn)
        if transfers_gl_number is None:
            transfers_in_process_c = 0
            transfers_in_process_label = "Transfers In-Process (not configured)"
        else:
            transfers_in_process_c = balances_cents.get(int(transfers_gl_number), 0)
            transfers_in_process_label = f"Transfers In-Process (GL {int(transfers_gl_number)})"

        equity_gls = sorted(get_bs_gls("TOTAL_EQUITY"))
        equity_total_c = sum_gl(get_bs_gls("TOTAL_EQUITY"))
        gl_to_name = {account.gl_number: account.name for account in accounts}

        year_start = date(as_of.year, 1, 1)
        if as_of > year_start:
            prior_day = year_start - timedelta(days=1)
            prior_str = prior_day.isoformat()
            balances_start_c = {}
            for account in accounts:
                bal_dc_start = get_account_balance_as_of(conn, account.id, prior_str)
                balances_start_c[account.gl_number] = bal_dc_start if account.type.value in ("ASSET", "EXPENSE") else -bal_dc_start
            total_assets_start_c = sum(
                balances_start_c.get(account.gl_number, 0)
                for account in accounts
                if account.type.value == "ASSET"
            )
            asset_growth_ytd_c = total_assets_c - total_assets_start_c
        else:
            asset_growth_ytd_c = 0

        income_lines = []
        expense_lines = []
        total_income_c = 0
        total_expense_c = 0
        for account in accounts:
            if account.type.value not in ("INCOME", "EXPENSE"):
                continue
            amt_c = balances_cents.get(account.gl_number, 0)
            if account.type.value == "INCOME":
                total_income_c += amt_c
                if amt_c != 0:
                    income_lines.append({"gl_number": account.gl_number, "name": account.name, "amount_cents": amt_c})
            else:
                total_expense_c += amt_c
                if amt_c != 0:
                    expense_lines.append({"gl_number": account.gl_number, "name": account.name, "amount_cents": amt_c})

        ytd_net_income_c = total_income_c - total_expense_c
        ytd_expenses_c = total_expense_c

        days_elapsed = (as_of - year_start).days + 1
        if ytd_expenses_c > 0 and days_elapsed > 0:
            annualized_expenses_c = int(ytd_expenses_c * 365 / days_elapsed)
            cash_on_hand_years = liquid_cash_c / annualized_expenses_c if annualized_expenses_c > 0 else None
        else:
            cash_on_hand_years = None
    finally:
        conn.close()

    liquid_cash = liquid_cash_c / 100.0
    certificates = certificates_c / 100.0
    total_cash = total_cash_c / 100.0
    equities = equities_c / 100.0
    bonds = bonds_c / 100.0
    cryptocurrency = cryptocurrency_c / 100.0
    retirement_invest = retirement_invest_c / 100.0
    liquid_invest = liquid_invest_c / 100.0
    illiquid_invest = illiquid_invest_c / 100.0
    total_invest = total_invest_c / 100.0
    total_cash_invest = total_cash_invest_c / 100.0
    inventory = inventory_c / 100.0
    accrued = accrued_c / 100.0
    allowance_doubtful = allowance_doubtful_c / 100.0
    net_accrued = net_accrued_c / 100.0
    real_estate = real_estate_c / 100.0
    fixed_assets = fixed_assets_c / 100.0
    accumulated_depreciation = accumulated_depreciation_c / 100.0
    net_fixed_assets = net_fixed_assets_c / 100.0
    other_assets = other_assets_c / 100.0
    liquid_assets = liquid_assets_c / 100.0
    illiquid_assets = illiquid_assets_c / 100.0
    total_assets = total_assets_c / 100.0
    asset_growth_ytd = asset_growth_ytd_c / 100.0
    revolving_debt = revolving_debt_c / 100.0
    installment_debt = installment_debt_c / 100.0
    education_debt = education_debt_c / 100.0
    real_estate_debt = real_estate_debt_c / 100.0
    total_debt = total_debt_c / 100.0
    transfers_in_process = transfers_in_process_c / 100.0
    payables = payables_c / 100.0
    other_liabilities = other_liabilities_c / 100.0
    total_liabilities = total_liabilities_c / 100.0
    net_worth = net_worth_c / 100.0
    current_earnings = ytd_net_income_c / 100.0
    equity_total = (equity_total_c + ytd_net_income_c) / 100.0
    total_liabilities_equity = total_liabilities + equity_total
    liquid_assets_ratio = safe_ratio(liquid_assets_c, total_assets_c)
    liquid_cash_ratio = safe_ratio(liquid_cash_c, total_assets_c)
    debt_assets_ratio = safe_ratio(total_debt_c, total_assets_c)
    debt_liquid_cash_ratio = safe_ratio(total_debt_c, liquid_cash_c)
    fixed_assets_ratio = safe_ratio(fixed_assets_c, total_assets_c)

    asset_rows_ordered = [
        ("Liquid Cash", liquid_cash_c, liquid_cash, "leaf"),
        ("Certificates", certificates_c, certificates, "leaf"),
        ("Total Cash", None, total_cash, "subtotal"),
        ("Equities", equities_c, equities, "leaf"),
        ("Bonds", bonds_c, bonds, "leaf"),
        ("Cryptocurrency", cryptocurrency_c, cryptocurrency, "leaf"),
        ("Retirement Investments", retirement_invest_c, retirement_invest, "leaf"),
        ("Liquid Investments", liquid_invest_c, liquid_invest, "leaf"),
        ("Illiquid Investments", illiquid_invest_c, illiquid_invest, "leaf"),
        ("Total Investments", None, total_invest, "subtotal"),
        ("Total Cash and Investments", None, total_cash_invest, "subtotal"),
        ("Inventory", inventory_c, inventory, "leaf"),
        ("Accrued Revenue / Receivables", accrued_c, accrued, "leaf"),
        ("Allowance for Doubtful Accounts", allowance_doubtful_c, allowance_doubtful, "leaf"),
        ("Net Accrued Revenue / Receivables", net_accrued_c, net_accrued, "leaf"),
        ("Real Estate", real_estate_c, real_estate, "leaf"),
        ("Fixed Assets", fixed_assets_c, fixed_assets, "leaf"),
        ("Accumulated Depreciation", accumulated_depreciation_c, accumulated_depreciation, "leaf"),
        ("Net Fixed Assets", net_fixed_assets_c, net_fixed_assets, "leaf"),
        ("Other Assets", other_assets_c, other_assets, "leaf"),
        ("Liquid Assets", None, liquid_assets, "subtotal"),
        ("Illiquid Assets", None, illiquid_assets, "subtotal"),
        ("Total Assets", None, total_assets, "total"),
        ("Asset Growth YTD", None, asset_growth_ytd, "total"),
    ]
    liab_rows_ordered = [
        ("Revolving Debt", revolving_debt_c, revolving_debt, "leaf"),
        ("Installment Debt", installment_debt_c, installment_debt, "leaf"),
        ("Education Debt", education_debt_c, education_debt, "leaf"),
        ("Real Estate Debt", real_estate_debt_c, real_estate_debt, "leaf"),
        ("Total Debt", None, total_debt, "subtotal"),
        (transfers_in_process_label, transfers_in_process_c, transfers_in_process, "operational"),
        ("Payables / Accrued Expenses", payables_c, payables, "leaf"),
        ("Other Liabilities", other_liabilities_c, other_liabilities, "leaf"),
        ("Total Liabilities", None, total_liabilities, "total"),
    ]

    def _pct(value):
        return value * 100 if value is not None else None

    def _format_pct(value):
        if value is None:
            return ""
        return f"{value:.2f}%"

    bs_rows_export = []
    for label, cents_val, dollars_val, row_type in asset_rows_ordered:
        if row_type == "leaf":
            if cents_val is not None and cents_val != 0:
                bs_rows_export.append({"Section": "Assets", "Item": label, "Amount": format_money(dollars_val)})
        else:
            bs_rows_export.append({"Section": "Assets", "Item": label, "Amount": format_money(dollars_val)})
    for label, cents_val, dollars_val, row_type in liab_rows_ordered:
        if row_type == "leaf":
            if cents_val is not None and cents_val != 0:
                bs_rows_export.append({"Section": "Liabilities & Equity", "Item": label, "Amount": format_money(dollars_val)})
        elif row_type == "operational":
            if transfers_gl_number is not None:
                bs_rows_export.append({"Section": "Liabilities & Equity", "Item": label, "Amount": format_money(dollars_val)})
        else:
            bs_rows_export.append({"Section": "Liabilities & Equity", "Item": label, "Amount": format_money(dollars_val)})
    for gl in equity_gls:
        bs_rows_export.append({"Section": "Liabilities & Equity", "Item": gl_to_name.get(gl, "Unknown account"), "Amount": format_money(balances_cents.get(gl, 0) / 100.0)})
    bs_rows_export.extend(
        [
            {"Section": "Liabilities & Equity", "Item": "Current Earnings (Net Income)", "Amount": format_money(current_earnings)},
            {"Section": "Liabilities & Equity", "Item": "Total Equity", "Amount": format_money(equity_total)},
            {"Section": "Liabilities & Equity", "Item": "Total Liabilities & Equity", "Amount": format_money(total_liabilities_equity)},
            {"Section": "Liabilities & Equity", "Item": "Net Worth", "Amount": format_money(net_worth)},
            {"Section": "Ratios & NI", "Item": "Net Income", "Amount": format_money(current_earnings)},
            {"Section": "Ratios & NI", "Item": "Liquid Assets / Assets (%)", "Amount": _format_pct(_pct(liquid_assets_ratio))},
            {"Section": "Ratios & NI", "Item": "Liquid Cash / Assets (%)", "Amount": _format_pct(_pct(liquid_cash_ratio))},
            {"Section": "Ratios & NI", "Item": "Debt / Assets (%)", "Amount": _format_pct(_pct(debt_assets_ratio))},
            {"Section": "Ratios & NI", "Item": "Debt / Liquid Cash (%)", "Amount": _format_pct(_pct(debt_liquid_cash_ratio))},
            {"Section": "Ratios & NI", "Item": "Fixed Assets / Assets (%)", "Amount": _format_pct(_pct(fixed_assets_ratio))},
        ]
    )
    df_bs_export = pd.DataFrame(bs_rows_export)

    total_income = total_income_c / 100.0
    total_expense = total_expense_c / 100.0
    net_income_mtd = ytd_net_income_c / 100.0
    df_is_export = pd.DataFrame(
        [
            *[
                {"Section": "Income", "GL Number": line["gl_number"], "Name": line["name"], "Amount": format_money(line["amount_cents"] / 100.0)}
                for line in income_lines
            ],
            *[
                {"Section": "Expense", "GL Number": line["gl_number"], "Name": line["name"], "Amount": format_money(line["amount_cents"] / 100.0)}
                for line in expense_lines
            ],
            {"Section": "Summary", "GL Number": "", "Name": "Total Income", "Amount": format_money(total_income)},
            {"Section": "Summary", "GL Number": "", "Name": "Total Expenses", "Amount": format_money(total_expense)},
            {"Section": "Summary", "GL Number": "", "Name": "Net Income (MTD)", "Amount": format_money(net_income_mtd)},
        ]
    )
    df_gl_export = pd.DataFrame(
        [
            {
                "GL Number": account.gl_number,
                "Name": account.name,
                "Type": account.type.value,
                "Normal Balance": account.normal_balance.value,
                "Balance": format_money(balances_cents.get(account.gl_number, 0) / 100.0),
            }
            for account in accounts
        ]
    ).sort_values("GL Number")

    mom_bs_df = None
    mom_is_df = None
    mom_ratios_df = None
    mom_error = None
    if report_type == "Historical (Month-End)":
        if not _is_month_end(as_of):
            mom_error = "As-of date for MoM financials must be a month-end date (e.g., 2025-10-31 or 2025-11-30)."
        else:
            conn = get_connection()
            try:
                mom_bs_df, mom_is_df, mom_ratios_df = compute_mom_financials_month_end(conn, as_of_str)
            except ValueError as exc:
                mom_error = str(exc)
            finally:
                conn.close()

    if report_type == "Historical (Month-End)":
        st.markdown(f"**Historical Financials (Month-End) through {as_of_str}**")
        if mom_error:
            st.error(mom_error)
        elif mom_bs_df is None or mom_is_df is None or mom_ratios_df is None:
            st.error("Unable to generate MoM financials.")
        else:
            st.markdown("#### Balance Sheet (MoM)")
            st.dataframe(_format_money_df_from_cents(mom_bs_df), use_container_width=True)
            st.markdown("#### Income Statement (MoM Summary)")
            st.dataframe(_format_money_df_from_cents(mom_is_df), use_container_width=True)
            st.markdown("#### Ratios (MoM)")
            st.dataframe(_format_ratio_df(mom_ratios_df), use_container_width=True)
            runtime.register_dataset(
                DATASET_FINANCIAL_REPORT,
                {"balance_sheet_mom": mom_bs_df, "income_statement_mom": mom_is_df, "ratios_mom": mom_ratios_df},
                kind="report_bundle",
                description="Historical month-end financial reports",
                metadata={"report_type": report_type},
            )
    elif report_type == "Trends & Outliers":
        st.markdown("**Trends & Outliers**")
        st.caption("Interactive time-series charts by Effective Date. Daily view is capped at 365 days for performance. Outliers are highlighted using a robust z-score on period-to-period variance.")
        series_mode = st.radio("Series type", options=["GL Account", "BS Subtotal", "Headline Total"], horizontal=True, key="trend_series_mode")
        freq_label = st.radio("Frequency", options=["Monthly (Month-End)", "Daily"], horizontal=True, key="trend_frequency")
        frequency = "MONTHLY" if freq_label.startswith("Monthly") else "DAILY"
        today = date.today()
        conn = get_connection()
        try:
            fin_start_str = get_financials_start_date(conn)
        except Exception:
            fin_start_str = None
        finally:
            conn.close()
        fin_start_date = None
        if fin_start_str:
            try:
                fin_start_date = datetime.strptime(fin_start_str, "%Y-%m-%d").date()
            except Exception:
                fin_start_date = None

        if frequency == "MONTHLY":
            default_end = _last_day_of_month(today) if _is_month_end(today) else _last_day_of_month(today.replace(day=1) - timedelta(days=1))
            default_start = _shift_month_end(default_end, -24)
            if fin_start_date is not None:
                fin_start_month_end = _last_day_of_month(fin_start_date)
                if fin_start_month_end > default_start:
                    default_start = fin_start_month_end
            c1, c2 = st.columns(2)
            with c1:
                start_d = st.date_input("Start (month-end)", value=default_start, key="trend_start_me")
            with c2:
                end_d = st.date_input("End (month-end)", value=default_end, key="trend_end_me")
            if not _is_month_end(start_d) or not _is_month_end(end_d):
                st.error("Monthly view requires strict calendar month-end dates (e.g., 2025-10-31).")
                return
            if start_d > end_d:
                st.error("Start date must be on or before end date.")
                return
        else:
            default_end = today
            default_start = today - timedelta(days=365)
            if fin_start_date is not None and fin_start_date > default_start:
                default_start = fin_start_date
            c1, c2 = st.columns(2)
            with c1:
                start_d = st.date_input("Start", value=default_start, key="trend_start_daily")
            with c2:
                end_d = st.date_input("End", value=default_end, key="trend_end_daily")
            if start_d > end_d:
                st.error("Start date must be on or before end date.")
                return
            if (end_d - start_d).days > 365:
                st.error("Daily view is capped at 365 days for performance. Please shorten the range or use Monthly.")
                return

        start_str_ts = start_d.isoformat()
        end_str_ts = end_d.isoformat()
        bs_label_to_key = {}
        if series_mode == "BS Subtotal":
            conn = get_connection()
            try:
                for key, meta in BS_CATEGORIES.items():
                    gls = get_bs_category_gl_numbers(conn, key) or []
                    if gls:
                        bs_label_to_key[meta.get("label", key)] = key
            finally:
                conn.close()
        headline_options = ["Total Assets", "Total Liabilities", "Total Income", "Total Expense", "Net Income", "Net Worth"]
        title_prefix = ""
        gls_primary = []
        gls_secondary = None
        combine_mode = None
        if series_mode == "GL Account":
            acct_labels = [f"{account.gl_number} – {account.name}" for account in accounts]
            selected_label = st.selectbox("Account", options=acct_labels, key="trend_gl_select")
            gls_primary = [int(selected_label.split(" – ", 1)[0].strip())]
            title_prefix = selected_label
        elif series_mode == "BS Subtotal":
            if not bs_label_to_key:
                st.info("No Balance Sheet subtotals are mapped in Preferences yet.")
                return
            selected_bs_label = st.selectbox("Balance Sheet subtotal", options=sorted(bs_label_to_key.keys(), key=lambda value: value.lower()), key="trend_bs_select")
            conn = get_connection()
            try:
                gls_primary = get_bs_category_gl_numbers(conn, bs_label_to_key[selected_bs_label]) or []
            finally:
                conn.close()
            title_prefix = f"BS: {selected_bs_label}"
        else:
            selected_headline = st.selectbox("Headline series", options=headline_options, key="trend_headline_select")
            assets_gls = [account.gl_number for account in accounts if account.type.value == "ASSET"]
            liab_gls = [account.gl_number for account in accounts if account.type.value == "LIABILITY"]
            income_gls = [account.gl_number for account in accounts if account.type.value == "INCOME"]
            expense_gls = [account.gl_number for account in accounts if account.type.value == "EXPENSE"]
            if selected_headline == "Total Assets":
                gls_primary, title_prefix = assets_gls, "Total Assets"
            elif selected_headline == "Total Liabilities":
                gls_primary, title_prefix = liab_gls, "Total Liabilities"
            elif selected_headline == "Total Income":
                gls_primary, title_prefix = income_gls, "Total Income"
            elif selected_headline == "Total Expense":
                gls_primary, title_prefix = expense_gls, "Total Expense"
            elif selected_headline == "Net Income":
                gls_primary, gls_secondary, combine_mode, title_prefix = income_gls, expense_gls, "SUBTRACT", "Net Income"
            else:
                gls_primary, gls_secondary, combine_mode, title_prefix = assets_gls, liab_gls, "SUBTRACT", "Net Worth (Assets − Liabilities)"

        if not gls_primary:
            st.info("No GLs available for the selected series.")
            return

        @st.cache_data(show_spinner=False, ttl=120)
        def _cached_series(gls: tuple[int, ...], start_s: str, end_s: str, freq: str) -> pd.DataFrame:
            conn = get_connection()
            try:
                return compute_balance_time_series(conn, list(gls), start_s, end_s, frequency=freq)
            finally:
                conn.close()

        df1 = _cached_series(tuple(sorted(set(gls_primary))), start_str_ts, end_str_ts, frequency)
        if combine_mode == "SUBTRACT" and gls_secondary is not None:
            df2 = _cached_series(tuple(sorted(set(gls_secondary))), start_str_ts, end_str_ts, frequency)
            dfm = pd.merge(df1, df2, on="date", how="outer", suffixes=("_p", "_s")).sort_values("date").fillna(0)
            dfm["balance_cents"] = (dfm["balance_cents_p"].astype(int) - dfm["balance_cents_s"].astype(int)).astype(int)
            dfm["delta_cents"] = dfm["balance_cents"].diff().fillna(0).astype(int)
            df_series = dfm[["date", "balance_cents", "delta_cents"]]
        else:
            df_series = df1
        if df_series.empty:
            st.info("No data available for the selected range.")
            return
        df_plot = df_series.copy()
        df_plot["balance"] = df_plot["balance_cents"].astype(float) / 100.0
        df_plot["delta"] = df_plot["delta_cents"].astype(float) / 100.0
        df_plot["is_outlier"] = _robust_outlier_flags(df_plot["delta"], z_thresh=3.5, fallback_top_n=3)
        _chart_balance_and_variance(df_plot[["date", "balance", "delta", "is_outlier"]].sort_values("date"), title_prefix)
    elif report_type == "Balance Sheet":
        st.markdown(f"**Simplified Balance Sheet as of {as_of_str}**")
        assets_rows = []
        for label, cents_val, dollars_val, row_type in asset_rows_ordered:
            if row_type == "leaf":
                if cents_val is not None and cents_val != 0:
                    assets_rows.append({"Item": label, "Amount": format_money(dollars_val)})
            else:
                assets_rows.append({"Item": label, "Amount": format_money(dollars_val)})
        st.markdown("#### Assets")
        st.table(pd.DataFrame(assets_rows))
        liab_rows = []
        for label, cents_val, dollars_val, row_type in liab_rows_ordered:
            if row_type == "leaf":
                if cents_val is not None and cents_val != 0:
                    liab_rows.append({"Item": label, "Amount": format_money(dollars_val)})
            elif row_type == "operational":
                if transfers_gl_number is not None:
                    liab_rows.append({"Item": label, "Amount": format_money(dollars_val)})
            else:
                liab_rows.append({"Item": label, "Amount": format_money(dollars_val)})
        for gl in equity_gls:
            liab_rows.append({"Item": gl_to_name.get(gl, "Unknown account"), "Amount": format_money(balances_cents.get(gl, 0) / 100.0)})
        liab_rows.extend(
            [
                {"Item": "Current Earnings (Net Income)", "Amount": format_money(current_earnings)},
                {"Item": "Total Equity", "Amount": format_money(equity_total)},
                {"Item": "Total Liabilities & Equity", "Amount": format_money(total_liabilities_equity)},
                {"Item": "Net Worth", "Amount": format_money(net_worth)},
            ]
        )
        st.markdown("#### Liabilities & Equity")
        st.table(pd.DataFrame(liab_rows))
        st.markdown("#### Net Income & Ratios")
        st.table(
            pd.DataFrame(
                [
                    {"Item": "Net Income", "Amount": format_money(current_earnings)},
                    {"Item": "Liquid Assets / Assets", "Amount": format_ratio(liquid_assets_ratio)},
                    {"Item": "Liquid Cash / Assets", "Amount": format_ratio(liquid_cash_ratio)},
                    {"Item": "Cash Available Now", "Amount": format_money(liquid_cash)},
                    {"Item": "Debt / Assets", "Amount": format_ratio(debt_assets_ratio)},
                    {"Item": "Debt / Liquid Cash", "Amount": format_ratio(debt_liquid_cash_ratio)},
                    {"Item": "Fixed Assets / Assets", "Amount": format_ratio(fixed_assets_ratio)},
                    {"Item": "Cash on Hand (Years)", "Amount": "" if cash_on_hand_years is None else f"{cash_on_hand_years:.2f}x"},
                ]
            )
        )
        runtime.register_dataset(
            DATASET_FINANCIAL_REPORT,
            {"balance_sheet": df_bs_export, "income_statement": df_is_export, "gl_balances": df_gl_export},
            kind="report_bundle",
            description="Current financial statement export tables",
            metadata={"report_type": report_type},
        )
    elif report_type == "GL Summary View":
        st.markdown(f"**GL Balances as of {as_of_str}**")
        show_non_zero = st.checkbox("Hide zero balance accounts", value=False, key="gl_summary_show_non_zero")
        df_gl = df_gl_export.copy()
        if show_non_zero:
            df_gl = df_gl[df_gl["Balance"] != format_money(0.0)]
        st.dataframe(df_gl, use_container_width=True)
        totals_data = [
            {"Type": account_type.title(), "Total Balance": format_money(sum(balances_cents.get(account.gl_number, 0) for account in accounts if account.type.value == account_type) / 100.0)}
            for account_type in ["ASSET", "LIABILITY", "EQUITY", "INCOME", "EXPENSE"]
        ]
        st.markdown("#### Totals by Account Type")
        st.table(pd.DataFrame(totals_data))
        runtime.register_dataset(
            DATASET_FINANCIAL_REPORT,
            df_gl,
            kind="dataframe",
            description="GL summary view",
            metadata={"report_type": report_type, "rows": len(df_gl)},
        )
    elif report_type == "Income Statement":
        st.markdown(f"**Income Statement (Month-to-Date through {as_of_str})**")
        if income_lines:
            income_rows = [{"Name": line["name"], "Amount": format_money(line["amount_cents"] / 100.0)} for line in income_lines]
            income_rows.append({"Name": "Total Income", "Amount": format_money(total_income)})
            st.markdown("#### Income")
            st.table(pd.DataFrame(income_rows))
        else:
            st.info("No income recorded for the selected year-to-date period.")
        if expense_lines:
            expense_rows = [{"Name": line["name"], "Amount": format_money(line["amount_cents"] / 100.0)} for line in expense_lines]
            expense_rows.append({"Name": "Total Expenses", "Amount": format_money(total_expense)})
            st.markdown("#### Expenses")
            st.table(pd.DataFrame(expense_rows))
        else:
            st.info("No expenses recorded for the selected year-to-date period.")
        summary_df = pd.DataFrame(
            [
                {"Item": "Total Income", "Amount": format_money(total_income)},
                {"Item": "Total Expenses", "Amount": format_money(total_expense)},
                {"Item": "Net Income (MTD)", "Amount": format_money(net_income_mtd)},
            ]
        )
        st.markdown("#### Summary")
        st.table(summary_df)
        runtime.register_dataset(
            DATASET_FINANCIAL_REPORT,
            summary_df,
            kind="dataframe",
            description="Income statement summary",
            metadata={"report_type": report_type, "rows": len(summary_df)},
        )

    export_buffer = io.BytesIO()
    with pd.ExcelWriter(export_buffer, engine="xlsxwriter") as writer:
        if report_type == "Historical (Month-End)" and not mom_error and mom_bs_df is not None:
            _format_money_df_from_cents(mom_bs_df).to_excel(writer, sheet_name="Balance Sheet (MoM)")
            _format_money_df_from_cents(mom_is_df).to_excel(writer, sheet_name="Income Statement (MoM)")
            _format_ratio_df(mom_ratios_df).to_excel(writer, sheet_name="Ratios (MoM)")
        else:
            df_bs_export.to_excel(writer, sheet_name="Balance Sheet", index=False)
            df_is_export.to_excel(writer, sheet_name="Income Statement", index=False)
            df_gl_export.to_excel(writer, sheet_name="GL Balances", index=False)
    export_buffer.seek(0)
    export_name = f"financial_statements_mom_{as_of_str}.xlsx" if report_type == "Historical (Month-End)" else f"financial_statements_{as_of_str}.xlsx"
    st.download_button(
        label="Download Financial Statements",
        data=export_buffer,
        file_name=export_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def render_upload_transactions_tab(runtime) -> None:
    st.subheader("Upload Transactions")
    st.markdown("Use this view to import transactions and post them as two-sided journal entries (bank/credit card side plus offset GL). Mappings are created in Preferences.")

    conn = get_connection()
    try:
        user_mappings_active = get_user_upload_mappings(conn, include_inactive=False)
    finally:
        conn.close()

    custom_options = [f"Custom: {mapping.name}" for mapping in (user_mappings_active or [])]
    institution = st.selectbox(
        "Institution / mapping",
        options=custom_options,
        index=0 if custom_options else None,
        placeholder="Create an institution mapping in Preferences first",
    )
    if not custom_options:
        st.info("No active institution mappings found. Create one in Preferences → Institution mappings.")
        uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"], accept_multiple_files=False, key="txn_upload_file", disabled=True)
    else:
        uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"], accept_multiple_files=False, key="txn_upload_file")

    conn = get_connection()
    try:
        all_accounts = get_accounts(conn, include_inactive=False)
        keyword_rules = get_txn_keyword_rules(conn, include_inactive=False)
    finally:
        conn.close()

    if not all_accounts:
        st.warning("No accounts available. Define accounts first in Chart of Accounts.")
        return

    offset_account_labels = []
    label_to_id = {}
    for account in all_accounts:
        label = f"{account.gl_number} – {account.name}"
        offset_account_labels.append(label)
        label_to_id[label] = account.id
    gl_to_label = {account.gl_number: f"{account.gl_number} – {account.name}" for account in all_accounts}

    def suggest_offset_gl(description: str) -> str:
        desc = (description or "").lower()
        if not desc:
            return ""
        for rule in keyword_rules:
            if rule.keyword.lower() in desc:
                return gl_to_label.get(rule.gl_number, "")
        return ""

    st.markdown("### 3. Load and review transactions")

    def stage_transactions(records: list[dict], bank_gl: int, source: str, institution_label: str) -> None:
        if not records:
            st.info("No non-empty rows found in the uploaded file.")
            return
        df_stage = pd.DataFrame(records)
        if "Offset GL" not in df_stage.columns:
            df_stage["Offset GL"] = ""
        runtime.state.set_upload_stage_df(df_stage)
        runtime.state.set_upload_stage_meta(UploadStageMeta(bank_gl=bank_gl, source=source, institution=institution_label))
        runtime.register_dataset(
            DATASET_UPLOAD_STAGE,
            df_stage,
            kind="dataframe",
            description="Staged upload transactions",
            metadata={"rows": len(df_stage), "institution": institution_label},
        )
        st.success(f"Loaded {len(records)} transactions from {institution_label}.")

    if uploaded_file is not None and institution.startswith("Custom: ") and st.button("Load transactions", key="load_custom_mapping_btn"):
        try:
            mapping_name = institution.replace("Custom: ", "", 1).strip()
            conn = get_connection()
            try:
                mapping = get_user_upload_mapping_by_name(conn, mapping_name)
            finally:
                conn.close()
            if mapping is None:
                raise ValueError(f"Custom mapping '{mapping_name}' not found (it may be archived).")
            txns = load_user_defined_file(uploaded_file, mapping)
            records = []
            for idx, txn in enumerate(txns):
                record = {
                    "id": idx,
                    "Include": True,
                    "Posting Date": date.today().isoformat(),
                    "Effective Date": txn.effective_date,
                    "Description": txn.description,
                    "Amount": format_money((1 if txn.is_debit_for_account else -1) * (txn.amount_cents / 100.0)),
                    "_amount_cents": txn.amount_cents,
                    "_is_debit_for_account": txn.is_debit_for_account,
                }
                suggested_label = suggest_offset_gl(txn.description)
                if suggested_label:
                    record["Offset GL"] = suggested_label
                records.append(record)
            stage_transactions(records, int(mapping.base_gl_number), f"USERMAP_{mapping.name}", f"Custom mapping: {mapping.name}")
        except Exception as exc:
            st.error(f"Could not load custom mapping file: {exc}")

    upload_stage_df = runtime.state.get_upload_stage_df()
    upload_stage_meta = runtime.state.get_upload_stage_meta()
    if upload_stage_df is None:
        st.info("Upload a file and click 'Load transactions' to begin.")
        return

    st.markdown("### 4. Review and assign GL accounts")
    with st.form("txn_stage_form"):
        edited_df = st.data_editor(
            upload_stage_df,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "Include": st.column_config.CheckboxColumn("Include", help="Check to post this transaction", default=True),
                "Amount": st.column_config.TextColumn("Amount", help="Transaction amount", disabled=True),
                "Offset GL": st.column_config.SelectboxColumn("Offset GL", help="GL account for the non-bank side of the entry", options=offset_account_labels),
                "_amount_cents": st.column_config.NumberColumn("_amount_cents", disabled=True, help="Internal amount in cents"),
                "_is_debit_for_account": st.column_config.CheckboxColumn("_is_debit_for_account", disabled=True, help="Internal flag: True = debit bank/CC, False = credit bank/CC"),
                "id": st.column_config.NumberColumn("id", disabled=True),
            },
            hide_index=True,
            key="upload_stage_editor",
        )
        st.markdown("### 5. Post approved transactions to the GL")
        post_clicked = st.form_submit_button("Post approved transactions")

    runtime.register_dataset(
        DATASET_UPLOAD_STAGE,
        edited_df,
        kind="dataframe",
        description="Editable staged upload transactions",
        metadata={"rows": len(edited_df), "institution": upload_stage_meta.institution if upload_stage_meta else None},
    )

    if post_clicked:
        rows_to_post = edited_df[(edited_df["Include"] == True) & (edited_df["Offset GL"] != "")]
        if rows_to_post.empty:
            st.error("No transactions selected to post. Check Include and Offset GL columns.")
            return
        if upload_stage_meta is None:
            st.error("Internal error: bank/credit card GL number is not set for this upload.")
            return
        conn = get_connection()
        try:
            accounts = get_accounts(conn, include_inactive=False)
            bank_account = next((account for account in accounts if account.gl_number == upload_stage_meta.bank_gl), None)
            if bank_account is None:
                st.error(f"Bank / credit card account (GL {upload_stage_meta.bank_gl}) not found.")
                return
            posted_count = 0
            errors = []
            for _, row in rows_to_post.iterrows():
                try:
                    amount_cents = int(row["_amount_cents"])
                    if amount_cents <= 0:
                        continue
                    offset_account_id = label_to_id.get(row["Offset GL"])
                    if offset_account_id is None:
                        raise ValueError(f"Offset GL '{row['Offset GL']}' is not valid.")
                    lines = [
                        JournalEntryLine(account_id=bank_account.id, amount_cents=amount_cents, is_debit=bool(row["_is_debit_for_account"]), memo=row["Description"]),
                        JournalEntryLine(account_id=offset_account_id, amount_cents=amount_cents, is_debit=not bool(row["_is_debit_for_account"]), memo=row["Description"]),
                    ]
                    insert_journal_entry(
                        conn=conn,
                        effective_date=row["Effective Date"],
                        post_date=row["Posting Date"],
                        description=row["Description"] or f"{upload_stage_meta.institution} transaction",
                        source=upload_stage_meta.source,
                        lines=lines,
                    )
                    posted_count += 1
                except Exception as exc:
                    errors.append(str(exc))
            if posted_count > 0:
                runtime.state.set_active_tab_hint("upload_transactions")
                runtime.state.append_ui_event("upload_post")
                st.success(f"Posted {posted_count} journal entries from {upload_stage_meta.institution}.")
                runtime.state.clear_upload_stage()
                runtime.datasets.clear(DATASET_UPLOAD_STAGE)
            if errors:
                st.error("Some transactions could not be posted:\n" + "\n".join(f"- {message}" for message in errors))
        finally:
            conn.close()


def render_preferences_tab(runtime) -> None:
    st.subheader("General Preferences")
    st.markdown("---")
    st.markdown("##### App Access Password")
    with st.expander("App access password (optional)"):
        current_password = load_stored_password()
        st.markdown("Use this section to set or clear the app access password.")
        col_pw1, col_pw2 = st.columns(2)
        with col_pw1:
            new_password = st.text_input("New password", type="password", key="prefs_app_access_pw_new", help="Enter a new password to require at app startup.")
        with col_pw2:
            confirm_password = st.text_input("Confirm new password", type="password", key="prefs_app_access_pw_confirm", help="Re-enter the new password to confirm.")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Save password", key="prefs_app_access_pw_save"):
                if not new_password:
                    st.error("Password cannot be empty. If you want to remove the password, use 'Clear password' instead.")
                elif new_password != confirm_password:
                    st.error("New password and confirmation do not match.")
                else:
                    conn = get_connection()
                    try:
                        set_app_access_password(conn, new_password)
                    finally:
                        conn.close()
                    runtime.state.append_ui_event("password_save")
                    st.success("App access password has been set/updated.")
                    st.rerun()
        with col_btn2:
            if st.button("Clear password", key="prefs_app_access_pw_clear"):
                conn = get_connection()
                try:
                    set_app_access_password(conn, None)
                finally:
                    conn.close()
                runtime.state.append_ui_event("password_clear")
                st.success("App access password has been cleared. No password will be required at startup.")
                st.rerun()
        st.info(
            "An app access password is currently set. When the startup gate is enabled, you will need to enter this password to access the app."
            if current_password
            else "No app access password is currently set. When the startup gate is enabled, you will be able to open the app without entering a password."
        )

    conn = get_connection()
    try:
        existing_closed_str = get_last_closed_period_end(conn)
        existing_pin = get_override_pin(conn)
        existing_closed_date = datetime.strptime(existing_closed_str, "%Y-%m-%d").date() if existing_closed_str else None
    except Exception:
        existing_closed_date = None
        existing_closed_str = None
        existing_pin = None
    finally:
        conn.close()

    st.markdown("##### Closed Period & Override PIN")
    with st.expander("Update closed period & override PIN"):
        closed_date_input = st.date_input("Last closed period end date", value=existing_closed_date or date.today(), help="Entries with effective dates on or before this will require an override PIN.")
        new_pin_input = st.text_input("Override PIN", type="password", help="Required to post entries into closed periods. Leave blank to keep the existing PIN.")
        if st.button("Save", key="closed_period_settings_save"):
            conn = get_connection()
            try:
                set_last_closed_period_end(conn, closed_date_input.isoformat())
                if new_pin_input.strip():
                    set_override_pin(conn, new_pin_input.strip())
            finally:
                conn.close()
            runtime.state.append_ui_event("period_lock_save")
            st.success("Period lock settings saved.")
            st.rerun()
        st.info(f"Current closed period end date: {existing_closed_str}" if existing_closed_str else "Current closed period end date: not set")
        st.info("Current override PIN: (set)" if existing_pin else "Current override PIN: not set")

    conn = get_connection()
    try:
        accounts = get_accounts(conn, include_inactive=False)
        retained_gl_number = get_retained_earnings_gl_number(conn)
        suspense_gl_number = get_suspense_gl_number(conn)
        transfers_clearing_gl_number = get_transfers_clearing_gl_number(conn)
    finally:
        conn.close()

    st.markdown("##### Operational GL Accounts")
    with st.expander("Select or update operational GL accounts"):
        equity_accounts = [account for account in accounts if account.type.value == "EQUITY"]
        if equity_accounts:
            equity_accounts_sorted = sorted(equity_accounts, key=lambda account: account.gl_number)
            equity_labels = [f"{account.gl_number} – {account.name}" for account in equity_accounts_sorted]
            label_to_equity_gl = {f"{account.gl_number} – {account.name}": account.gl_number for account in equity_accounts_sorted}
            default_index = next((idx for idx, account in enumerate(equity_accounts_sorted) if account.gl_number == retained_gl_number), 0) if retained_gl_number is not None else 0
            with st.form("retained_earnings_form"):
                selected_equity_label = st.selectbox("Select retained earnings equity account.", options=equity_labels, index=default_index, key="retained_earnings_select")
                save_re_btn = st.form_submit_button("Save retained earnings account")
            if save_re_btn:
                conn = get_connection()
                try:
                    set_retained_earnings_gl_number(conn, label_to_equity_gl[selected_equity_label])
                finally:
                    conn.close()
                runtime.state.append_ui_event("retained_earnings_save")
                st.success(f"Retained earnings account set to GL {label_to_equity_gl[selected_equity_label]}.")

        st.divider()
        asset_liability_accounts = [account for account in accounts if account.type.value in ("ASSET", "LIABILITY")]
        asset_liability_accounts_sorted = sorted(asset_liability_accounts, key=lambda account: account.gl_number)
        asset_liability_labels = [f"{account.gl_number} – {account.name}" for account in asset_liability_accounts_sorted]
        label_to_gl = {f"{account.gl_number} – {account.name}": account.gl_number for account in asset_liability_accounts_sorted}
        optional_labels = ["(not set)"] + asset_liability_labels
        if asset_liability_accounts_sorted:
            suspense_default_index = 0
            if suspense_gl_number is not None:
                suspense_label = next((label for label, gl in label_to_gl.items() if gl == suspense_gl_number), None)
                if suspense_label is not None:
                    suspense_default_index = optional_labels.index(suspense_label)
            with st.form("suspense_account_form"):
                selected_suspense_label = st.selectbox("Select suspense account (optional)", options=optional_labels, index=suspense_default_index, key="prefs_suspense_select")
                save_suspense_btn = st.form_submit_button("Save suspense account")
            if save_suspense_btn:
                conn = get_connection()
                try:
                    set_suspense_gl_number(conn, None if selected_suspense_label == "(not set)" else label_to_gl[selected_suspense_label])
                finally:
                    conn.close()
                runtime.state.append_ui_event("suspense_save")
                st.success("Suspense account updated.")
            transfers_default_index = 0
            if transfers_clearing_gl_number is not None:
                transfers_label = next((label for label, gl in label_to_gl.items() if gl == transfers_clearing_gl_number), None)
                if transfers_label is not None:
                    transfers_default_index = optional_labels.index(transfers_label)
            with st.form("transfers_clearing_form"):
                selected_transfers_label = st.selectbox("Select transfers clearing account (optional)", options=optional_labels, index=transfers_default_index, key="prefs_transfers_clearing_select")
                save_transfers_btn = st.form_submit_button("Save transfers clearing account")
            if save_transfers_btn:
                conn = get_connection()
                try:
                    set_transfers_clearing_gl_number(conn, None if selected_transfers_label == "(not set)" else label_to_gl[selected_transfers_label])
                finally:
                    conn.close()
                runtime.state.append_ui_event("transfers_clearing_save")
                st.success("Transfers clearing account updated.")

    st.markdown("##### Month End Closing Checklist")
    with st.expander("Month End Closing Checklist"):
        conn = get_connection()
        try:
            active_accounts = get_accounts(conn, include_inactive=False)
            month_end_checklists_enabled = get_month_end_closing_checklist_enabled(conn)
            existing_checklists = get_month_end_checklists(conn, include_inactive=True)
        finally:
            conn.close()
        active_accounts_sorted = sorted(active_accounts, key=lambda account: (account.gl_number, account.id))
        account_label_to_id = {f"{account.gl_number} – {account.name}": account.id for account in active_accounts_sorted}
        account_id_to_label = {account.id: f"{account.gl_number} – {account.name}" for account in active_accounts_sorted}
        account_options = list(account_label_to_id.keys())
        enabled_value = st.checkbox("Enable Month End Closing Checklist", value=month_end_checklists_enabled, key="pref_month_end_checklists_enabled")
        if st.button("Save Checklist Settings", key="pref_month_end_checklists_save_settings"):
            conn = get_connection()
            try:
                set_month_end_closing_checklist_enabled(conn, enabled_value)
            finally:
                conn.close()
            runtime.state.append_ui_event("month_end_settings_save")
            st.success("Month End Closing Checklist settings saved.")
            st.rerun()
        st.divider()
        new_checklist_name = st.text_input("Checklist Name", key="pref_month_end_new_checklist_name")
        new_checklist_accounts = st.multiselect("Accounts", options=account_options, key="pref_month_end_new_checklist_accounts")
        if st.button("Create Checklist", key="pref_month_end_create_checklist"):
            try:
                checklist_name_clean = new_checklist_name.strip()
                if not checklist_name_clean:
                    raise ValueError("Checklist name cannot be empty.")
                if not new_checklist_accounts:
                    raise ValueError("Select at least one account for the checklist.")
                selected_account_ids = [account_label_to_id[label] for label in new_checklist_accounts if label in account_label_to_id]
                conn = get_connection()
                try:
                    create_month_end_checklist(conn, name=checklist_name_clean, account_ids=selected_account_ids)
                finally:
                    conn.close()
                runtime.state.append_ui_event("month_end_checklist_create")
                st.success(f"Checklist '{checklist_name_clean}' created.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.divider()
        if existing_checklists:
            checklist_option_labels = [checklist.name if checklist.is_active else f"{checklist.name} (Archived)" for checklist in existing_checklists]
            checklist_label_to_id = {label: checklist.id for label, checklist in zip(checklist_option_labels, existing_checklists)}
            selected_checklist_label = st.selectbox("Select Checklist", options=checklist_option_labels, key="pref_month_end_manage_checklist_select")
            selected_checklist_id = checklist_label_to_id[selected_checklist_label]
            selected_checklist = next(checklist for checklist in existing_checklists if checklist.id == selected_checklist_id)
            conn = get_connection()
            try:
                current_account_ids = get_month_end_checklist_account_ids(conn, selected_checklist_id)
            finally:
                conn.close()
            default_account_labels = [account_id_to_label[acct_id] for acct_id in current_account_ids if acct_id in account_id_to_label]
            managed_accounts = st.multiselect("Checklist Accounts", options=account_options, default=default_account_labels, key=f"pref_month_end_manage_accounts_{selected_checklist_id}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Checklist Accounts", key=f"pref_month_end_save_accounts_{selected_checklist_id}"):
                    try:
                        if not managed_accounts:
                            raise ValueError("Select at least one account for the checklist.")
                        selected_account_ids = [account_label_to_id[label] for label in managed_accounts if label in account_label_to_id]
                        conn = get_connection()
                        try:
                            update_month_end_checklist_accounts(conn, checklist_id=selected_checklist_id, account_ids=selected_account_ids)
                        finally:
                            conn.close()
                        runtime.state.append_ui_event("month_end_checklist_accounts_save")
                        st.success("Checklist accounts updated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            with col2:
                button_label = "Archive Checklist" if selected_checklist.is_active else "Unarchive Checklist"
                if st.button(button_label, key=f"pref_month_end_archive_{selected_checklist_id}"):
                    conn = get_connection()
                    try:
                        set_month_end_checklist_active(conn, selected_checklist_id, not selected_checklist.is_active)
                    finally:
                        conn.close()
                    runtime.state.append_ui_event("month_end_checklist_archive")
                    st.success("Checklist updated.")
                    st.rerun()

    st.markdown("---")
    st.subheader("Import Transactions")
    st.markdown("---")
    st.markdown("##### Auto-mapping rules for imported transactions")
    conn = get_connection()
    try:
        accounts = get_accounts(conn, include_inactive=False)
        rules = get_txn_keyword_rules(conn, include_inactive=True)
    finally:
        conn.close()
    if accounts:
        gl_label_to_number = {f"{account.gl_number} – {account.name}": account.gl_number for account in accounts}
        gl_number_to_label = {account.gl_number: f"{account.gl_number} – {account.name}" for account in accounts}
        with st.expander("Add or update keyword rule", expanded=False):
            new_keyword = st.text_input("Keyword (case-insensitive substring)", key="kw_rule_keyword", help="Example: 'Netflix', 'Spotify', 'Uber'")
            new_gl_label = st.selectbox("Offset GL to suggest", options=sorted(gl_label_to_number.keys()), key="kw_rule_gl_label")
            col_active, col_prio = st.columns([1, 1])
            with col_active:
                new_active = st.checkbox("Active", value=True, key="kw_rule_active")
            with col_prio:
                new_priority = st.number_input("Priority (higher evaluated first)", min_value=0, max_value=1000, value=0, step=1, key="kw_rule_priority")
            if st.button("Save", key="kw_rule_save_btn"):
                keyword_clean = (new_keyword or "").strip()
                if not keyword_clean:
                    st.error("Keyword cannot be empty.")
                else:
                    conn = get_connection()
                    try:
                        upsert_txn_keyword_rule(conn=conn, keyword=keyword_clean, gl_number=gl_label_to_number[new_gl_label], is_active=new_active, priority=int(new_priority))
                    finally:
                        conn.close()
                    runtime.state.append_ui_event("keyword_rule_save")
                    st.success(f"Saved keyword rule for '{keyword_clean}' → {gl_label_to_number[new_gl_label]}.")
        with st.expander("View or delete existing keyword rules"):
            if not rules:
                st.info("No keyword rules defined yet.")
            else:
                df_rules = pd.DataFrame(
                    [
                        {
                            "ID": rule.id,
                            "Keyword": rule.keyword,
                            "GL Number": rule.gl_number,
                            "GL Account": gl_number_to_label.get(rule.gl_number, str(rule.gl_number)),
                            "Active": rule.is_active,
                            "Priority": rule.priority,
                            "Delete": False,
                        }
                        for rule in rules
                    ]
                )
                with st.form("kw_rules_form"):
                    edited_rules = st.data_editor(df_rules, use_container_width=True, hide_index=True, column_config={"Delete": st.column_config.CheckboxColumn("Delete", help="Checked rows will be deleted when you save changes.", default=False)}, key="kw_rules_editor")
                    save_rules = st.form_submit_button("Save")
                if save_rules:
                    to_delete = edited_rules[edited_rules["Delete"] == True]
                    if to_delete.empty:
                        st.info("No rules selected for deletion.")
                    else:
                        conn = get_connection()
                        try:
                            for rule_id in to_delete["ID"]:
                                delete_txn_keyword_rule(conn, int(rule_id))
                        finally:
                            conn.close()
                        runtime.state.append_ui_event("keyword_rule_delete")
                        st.success(f"Deleted {len(to_delete)} rule(s).")
                        st.rerun()

    st.markdown("##### Institution mappings")
    with st.expander("Create and manage custom upload mappings"):
        conn = get_connection()
        try:
            accounts_active = get_accounts(conn, include_inactive=False)
            existing_mappings_all = get_user_upload_mappings(conn, include_inactive=True)
        finally:
            conn.close()
        if accounts_active:
            gl_labels = [f"{account.gl_number} – {account.name}" for account in accounts_active]
            label_to_gl = {f"{account.gl_number} – {account.name}": account.gl_number for account in accounts_active}
            with st.form("create_user_upload_mapping_form"):
                new_name = st.text_input("Mapping name (must be unique)", value="")
                base_gl_label = st.selectbox("Base GL (the bank/credit card account this file belongs to)", options=gl_labels, index=0)
                positive_side = st.selectbox("Positive amounts are posted to the Base GL as", options=["DEBIT", "CREDIT"], index=0)
                col_effective = st.text_input("Effective Date column header", value="Date")
                col_amount = st.text_input("Amount column header", value="Amount")
                col_desc = st.text_input("Description column header", value="Description")
                create_clicked = st.form_submit_button("Save mapping")
            if create_clicked:
                try:
                    conn = get_connection()
                    try:
                        create_user_upload_mapping(
                            conn=conn,
                            name=new_name,
                            base_gl_number=int(label_to_gl[base_gl_label]),
                            positive_is_debit=(positive_side == "DEBIT"),
                            effective_date_col=col_effective,
                            amount_col=col_amount,
                            description_col=col_desc,
                        )
                    finally:
                        conn.close()
                    runtime.state.append_ui_event("upload_mapping_create")
                    st.success(f"Saved mapping '{new_name}'. It will appear in Upload Transactions.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if existing_mappings_all:
                st.dataframe(
                    [
                        {
                            "Name": mapping.name,
                            "Base GL": mapping.base_gl_number,
                            "Positive is": "DEBIT" if mapping.positive_is_debit else "CREDIT",
                            "Effective Date Col": mapping.effective_date_col,
                            "Amount Col": mapping.amount_col,
                            "Description Col": mapping.description_col,
                            "Active": bool(mapping.is_active),
                        }
                        for mapping in existing_mappings_all
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                selected_name = st.selectbox("Select a mapping to archive/unarchive", options=[mapping.name for mapping in existing_mappings_all], index=0)
                selected_obj = next((mapping for mapping in existing_mappings_all if mapping.name == selected_name), None)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Archive mapping", disabled=(selected_obj is None or not selected_obj.is_active)):
                        conn = get_connection()
                        try:
                            set_user_upload_mapping_active(conn, selected_obj.id, False)
                        finally:
                            conn.close()
                        runtime.state.append_ui_event("upload_mapping_archive")
                        st.success(f"Archived '{selected_name}'.")
                        st.rerun()
                with col2:
                    if st.button("Unarchive mapping", disabled=(selected_obj is None or selected_obj.is_active)):
                        conn = get_connection()
                        try:
                            set_user_upload_mapping_active(conn, selected_obj.id, True)
                        finally:
                            conn.close()
                        runtime.state.append_ui_event("upload_mapping_unarchive")
                        st.success(f"Unarchived '{selected_name}'.")
                        st.rerun()

    st.markdown("##### Streamlined Manual Transactions")
    with st.expander("Enable streamlined manual transactions and manage templates"):
        conn = get_connection()
        try:
            streamlined_enabled = bool(get_streamlined_manual_txns_enabled(conn))
        except Exception:
            streamlined_enabled = False
        finally:
            conn.close()
        with st.form("smt_toggle_form"):
            enabled_choice = st.checkbox("Enable Streamlined Manual Transactions", value=streamlined_enabled, key="smt_enabled_checkbox")
            save_toggle = st.form_submit_button("Save")
        if save_toggle:
            conn = get_connection()
            try:
                set_streamlined_manual_txns_enabled(conn, bool(enabled_choice))
            finally:
                conn.close()
            runtime.state.append_ui_event("streamlined_toggle_save")
            st.success("Streamlined Manual Transactions setting saved.")
            st.rerun()
        conn = get_connection()
        try:
            active_accounts = get_accounts(conn, include_inactive=False)
        finally:
            conn.close()
        if active_accounts:
            gl_label_to_gl = {f"{account.gl_number} – {account.name}": account.gl_number for account in active_accounts}
            with st.form("smt_create_template_form"):
                tmpl_name = st.text_input("Template name (unique)", value="", key="smt_new_template_name")
                base_gl_label = st.selectbox("Base GL", options=list(gl_label_to_gl.keys()), index=0, key="smt_new_template_base_gl")
                base_side = st.radio("Base GL side", options=["Credit", "Debit"], index=0, key="smt_new_template_base_side", horizontal=True)
                create_btn = st.form_submit_button("Save Template")
            if create_btn:
                conn = get_connection()
                try:
                    create_manual_txn_template(conn, name=(tmpl_name or "").strip(), base_gl_number=int(gl_label_to_gl[base_gl_label]), base_side="CREDIT" if base_side == "Credit" else "DEBIT")
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    runtime.state.append_ui_event("streamlined_template_create")
                    st.success("Template saved.")
                    st.rerun()
                finally:
                    conn.close()
        conn = get_connection()
        try:
            templates_all = get_manual_txn_templates(conn, include_inactive=True)
        except Exception:
            templates_all = []
        finally:
            conn.close()
        if templates_all:
            st.dataframe(pd.DataFrame([{"Name": template.name, "Base GL": int(template.base_gl_number), "Base Side": template.base_side, "Status": "Active" if template.is_active else "Archived"} for template in templates_all]), use_container_width=True, hide_index=True)
            label_to_t = {f"{template.name} ({'Active' if template.is_active else 'Archived'})": template for template in templates_all}
            selected_label = st.selectbox("Select template to modify", options=list(label_to_t.keys()), index=0, key="smt_template_modify_select")
            selected_t = label_to_t[selected_label]
            if selected_t.is_active:
                if st.button("Archive template", key="smt_archive_template_btn"):
                    conn = get_connection()
                    try:
                        set_manual_txn_template_active(conn, int(selected_t.id), False)
                    finally:
                        conn.close()
                    runtime.state.append_ui_event("streamlined_template_archive")
                    st.success("Template archived.")
                    st.rerun()
            else:
                if st.button("Unarchive template", key="smt_unarchive_template_btn"):
                    conn = get_connection()
                    try:
                        set_manual_txn_template_active(conn, int(selected_t.id), True)
                    finally:
                        conn.close()
                    runtime.state.append_ui_event("streamlined_template_unarchive")
                    st.success("Template unarchived.")
                    st.rerun()

    conn = get_connection()
    try:
        accounts = get_accounts(conn, include_inactive=False)
        current_mappings = {key: get_bs_category_gl_numbers(conn, key) for key in BS_CATEGORIES.keys()}
    finally:
        conn.close()
    if accounts:
        st.markdown("---")
        st.subheader("Financial Statement Settings")
        st.markdown("---")
        st.markdown("##### Simplified Balance Sheet Mappings")
        with st.expander("Add or update financial statement GL mappings"):
            gl_to_label = {account.gl_number: f"{account.gl_number} – {account.name}" for account in accounts}
            label_to_gl = {label: gl for gl, label in gl_to_label.items()}
            updated_selections = {}
            with st.form("bs_mappings_form"):
                for key, meta in BS_CATEGORIES.items():
                    existing_gls = current_mappings.get(key) or []
                    preselected_labels = [gl_to_label[gl] for gl in existing_gls if gl in gl_to_label]
                    updated_selections[key] = st.multiselect(f"{meta['label']}", options=list(gl_to_label.values()), default=preselected_labels, key=f"bs_map_{key}")
                save_clicked = st.form_submit_button("Save")
            if save_clicked:
                conn = get_connection()
                try:
                    for key, labels in updated_selections.items():
                        set_bs_category_gl_numbers(conn, key, [label_to_gl[label] for label in labels if label in label_to_gl])
                finally:
                    conn.close()
                runtime.state.append_ui_event("bs_mapping_save")
                st.success("Simplified balance sheet mappings updated.")
        st.markdown("##### Financials Start Date")
        with st.expander("Financials Start Date"):
            conn = get_connection()
            try:
                current_start = get_financials_start_date(conn)
            finally:
                conn.close()
            if current_start:
                st.info(f"Financials start date is currently set to: {current_start}")
            try:
                default_start_date = datetime.strptime(current_start, "%Y-%m-%d").date() if current_start else date.today()
            except Exception:
                default_start_date = date.today()
            new_start_date = st.date_input("Financials start date (must be month-end)", value=default_start_date, key="financials_start_date_input")
            if st.button("Save Financials Start Date", key="save_financials_start_date"):
                conn = get_connection()
                try:
                    set_financials_start_date(conn, new_start_date.isoformat())
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    runtime.state.append_ui_event("financials_start_date_save")
                    st.success("Financials start date saved.")
                    st.rerun()
                finally:
                    conn.close()


def render_journal_entries_tab(runtime) -> None:
    st.subheader("Journal Entries")
    st.markdown("Use this form to post manual journal entries. All amounts are in dollars; debits must equal credits.")
    st.selectbox("Number of lines", options=[2, 3, 4, 5, 6], index=0, key="je_num_lines")
    lock_col1, lock_col2 = st.columns([1, 2])
    with lock_col1:
        lock_dates = st.checkbox("Lock dates", key="je_lock_dates")
    with lock_col2:
        if lock_dates:
            default_locked = st.session_state.get("je_locked_date") or date.today()
            runtime.state.storage["je_locked_date"] = st.date_input("Locked date", value=default_locked, key="je_locked_date_input")
        else:
            runtime.state.storage.pop("je_locked_date", None)
    with st.form("journal_entry_form", clear_on_submit=True):
        col_dates = st.columns(3)
        default_je_date = st.session_state.get("je_locked_date") or date.today()
        with col_dates[0]:
            effective_date = st.date_input("Effective date", value=default_je_date)
        with col_dates[1]:
            post_date = st.date_input("Post date", value=default_je_date)
        with col_dates[2]:
            source = st.text_input("Source", value="MANUAL")
        description = st.text_input("Entry description", value=st.session_state.get("je_description", ""), key="je_description")
        new_desc = st.session_state.get("je_description", "")
        prev_desc = st.session_state.get("je_prev_description")
        if prev_desc is not None and new_desc != prev_desc:
            for key in list(st.session_state.keys()):
                if key.startswith("je_memo_") and st.session_state.get(key) == prev_desc:
                    st.session_state[key] = new_desc
        st.session_state["je_prev_description"] = new_desc

        conn = get_connection()
        try:
            accounts = get_accounts(conn, include_inactive=False)
        finally:
            conn.close()
        if not accounts:
            st.warning("No accounts available. Define accounts first in Chart of Accounts.")
            return
        account_options = {f"{account.gl_number} – {account.name}": account.id for account in accounts}
        option_labels = list(account_options.keys())
        line_inputs = []
        for i in range(int(st.session_state.get("je_num_lines", 2))):
            c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
            with c1:
                account_label = st.selectbox("Account", options=option_labels, key=f"je_account_{i}")
            with c2:
                amount_dollars = st.number_input("Amount", min_value=0.00, step=0.01, format="%.2f", key=f"je_amount_{i}")
            with c3:
                side = st.selectbox("Side", options=["Debit", "Credit"], key=f"je_side_{i}")
            with c4:
                memo = st.text_input("Memo (optional)", value=description, key=f"je_memo_{i}")
            line_inputs.append({"account_label": account_label, "amount_dollars": amount_dollars, "side": side, "memo": memo})
        conn = get_connection()
        try:
            last_closed = get_last_closed_period_end(conn)
        finally:
            conn.close()
        is_in_closed_period = last_closed is not None and effective_date.isoformat() <= last_closed
        override_pin_input = st.text_input("Override PIN (only required for closed periods)", type="password", key="manual_je_override_pin")
        submitted = st.form_submit_button("Post Journal Entry")
        if submitted:
            lines = [
                JournalEntryLine(
                    account_id=account_options[line["account_label"]],
                    amount_cents=int(round(line["amount_dollars"] * 100)),
                    is_debit=(line["side"] == "Debit"),
                    memo=line["memo"] or None,
                )
                for line in line_inputs
                if line["amount_dollars"] > 0 and account_options.get(line["account_label"]) is not None
            ]
            if not lines:
                st.error("No non-zero lines were provided. Please enter at least one amount.")
            else:
                conn = get_connection()
                try:
                    je_id = post_journal_entry_with_period_lock(
                        conn=conn,
                        effective_date=effective_date.isoformat(),
                        post_date=post_date.isoformat(),
                        description=description or "Manual entry",
                        source=source or "MANUAL",
                        lines=lines,
                        override_pin=(override_pin_input if is_in_closed_period else None),
                    )
                except ValueError as exc:
                    st.error(f"Entry not posted: {exc}")
                else:
                    runtime.state.append_ui_event("manual_je_post")
                    st.success(f"Journal entry posted successfully (ID: {je_id}).")
                finally:
                    conn.close()

    conn = get_connection()
    try:
        streamlined_enabled = bool(get_streamlined_manual_txns_enabled(conn))
        templates_active = get_manual_txn_templates(conn, include_inactive=False) if streamlined_enabled else []
        last_closed = get_last_closed_period_end(conn)
    except Exception:
        streamlined_enabled = False
        templates_active = []
        last_closed = None
    finally:
        conn.close()
    if streamlined_enabled and templates_active:
        with st.expander("Streamlined Manual Transactions"):
            conn = get_connection()
            try:
                active_accounts = get_accounts(conn, include_inactive=False)
            finally:
                conn.close()
            if active_accounts:
                gl_to_acct = {account.gl_number: account for account in active_accounts}
                label_to_id = {f"{account.gl_number} – {account.name}": account.id for account in active_accounts}
                acct_labels = list(label_to_id.keys())
                tmpl_by_name = {template.name: template for template in templates_active}
                selected_name = st.selectbox("Template", options=sorted(tmpl_by_name.keys(), key=lambda value: value.lower()), index=0, key="smt_je_template_select")
                tmpl = tmpl_by_name[selected_name]
                base_gl = int(tmpl.base_gl_number)
                base_acct = gl_to_acct.get(base_gl)
                if base_acct is None:
                    st.error(f"Base GL {base_gl} is not an active account. Unarchive the base account or archive this template.")
                else:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        eff_date = st.date_input("Effective date", value=date.today(), key=f"smt_eff_{tmpl.id}")
                    with c2:
                        amt = st.number_input("Amount (USD)", min_value=0.00, step=0.01, format="%.2f", key=f"smt_amt_{tmpl.id}")
                    desc = st.text_input("Description", value="", key=f"smt_desc_{tmpl.id}")
                    offset_labels = [label for label in acct_labels if int(label.split(" – ", 1)[0].strip()) != base_gl]
                    offset_label = st.selectbox(f"Offset GL – {'Credit' if str(tmpl.base_side).lower() == 'debit' else 'Debit'}", options=offset_labels, index=0, key=f"smt_offset_{tmpl.id}") if offset_labels else None
                    is_closed = last_closed is not None and eff_date.isoformat() <= last_closed
                    if is_closed:
                        st.warning(f"Effective date {eff_date.isoformat()} is in a closed period (closed through {last_closed}). Streamlined posting is blocked for closed periods.")
                    if st.button("Post", key=f"smt_post_{tmpl.id}", disabled=is_closed):
                        if amt <= 0:
                            st.error("Amount must be greater than $0.00.")
                        elif not (desc or "").strip():
                            st.error("Description is required.")
                        elif not offset_label:
                            st.error("Offset GL is required.")
                        else:
                            amount_cents = int(round(float(amt) * 100))
                            base_is_debit = str(tmpl.base_side).upper() == "DEBIT"
                            lines = [
                                JournalEntryLine(account_id=int(base_acct.id), amount_cents=amount_cents, is_debit=base_is_debit, memo=desc.strip()),
                                JournalEntryLine(account_id=int(label_to_id[offset_label]), amount_cents=amount_cents, is_debit=(not base_is_debit), memo=desc.strip()),
                            ]
                            conn = get_connection()
                            try:
                                je_id = post_journal_entry_with_period_lock(
                                    conn=conn,
                                    effective_date=eff_date.isoformat(),
                                    post_date=date.today().isoformat(),
                                    description=desc.strip(),
                                    source="MANUAL_" + tmpl.name.replace(" ", "_"),
                                    lines=lines,
                                    override_pin=None,
                                )
                            except ValueError as exc:
                                st.error(f"Entry not posted: {exc}")
                            else:
                                runtime.state.append_ui_event("streamlined_je_post")
                                st.success(f"Posted streamlined entry (JE ID: {je_id}).")
                                st.rerun()
                            finally:
                                conn.close()

    st.markdown("#### Journal Entry History")
    filt_col1, filt_col2, filt_col3 = st.columns([1, 1.5, 1.5])
    with filt_col1:
        enable_date_filter = st.checkbox("Filter by date range", key="je_hist_date_filter", value=False)
    with filt_col2:
        start_date = st.date_input("Start date", value=date.today() - timedelta(days=30), key="je_hist_start_date") if enable_date_filter else None
    with filt_col3:
        end_date = st.date_input("End date", value=date.today(), key="je_hist_end_date") if enable_date_filter else None
    conn = get_connection()
    try:
        account_filter_options = ["All accounts"]
        gl_label_to_gl = {}
        filter_accounts = get_accounts(conn, include_inactive=False)
        for account in filter_accounts:
            label = f"{account.gl_number} – {account.name}"
            account_filter_options.append(label)
            gl_label_to_gl[label] = account.gl_number
        selected_filter_label = st.selectbox("Filter by GL account", options=account_filter_options, index=0, key="je_hist_gl_filter")
        selected_gl_for_filter = None if selected_filter_label == "All accounts" else gl_label_to_gl.get(selected_filter_label)
        sql = """
            SELECT
                je.id AS je_id,
                je.effective_date,
                je.post_date,
                je.description,
                je.source,
                jl.account_id,
                a.gl_number,
                a.name AS account_name,
                jl.is_debit,
                jl.amount_cents,
                jl.id AS line_id
            FROM journal_entries je
            JOIN journal_lines jl ON jl.journal_entry_id = je.id
            JOIN accounts a ON a.id = jl.account_id
        """
        conditions = []
        params = []
        if enable_date_filter:
            if start_date is not None:
                conditions.append("je.effective_date >= ?")
                params.append(start_date.isoformat())
            if end_date is not None:
                conditions.append("je.effective_date <= ?")
                params.append(end_date.isoformat())
        if selected_gl_for_filter is not None:
            conditions.append("a.gl_number = ?")
            params.append(selected_gl_for_filter)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY je.id DESC, jl.id"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    if rows:
        history_data = []
        for row in rows:
            amount_d = (row["amount_cents"] or 0) / 100.0
            history_data.append(
                {
                    "JE ID": row["je_id"],
                    "Line ID": row["line_id"],
                    "Effective Date": row["effective_date"],
                    "Post Date": row["post_date"],
                    "Description": row["description"],
                    "Source": row["source"],
                    "GL Number": row["gl_number"],
                    "Account Name": row["account_name"],
                    "Debit": format_money(amount_d if row["is_debit"] else 0.0),
                    "Credit": format_money(amount_d if not row["is_debit"] else 0.0),
                }
            )
        df_je_hist = pd.DataFrame(history_data)
        reverse_state = runtime.state.get_je_reverse_state()
        df_je_hist["Reverse"] = df_je_hist["JE ID"].map(reverse_state).fillna(False)
        with st.form("je_history_form"):
            edited_df = st.data_editor(
                df_je_hist,
                use_container_width=True,
                hide_index=True,
                column_config={"Reverse": st.column_config.CheckboxColumn("Reverse", help="Mark this journal entry to post an automatic reversing entry.", default=False)},
                key="je_history_editor",
            )
            post_updates = st.form_submit_button("Post Updates")
        new_reverse_state = {int(je_id): bool(group["Reverse"].any()) for je_id, group in edited_df.groupby("JE ID")} if not edited_df.empty else {}
        runtime.state.set_je_reverse_state(new_reverse_state)
        if post_updates:
            if not new_reverse_state:
                st.info("No journal entries selected for reversal.")
            else:
                conn = get_connection()
                try:
                    today_str = date.today().isoformat()
                    posted_count = 0
                    errors = []
                    for je_id, selected in new_reverse_state.items():
                        if not selected:
                            continue
                        try:
                            header = conn.execute("SELECT effective_date, description, source FROM journal_entries WHERE id = ?", (je_id,)).fetchone()
                            if header is None:
                                continue
                            line_rows = conn.execute("SELECT account_id, amount_cents, is_debit, memo FROM journal_lines WHERE journal_entry_id = ? ORDER BY id", (je_id,)).fetchall()
                            rev_lines = [JournalEntryLine(account_id=row["account_id"], amount_cents=row["amount_cents"], is_debit=not bool(row["is_debit"]), memo=row["memo"]) for row in line_rows]
                            insert_journal_entry(
                                conn=conn,
                                effective_date=header["effective_date"],
                                post_date=today_str,
                                description=f"Reversal of JE {je_id}: {header['description']}" if header["description"] else f"Reversal of JE {je_id}",
                                source=f"{header['source']}_REV" if header["source"] else "REVERSAL",
                                lines=rev_lines,
                            )
                            posted_count += 1
                        except Exception as exc:
                            errors.append(f"JE {je_id}: {exc}")
                    if posted_count > 0:
                        runtime.state.append_ui_event("je_reversal_post")
                        st.success(f"Posted {posted_count} reversing journal entries.")
                        runtime.state.set_je_reverse_state({})
                    if errors:
                        st.error("Some reversing entries could not be posted:\n" + "\n".join(f"- {message}" for message in errors))
                finally:
                    conn.close()
    else:
        st.info("No journal entries have been posted yet.")

    with st.expander("Bulk Journal Entry Upload"):
        template_file_path = Path(__file__).resolve().parents[1] / "Bulk_JE_Upload_Template.xlsx"
        if template_file_path.exists():
            with open(template_file_path, "rb") as file_handle:
                st.download_button(
                    label="Download template file",
                    data=file_handle.read(),
                    file_name="Bulk_JE_Upload_Template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="bulk_je_template_download",
                )
        bulk_file = st.file_uploader("Upload JE template (Excel)", type=["xlsx", "xls"], accept_multiple_files=False, key="je_bulk_upload_file")

        def stage_bulk_jes(uploaded) -> None:
            try:
                df_raw = pd.read_excel(uploaded)
            except Exception as exc:
                st.error(f"Could not read Excel file: {exc}")
                return
            df_raw.columns = [str(column).strip() for column in df_raw.columns]
            required_cols = ["JE #", "DR", "CR", "AMT", "Eff Date"]
            missing = [column for column in required_cols if column not in df_raw.columns]
            if missing:
                st.error(f"Missing required columns in template: {', '.join(missing)}")
                return
            has_memo = "Memo" in df_raw.columns
            has_desc = "Desc" in df_raw.columns
            today_str = date.today().isoformat()
            records = []
            errors = []
            for idx, row in df_raw.iterrows():
                try:
                    je_batch = row["JE #"]
                    if pd.isna(je_batch):
                        continue
                    je_batch_str = str(int(je_batch)) if not isinstance(je_batch, str) else je_batch.strip()
                    amt = row["AMT"]
                    if pd.isna(amt) or float(amt) <= 0:
                        continue
                    amount_c = int(round(float(amt) * 100))
                    dr_gl = row["DR"]
                    cr_gl = row["CR"]
                    dr_provided = not pd.isna(dr_gl)
                    cr_provided = not pd.isna(cr_gl)
                    if dr_provided == cr_provided:
                        errors.append(f"Row {idx + 2}: exactly one of DR or CR must be non-empty.")
                        continue
                    gl_number = int(dr_gl) if dr_provided else int(cr_gl)
                    is_debit = bool(dr_provided)
                    eff_dt = pd.to_datetime(row["Eff Date"]).date()
                    records.append(
                        {
                            "Include": True,
                            "Template JE #": je_batch_str,
                            "Eff Date": eff_dt.strftime("%Y-%m-%d"),
                            "Post Date": today_str,
                            "GL Number": gl_number,
                            "Side": "Debit" if is_debit else "Credit",
                            "Amount": format_money(float(amt)),
                            "Memo": "" if not has_memo or pd.isna(row["Memo"]) else str(row["Memo"]).strip(),
                            "Description": "" if not has_desc or pd.isna(row["Desc"]) else str(row["Desc"]).strip(),
                            "_je_batch": je_batch_str,
                            "_gl_number": gl_number,
                            "_is_debit": is_debit,
                            "_amount_cents": amount_c,
                            "_eff_date": eff_dt.isoformat(),
                        }
                    )
                except Exception as exc:
                    errors.append(f"Row {idx + 2}: {exc}")
            if errors:
                st.error("Some rows in the template could not be staged:\n" + "\n".join(f"- {message}" for message in errors))
            if not records:
                st.info("No valid journal entry lines were found in the uploaded template.")
                return
            df_stage = pd.DataFrame(records)
            runtime.state.set_bulk_je_stage_df(df_stage)
            runtime.register_dataset(DATASET_BULK_JE_STAGE, df_stage, kind="dataframe", description="Bulk JE staged lines", metadata={"rows": len(df_stage)})
            st.success(f"Staged {len(records)} journal entry lines from template.")

        if bulk_file is not None and st.button("Review template journal entries", key="je_bulk_review_btn"):
            stage_bulk_jes(bulk_file)

        df_stage = runtime.state.get_bulk_je_stage_df()
        if df_stage is not None:
            try:
                conn = get_connection()
                try:
                    acct_list = get_accounts(conn, include_inactive=False)
                finally:
                    conn.close()
                gl_to_name = {account.gl_number: account.name for account in acct_list}
                if "_gl_number" in df_stage.columns:
                    df_stage["GL Description"] = df_stage["_gl_number"].map(gl_to_name).fillna("")
            except Exception:
                if "GL Description" not in df_stage.columns:
                    df_stage["GL Description"] = ""
            runtime.state.set_bulk_je_stage_df(df_stage)
            visible_column_order = ["Include", "Template JE #", "Eff Date", "Post Date", "GL Number", "GL Description", "Side", "Amount", "Memo", "Description"]
            edited_df = st.data_editor(
                df_stage,
                use_container_width=True,
                hide_index=True,
                column_order=visible_column_order,
                column_config={
                    "Include": st.column_config.CheckboxColumn("Include", help="Uncheck to skip this line when posting.", default=True),
                    "Amount": st.column_config.TextColumn("Amount", disabled=True),
                    "GL Number": st.column_config.NumberColumn("GL Number", disabled=True),
                    "GL Description": st.column_config.TextColumn("GL Description", disabled=True),
                    "Eff Date": st.column_config.TextColumn("Eff Date", disabled=True),
                    "Post Date": st.column_config.TextColumn("Post Date", disabled=True),
                    "Side": st.column_config.TextColumn("Side", disabled=True),
                },
                key="je_bulk_stage_editor",
            )
            runtime.register_dataset(DATASET_BULK_JE_STAGE, edited_df, kind="dataframe", description="Editable bulk JE stage", metadata={"rows": len(edited_df)})
            included_rows = edited_df[edited_df["Include"] == True] if not edited_df.empty else pd.DataFrame()
            total_debit_cents = int(included_rows.loc[included_rows["_is_debit"] == True, "_amount_cents"].sum()) if not included_rows.empty else 0
            total_credit_cents = int(included_rows.loc[included_rows["_is_debit"] == False, "_amount_cents"].sum()) if not included_rows.empty else 0
            st.markdown(f"**Diff (DR - CR) for included lines:** {format_money((total_debit_cents - total_credit_cents) / 100.0)}")
            if st.button("Post template JEs", key="post_template_jes_btn"):
                rows_to_post = edited_df[edited_df["Include"] == True]
                if rows_to_post.empty:
                    st.error("No lines selected to post. Check the Include column.")
                else:
                    conn = get_connection()
                    try:
                        acct_list = get_accounts(conn, include_inactive=False)
                        gl_to_acct_id = {account.gl_number: account.id for account in acct_list}
                        today_str = date.today().isoformat()
                        posted_count = 0
                        errors = []
                        for batch_id, group in rows_to_post.groupby("_je_batch"):
                            try:
                                lines = []
                                first_row = group.iloc[0]
                                for _, row in group.iterrows():
                                    acct_id = gl_to_acct_id.get(int(row["_gl_number"]))
                                    if acct_id is None:
                                        raise ValueError(f"GL {int(row['_gl_number'])} not found in Chart of Accounts.")
                                    lines.append(
                                        JournalEntryLine(
                                            account_id=acct_id,
                                            amount_cents=int(row["_amount_cents"]),
                                            is_debit=bool(row["_is_debit"]),
                                            memo=row.get("Memo") or first_row.get("Description") or f"Bulk JE {batch_id}",
                                        )
                                    )
                                insert_journal_entry(
                                    conn=conn,
                                    effective_date=first_row["_eff_date"],
                                    post_date=today_str,
                                    description=first_row.get("Description") or f"Bulk JE {batch_id}",
                                    source="BULK_JE",
                                    lines=lines,
                                )
                                posted_count += 1
                            except Exception as exc:
                                errors.append(f"Template JE {batch_id}: {exc}")
                        if posted_count > 0:
                            runtime.state.append_ui_event("bulk_je_post")
                            st.success(f"Posted {posted_count} journal entries from template.")
                            runtime.state.clear_bulk_je_stage_df()
                            runtime.datasets.clear(DATASET_BULK_JE_STAGE)
                        if errors:
                            st.error("Some template journal entries could not be posted:\n" + "\n".join(f"- {message}" for message in errors))
                    finally:
                        conn.close()

    with st.expander("Period Close"):
        today = date.today()
        default_period_end = date(today.year, today.month, 1) - timedelta(days=1)
        prev_period_end = runtime.state.get_period_close_prev_end()
        period_end = st.date_input("Close period ending", value=prev_period_end or default_period_end, key="close_period_end")
        if prev_period_end is None or prev_period_end != period_end:
            runtime.state.set_period_close_prev_end(period_end)
            runtime.state.clear_period_close_ready()
        preview_close = st.button("Preview closing entry", key="preview_close_btn")
        preview = None
        period_end_str = period_end.isoformat()
        period_start_str = date(period_end.year, period_end.month, 1).isoformat()
        if preview_close:
            conn = get_connection()
            try:
                re_gl = get_retained_earnings_gl_number(conn)
                if re_gl is None:
                    st.error("Retained earnings account is not configured. Please set it in the Preferences tab before closing a period.")
                    runtime.state.clear_period_close_ready()
                elif has_period_close_entry(conn, period_end_str):
                    st.error(f"A period-closing entry already exists for {period_end_str}. Reverse it before posting another closing entry for this period.")
                    runtime.state.clear_period_close_ready()
                else:
                    preview = compute_period_close_preview(conn=conn, period_start=period_start_str, period_end=period_end_str, retained_earnings_gl_number=re_gl)
                    if not preview.lines:
                        st.info(f"No income or expense activity found between {period_start_str} and {period_end_str}; no closing entry is necessary.")
                        runtime.state.clear_period_close_ready()
                    else:
                        df_preview = pd.DataFrame(
                            [
                                {
                                    "GL Number": line.account.gl_number,
                                    "Name": line.account.name,
                                    "Type": line.account.type.value,
                                    "Debit": format_money(line.amount_cents / 100.0 if line.is_debit else 0.0),
                                    "Credit": format_money(line.amount_cents / 100.0 if not line.is_debit else 0.0),
                                }
                                for line in preview.lines
                            ]
                        )
                        st.table(df_preview)
                        st.markdown(
                            f"- Total Income (period): {format_money(preview.total_income_cents / 100.0)}  \n"
                            f"- Total Expenses (period): {format_money(preview.total_expense_cents / 100.0)}  \n"
                            f"- Net Income (period): {format_money(preview.net_income_cents / 100.0)}"
                        )
                        runtime.state.set_period_close_ready(True, period_end_str)
            except ValueError as exc:
                st.error(str(exc))
                runtime.state.clear_period_close_ready()
            finally:
                conn.close()
        if runtime.state.get_period_close_ready() and runtime.state.get_period_close_ready_date() == period_end_str:
            if st.button("Post closing entry", key="post_close_btn"):
                conn = get_connection()
                try:
                    re_gl = get_retained_earnings_gl_number(conn)
                    if re_gl is None:
                        st.error("Retained earnings account is not configured. Please set it in the Preferences tab before closing a period.")
                        runtime.state.clear_period_close_ready()
                    elif has_period_close_entry(conn, period_end_str):
                        st.error(f"A period-closing entry already exists for {period_end_str}. Reverse it before posting another closing entry for this period.")
                        runtime.state.clear_period_close_ready()
                    else:
                        je_id = post_period_close_entry(
                            conn=conn,
                            period_start=period_start_str,
                            period_end=period_end_str,
                            retained_earnings_gl_number=re_gl,
                            description=f"Closing entry for {period_end_str}",
                        )
                        runtime.state.append_ui_event("period_close_post")
                        st.success(f"Period closing entry posted successfully (JE ID: {je_id}).")
                        runtime.state.clear_period_close_ready()
                except ValueError as exc:
                    st.error(str(exc))
                finally:
                    conn.close()


def render_personal_gl(*, hosted: bool = False) -> None:
    init_db()
    if not hosted:
        st.set_page_config(page_title="Personal GL", layout="wide")
    st.title("Personal General Ledger")

    runtime = build_runtime(st.session_state)
    runtime.register_dataset(
        DATASET_SQLITE_DB,
        {"path": str(runtime.db_path.resolve())},
        kind="sqlite",
        description="Active SQLite database for the Personal GL app",
    )

    stored_password = load_stored_password()
    render_lock_gate(runtime, stored_password)
    render_global_controls(runtime, stored_password)

    tabs = st.tabs(TAB_LABELS)
    with tabs[0]:
        render_financial_statements_tab(runtime)
    with tabs[1]:
        render_account_history_tab(runtime)
    with tabs[2]:
        render_chart_of_accounts_tab(runtime)
    with tabs[3]:
        render_journal_entries_tab(runtime)
    with tabs[4]:
        render_upload_transactions_tab(runtime)
    with tabs[5]:
        render_preferences_tab(runtime)
    with tabs[6]:
        render_warnings_tab(runtime)
    with tabs[7]:
        render_notes_tab(runtime)
    with tabs[8]:
        render_search_tab(runtime)
    with tabs[9]:
        render_logs_tab(runtime)
    with tabs[10]:
        render_documentation_tab(runtime)


def run() -> None:
    render_personal_gl(hosted=False)
