import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import datetime
import pytz
from collections import Counter
import io

# --------------------------------------------------
# 1. 기본 설정 & 스타일
# --------------------------------------------------
st.set_page_config(
    page_title="HEIMDALL GT Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 3rem !important; padding-bottom: 3rem !important; }
        .main-header { font-size: 2.2rem; font-weight: 800; color: #2C3E50; margin-bottom: 0.5rem; }
        .sub-header { font-size: 1.0rem; color: #7F8C8D; margin-bottom: 2rem; border-bottom: 2px solid #ECF0F1; padding-bottom: 10px; }
        .kpi-card {
            background-color: white; border-radius: 8px; padding: 15px;
            border: 1px solid #E0E0E0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 10px; transition: box-shadow 0.3s;
        }
        .kpi-card:hover { box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        .kpi-title { font-size: 0.85rem; color:#666; font-weight:700; margin-bottom:5px;}
        .kpi-main { font-size:1.4rem; font-weight:800; color:#2C3E50;}
        .kpi-unit { font-size:0.9rem; color:#95A5A6; margin-left:4px;}
        .kpi-sub { font-size:0.8rem; margin-top:2px;}
        .pos { color:#27AE60; } .neg { color:#C0392B; }
        .insight-box { background-color:#F7F9F9; border-left:4px solid #34495E; padding:15px; border-radius:4px; font-size:0.95rem; margin-bottom: 15px; }
        .heimdall-box { background-color:#EBF5FB; border-left: 5px solid #2980B9; padding: 15px 20px; border-radius: 5px; margin-bottom: 20px; }
        .heimdall-header { font-size: 1.1rem; font-weight: 800; color: #2980B9; margin-bottom: 8px; }
        .heimdall-content { font-size: 0.95rem; line-height: 1.6; color: #2C3E50; }
    </style>
    """, unsafe_allow_html=True
)

# --------------------------------------------------
# 2. 데이터 로딩 및 유틸리티 (웹 업로드 방식으로 변경됨)
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(uploaded_file):
    """
    [변경됨] 로컬 경로 대신 업로드된 파일 객체를 받아서 처리합니다.
    기존의 전처리 로직은 100% 유지됩니다.
    """
    if uploaded_file is None:
        return None
    
    try:
        # 업로드된 파일을 pandas로 읽음
        df = pd.read_parquet(uploaded_file)
        
        # --- 기존 전처리 로직 시작 ---
        for c in ["Year", "WeekNum"]: df[c] = df[c].astype(int)
        numeric_cols = ["Sales", "Qty", "Store_Count", "Distribution", "Store_Universe", "ASP"]
        for c in numeric_cols:
            if c not in df.columns: df[c] = 0.0
            else: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        if "Code" not in df.columns: df["Code"] = "UNKNOWN"
        else: df["Code"] = df["Code"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

        meta_cols = ["Maker_Main", "Maker_Detail", "Line", "Brand", "Product_Name"]
        for c in meta_cols:
            if c not in df.columns: df[c] = "Unknown"
            df[c] = df[c].fillna("Unknown").astype(str).str.strip()

        df["WeekIndex"] = df["Year"] * 100 + df["WeekNum"]
        
        launch_info = df[df["Sales"] > 0].groupby("Code")["WeekIndex"].min().rename("Launch_WeekIdx")
        df = df.merge(launch_info, on="Code", how="left")
        # --- 기존 전처리 로직 끝 ---
        
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {e}")
        return None

def pct_change(now, base):
    return (now - base) / base * 100.0 if base != 0 else 0.0

def expand_week_series(df_year, max_week, value_col):
    weeks = pd.DataFrame({"WeekNum": range(1, max_week + 1)})
    if df_year.empty: weeks[value_col] = np.nan
    else:
        merged = weeks.merge(df_year[["WeekNum", value_col]], on="WeekNum", how="left")
        weeks[value_col] = pd.to_numeric(merged[value_col], errors="coerce")
    return weeks["WeekNum"], weeks[value_col].astype(float)

def get_current_time_info():
    KST = pytz.timezone('Asia/Seoul')
    now = datetime.datetime.now(KST)
    iso_year, iso_week, _ = now.isocalendar()
    curr_week_str = f"{now.strftime('%Y년 %m월')} {iso_week}주차 (ISO)"
    return now, iso_year, iso_week, curr_week_str

def get_cutoff_week_idx(curr_y, curr_w, weeks_back):
    try:
        d = datetime.date.fromisocalendar(curr_y, curr_w, 1)
        t = d - datetime.timedelta(weeks=weeks_back)
        return t.isocalendar().year * 100 + t.isocalendar().week
    except:
        return (curr_y * 100 + curr_w) - weeks_back

def show_download_button(df, filename, label="💾 데이터 다운로드 (CSV)"):
    if df is not None and not df.empty:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=label, 
            data=csv, 
            file_name=f"{filename}_{datetime.datetime.now().strftime('%Y%m%d')}.csv", 
            mime="text/csv", 
            key=f"dl_{filename}_{datetime.datetime.now().timestamp()}"
        )

# --------------------------------------------------
# [Logic] Helper Functions (원본 유지)
# --------------------------------------------------
def determine_price_status(current_asp, history_asps):
    if not history_asps or len(history_asps) < 4:
        return "New"

    rounded_history = [round(p, -1) for p in history_asps]
    if not rounded_history: return "Error"
        
    count = Counter(rounded_history)
    mode_price = count.most_common(1)[0][0]
    
    if mode_price == 0: return "Error"

    ratio = current_asp / mode_price

    if 0.96 <= ratio <= 1.04:
        return "Regular"
    elif ratio < 0.96:
        if ratio < 0.85: return "Deep Promo"
        else: return "Promo"
    else:
        return "Price Hike"

def calculate_growth_drivers(df_curr, df_prev):
    cols = ["Code", "Product_Name", "Brand", "Line", "Sales", "Qty", "Distribution"]
    cols_curr = [c for c in cols if c in df_curr.columns]
    cols_prev = [c for c in cols if c in df_prev.columns]

    if len(df_curr) > 0:
        curr = df_curr[cols_curr].groupby("Code").agg(
            {"Product_Name":"first", "Brand":"first", "Line":"first", "Sales":"sum", "Qty":"sum", "Distribution":"mean"}
        ).reset_index()
    else: curr = pd.DataFrame(columns=cols)

    if len(df_prev) > 0:
        prev = df_prev[cols_prev].groupby("Code").agg(
            {"Sales":"sum", "Qty":"sum", "Distribution":"mean"}
        ).reset_index()
    else: prev = pd.DataFrame(columns=["Code", "Sales", "Qty", "Distribution"])

    merged = pd.merge(curr, prev, on="Code", how="outer", suffixes=("", "_LY")).fillna(0)
    merged["ASP"] = merged.apply(lambda x: x["Sales"]/x["Qty"] if x["Qty"]>0 else 0, axis=1)
    merged["ASP_LY"] = merged.apply(lambda x: x["Sales_LY"]/x["Qty_LY"] if x["Qty_LY"]>0 else 0, axis=1)

    def decomp(row):
        if row["Sales_LY"] == 0 and row["Sales"] > 0: return row["Sales"], 0, 0, 0
        elif row["Sales_LY"] > 0 and row["Sales"] == 0: return 0, -row["Sales_LY"], 0, 0
        else:
            price_effect = (row["ASP"] - row["ASP_LY"]) * row["Qty_LY"]
            volume_effect = (row["Qty"] - row["Qty_LY"]) * row["ASP_LY"]
            return 0, 0, price_effect, volume_effect

    merged[["New", "End", "Price", "Vol"]] = merged.apply(decomp, axis=1, result_type="expand")
    return merged

def compute_wf6(s_c, s_p, d_c, d_p, drivers):
    new_v = drivers["New"].sum(); end_v = drivers["End"].sum()
    price_v = drivers["Price"].sum(); vol_v = drivers["Vol"].sum()
    dist_effect = s_p * ((d_c - d_p) / d_p) if d_p > 0 else 0.0
    total_gap = s_c - s_p
    etc_v = total_gap - (new_v + end_v + price_v + vol_v + dist_effect)
    return {"New":new_v, "End":end_v, "Price":price_v, "Vol":vol_v, "Dist":dist_effect, "Etc":etc_v, "TotalGap":total_gap}

def kpi_dict_from_scope(df_scope, df_market, curr_year, prev_year, week, include_ms=False):
    curr = df_scope[(df_scope["Year"]==curr_year) & (df_scope["WeekNum"]==week)]
    prev = df_scope[(df_scope["Year"]==prev_year) & (df_scope["WeekNum"]==week)]
    
    if week > 1: wow_year = curr_year; wow_week = week - 1
    else: wow_year = prev_year; wow_week = df_scope[df_scope["Year"]==prev_year]["WeekNum"].max() if not df_scope[df_scope["Year"]==prev_year].empty else 52
    wow = df_scope[(df_scope["Year"]==wow_year) & (df_scope["WeekNum"]==wow_week)]

    def agg(d):
        if d.empty: return 0.0, 0.0, 0.0, 0.0
        s = float(d["Sales"].sum()); q = float(d["Qty"].sum())
        dist = float(d["Distribution"].mean()) if "Distribution" in d.columns else 0.0
        asp = s/q if q>0 else 0.0
        return s, q, dist, asp

    s_c, q_c, d_c, a_c = agg(curr); s_p, q_p, d_p, a_p = agg(prev); s_w, q_w, d_w, a_w = agg(wow)
    kpi = {
        "Sales": s_c, "Sales_Prev": s_p, "Diff_YoY": s_c-s_p, "Gr_YoY": pct_change(s_c, s_p),
        "Qty": q_c, "Qty_Prev": q_p, "Qty_Diff_YoY": q_c-q_p, "Qty_Gr_YoY": pct_change(q_c, q_p),
        "Dist": d_c, "Dist_Prev": d_p, "Dist_Gap_YoY": d_c-d_p,
        "ASP": a_c, "ASP_Prev": a_p, "ASP_Diff_YoY": a_c-a_p, "ASP_Gr_YoY": pct_change(a_c, a_p),
        "Gr_WoW": pct_change(s_c, s_w), "Diff_WoW": s_c-s_w,
        "Qty_Gr_WoW": pct_change(q_c, q_w), "Qty_Diff_WoW": q_c-q_w,
        "Dist_Gap_WoW": d_c-d_w,
        "ASP_Gr_WoW": pct_change(a_c, a_w), "ASP_Diff_WoW": a_c-a_w
    }

    if include_ms:
        m_curr = df_market[(df_market["Year"]==curr_year) & (df_market["WeekNum"]==week)]
        m_prev = df_market[(df_market["Year"]==prev_year) & (df_market["WeekNum"]==week)]
        m_wow = df_market[(df_market["Year"]==wow_year) & (df_market["WeekNum"]==wow_week)]
        s_m_c = float(m_curr["Sales"].sum()) if not m_curr.empty else 0.0
        s_m_p = float(m_prev["Sales"].sum()) if not m_prev.empty else 0.0
        s_m_w = float(m_wow["Sales"].sum()) if not m_wow.empty else 0.0
        ms_c = s_c/s_m_c*100 if s_m_c>0 else 0.0
        ms_p = s_p/s_m_p*100 if s_m_p>0 else 0.0
        ms_w = s_w/s_m_w*100 if s_m_w>0 else 0.0
        kpi.update({
            "MS": ms_c, "MS_Prev": ms_p, "MS_Gap_YoY": ms_c - ms_p, "MS_Gr_YoY": pct_change(ms_c, ms_p),
            "MS_Gap_WoW": ms_c - ms_w, "MS_Gr_WoW": pct_change(ms_c, ms_w)
        })

    drivers = calculate_growth_drivers(curr, prev)
    kpi["WF6"] = compute_wf6(s_c, s_p, d_c, d_p, drivers)
    return kpi, drivers

def build_wf_kpi(df_scope, df_base, curr_year, prev_year, week, mode_label):
    base_week = df_base[(df_base["Year"] == curr_year) & (df_base["WeekNum"] == week)]
    sel_month = int(base_week["Month"].iloc[0]) if not base_week.empty and "Month" in base_week.columns else 1

    if mode_label.startswith("주간"):
        curr_period = df_scope[(df_scope["Year"] == curr_year) & (df_scope["WeekNum"] == week)]
        prev_period = df_scope[(df_scope["Year"] == prev_year) & (df_scope["WeekNum"] == week)]
        period_desc = f"W{week}"
    elif mode_label.startswith("월누계"):
        curr_period = df_scope[(df_scope["Year"] == curr_year) & (df_scope["Month"] == sel_month)]
        prev_period = df_scope[(df_scope["Year"] == prev_year) & (df_scope["Month"] == sel_month)]
        period_desc = f"{sel_month}월 누계"
    elif mode_label.startswith("분기누계"):
        q = (sel_month - 1) // 3 + 1; q_start = (q - 1) * 3 + 1; q_end = q * 3
        curr_period = df_scope[(df_scope["Year"] == curr_year) & (df_scope["Month"] >= q_start) & (df_scope["Month"] <= q_end)]
        prev_period = df_scope[(df_scope["Year"] == prev_year) & (df_scope["Month"] >= q_start) & (df_scope["Month"] <= q_end)]
        period_desc = f"Q{q} 누계"
    else:
        curr_period = df_scope[(df_scope["Year"] == curr_year) & (df_scope["WeekNum"] <= week)]
        prev_period = df_scope[(df_scope["Year"] == prev_year) & (df_scope["WeekNum"] <= week)]
        period_desc = f"YTD ~W{week}"

    s_c, q_c, d_c, a_c = 0, 0, 0, 0
    if not curr_period.empty:
        s_c = curr_period["Sales"].sum(); q_c = curr_period["Qty"].sum()
        d_c = curr_period["Distribution"].mean() if "Distribution" in curr_period.columns else 0.0
        a_c = s_c / q_c if q_c > 0 else 0.0
    
    s_p, q_p, d_p, a_p = 0, 0, 0, 0
    if not prev_period.empty:
        s_p = prev_period["Sales"].sum(); q_p = prev_period["Qty"].sum()
        d_p = prev_period["Distribution"].mean() if "Distribution" in prev_period.columns else 0.0
        a_p = s_p / q_p if q_p > 0 else 0.0

    drivers = calculate_growth_drivers(curr_period, prev_period)
    wf6 = compute_wf6(s_c, s_p, d_c, d_p, drivers)
    
    kpi = {
        "Sales": s_c, "Sales_Prev": s_p, "Diff_YoY": s_c - s_p, "Gr_YoY": pct_change(s_c, s_p),
        "Qty": q_c, "Qty_Prev": q_p, "Qty_Diff_YoY": q_c - q_p, "Qty_Gr_YoY": pct_change(q_c, q_p),
        "Dist": d_c, "Dist_Prev": d_p, "Dist_Gap_YoY": d_c - d_p,
        "ASP": a_c, "ASP_Prev": a_p, "ASP_Diff_YoY": a_c - a_p, "ASP_Gr_YoY": pct_change(a_c, a_p),
        "WF6": wf6,
    }
    return kpi, drivers, period_desc

def line_level_table(df_scope, df_market, curr_year, prev_year, week, current_universe):
    def agg_by(df, y, w):
        d = df[(df["Year"] == y) & (df["WeekNum"] == w)]
        if d.empty: return pd.DataFrame(columns=["Line", "Sales", "Qty", "Dist", "ASP", "Sales_per_Store", "Qty_per_Store"])
        g = d.groupby("Line").agg({"Sales": "sum", "Qty": "sum", "Distribution": "mean"}).reset_index()
        g["ASP"] = g.apply(lambda x: x["Sales"] / x["Qty"] if x["Qty"] > 0 else 0, axis=1)
        
        if current_universe > 0:
            g["Est_Store_Count"] = g["Distribution"] / 100.0 * current_universe
            g["Sales_per_Store"] = g.apply(lambda x: x["Sales"] / x["Est_Store_Count"] if x["Est_Store_Count"] > 1 else 0, axis=1)
            g["Qty_per_Store"] = g.apply(lambda x: x["Qty"] / x["Est_Store_Count"] if x["Est_Store_Count"] > 1 else 0, axis=1)
        else:
            g["Sales_per_Store"] = 0.0; g["Qty_per_Store"] = 0.0
        g = g.rename(columns={"Distribution": "Dist"})
        return g

    cur_s = agg_by(df_scope, curr_year, week)
    prev_s = agg_by(df_scope, prev_year, week)
    base = cur_s.merge(prev_s[["Line", "Sales", "Dist", "ASP"]], on="Line", how="left", suffixes=("", "_LY"))
    base = base.fillna(0)
    base["Sales_YoY_%"] = base.apply(lambda x: pct_change(x["Sales"], x["Sales_LY"]), axis=1)
    base["Dist_YoY_%p"] = base["Dist"] - base["Dist_LY"]
    base["ASP_YoY_%"] = base.apply(lambda x: pct_change(x["ASP"], x["ASP_LY"]), axis=1)
    return base

# --------------------------------------------------
# UI Components (원본 유지)
# --------------------------------------------------
def kpi_card(container, title, main_val, unit, yoy_pct, yoy_diff_text, wow_pct=None, wow_diff_text=None, extra_line=None, tooltip=""):
    yoy_cls = "pos" if yoy_pct > 0 else ("neg" if yoy_pct < 0 else "")
    wow_cls = "pos" if (wow_pct is not None and wow_pct > 0) else ("neg" if (wow_pct is not None and wow_pct < 0) else "")
    wow_display = wow_pct if wow_pct is not None else 0.0
    wow_text = wow_diff_text if wow_diff_text else "-"
    wow_html = f'<div class="kpi-sub {wow_cls}">WoW {wow_display:+.1f}% ({wow_text})</div>'
    with container:
        st.markdown(
            f"""
            <div class="kpi-card" title="{tooltip}">
              <div class="kpi-title">{title} <span style="color:#999; font-size:0.8em;">ℹ️</span></div>
              <div class="kpi-main">{main_val}<span class="kpi-unit">{unit}</span></div>
              <div class="kpi-sub {yoy_cls}">YoY {yoy_pct:+.1f}% ({yoy_diff_text})</div>
              {wow_html}
              {f'<div class="kpi-sub" style="color:#555; font-size:0.75rem; margin-top:4px;">{extra_line}</div>' if extra_line else ''}
            </div>
            """, unsafe_allow_html=True
        )

def insight_from_wf6(scope_name, kpi, include_ms=False, mkt_kpi=None):
    wf = kpi["WF6"]; txt = []
    txt.append(f"{scope_name} 매출은 전년 대비 {kpi['Diff_YoY']/1e8:+.1f}억 ({kpi['Gr_YoY']:+.1f}%) 변동했습니다.")
    txt.append(f"요인별로는 신규 {wf['New']/1e8:+.1f}억, 중단 {wf['End']/1e8:+.1f}억, 단가 {wf['Price']/1e8:+.1f}억, 물량 {wf['Vol']/1e8:+.1f}억, 취급율 {wf['Dist']/1e8:+.1f}억, 기타 {wf['Etc']/1e8:+.1f}억 수준입니다.")
    if include_ms and mkt_kpi is not None:
        gap_yoy = kpi["Gr_YoY"] - mkt_kpi["Gr_YoY"]
        txt.append(f"시장 성장률 대비 갭은 YoY {gap_yoy:+.1f}%p 입니다.")
    return " ".join(txt)

def plot_waterfall_from_kpi(kpi, drivers_df, title, prev_year, curr_year):
    unit = 1e8; wf = kpi["WF6"]
    base = kpi["Sales_Prev"] / unit; new = wf["New"] / unit; end = wf["End"] / unit
    price = wf["Price"] / unit; vol = wf["Vol"] / unit; dist = wf["Dist"] / unit; etc = wf["Etc"] / unit
    final = kpi["Sales"] / unit
    prev_sales = kpi["Sales_Prev"]
    
    x_labels = [str(prev_year), "중단", "단가", "물량", "신규", "취급율", "기타", str(curr_year)]
    y_vals = [base, end, price, vol, new, dist, etc, final]
    measures = ["absolute"] + ["relative"] * 6 + ["total"]
    
    def get_top3_str(factor_col, direction="desc"):
        if drivers_df.empty: return ""
        df_sorted = drivers_df.sort_values(factor_col, ascending=(direction=="asc"))
        top3 = df_sorted.head(3)
        if top3[factor_col].abs().sum() == 0: return ""
        lines = [f"<b>[Top Contributors]</b>"]
        for _, row in top3.iterrows():
            val = row[factor_col] / 1e8
            if abs(val) > 0.01: lines.append(f"- {row['Product_Name']}: {val:+.1f}억")
        return "<br>".join(lines)
    
    hover_texts = []
    for i, lbl in enumerate(x_labels):
        val = y_vals[i]; top3 = ""; pct_str = "-"
        if prev_sales > 0:
            if i == 0: pct_str = "(기준)"
            elif i == len(y_vals) - 1:
                gr = (val * unit - prev_sales) / prev_sales * 100
                pct_str = f"YoY {gr:+.1f}%"
            else:
                contrib = (val * unit / prev_sales) * 100
                pct_str = f"기여도 {contrib:+.1f}%p"
        if lbl == "신규": top3 = get_top3_str("New", "desc")
        elif lbl == "중단": top3 = get_top3_str("End", "asc")
        elif lbl == "단가": top3 = get_top3_str("Price", "desc" if val >= 0 else "asc")
        elif lbl == "물량": top3 = get_top3_str("Vol", "desc" if val >= 0 else "asc")
        
        logic_desc = ""
        if lbl == "취급율": logic_desc = "<br>ℹ️ <b>취급율 효과:</b> 점포 수 변화로 인한 순수 매출 증감분"
        hover_texts.append(f"<b>{lbl}</b><br>값: {val:+.1f}억<br>{pct_str}<br><br>{top3}{logic_desc}")

    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=x_labels, y=y_vals,
        text=[f"{v:+.1f}" for v in y_vals], textposition="outside",
        connector={"line": {"color": "#999"}},
        decreasing={"marker": {"color": "#C62828"}}, increasing={"marker": {"color": "#2E7D32"}},
        totals={"marker": {"color": "#2C3E50"}},
        hovertext=hover_texts, hovertemplate="%{hovertext}<extra></extra>"
    ))
    if len(y_vals) > 0:
        ymax = max(y_vals); ymin = min(0, min(y_vals)); dy = ymax - ymin
        pad = dy * 0.25 if dy > 0 else 1.0
    fig.update_yaxes(range=[ymin - pad * 0.1, ymax + pad], automargin=True)
    fig.update_layout(height=300, title=title, xaxis=dict(type="category"), yaxis_title="증감액 (억)", margin=dict(t=90, b=20, l=10, r=10), showlegend=False)
    return fig

def plot_ms_chart(t_my_prev, t_my_curr, prev_year, latest_year):
    if t_my_prev.empty and t_my_curr.empty:
        fig = go.Figure(); fig.add_annotation(text="M/S 데이터가 없습니다.", showarrow=False); return fig
    max_week = max(int(t_my_prev["WeekNum"].max()) if not t_my_prev.empty else 0,
                   int(t_my_curr["WeekNum"].max()) if not t_my_curr.empty else 0, 1)
    weeks = pd.DataFrame({"WeekNum": range(1, max_week + 1)})
    if not t_my_prev.empty:
        weeks = weeks.merge(t_my_prev[["WeekNum", "MS"]].rename(columns={"MS": "MS_prev"}), on="WeekNum", how="left")
    else: weeks["MS_prev"] = np.nan
    if not t_my_curr.empty:
        weeks = weeks.merge(t_my_curr[["WeekNum", "MS"]].rename(columns={"MS": "MS_curr"}), on="WeekNum", how="left")
    else: weeks["MS_curr"] = np.nan
    weeks["Gap"] = weeks["MS_curr"] - weeks["MS_prev"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks["WeekNum"], y=weeks["MS_prev"], name=f"{prev_year} M/S", line=dict(color="#B0B0B0", width=2, dash="dot"), connectgaps=False))
    fig.add_trace(go.Scatter(x=weeks["WeekNum"], y=weeks["MS_curr"], name=f"{latest_year} M/S", line=dict(color="#2980B9", width=3), connectgaps=False,
                             customdata=weeks[["MS_prev", "Gap"]].values, hovertemplate="Week %{x}<br>Curr: %{y:.1f}%<br>Prev: %{customdata[0]:.1f}%<br>Gap: %{customdata[1]:+.1f}p<extra></extra>"))
    fig.update_layout(height=240, title="M/S 추이 및 Gap (%, Week)", hovermode="x unified", margin=dict(t=80, b=20, l=10, r=10))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig

def format_line_table_for_display(tbl: pd.DataFrame, include_ms: bool) -> pd.DataFrame:
    df_disp = tbl.copy()
    for col in df_disp.columns:
        if col.startswith("Sales"): df_disp[col] = (df_disp[col] / 1_000_000).round(1)
        if col.startswith("Qty"): df_disp[col] = (df_disp[col] / 1_000).round(1)
        if "ASP" in col: df_disp[col] = df_disp[col].round(0).astype(int)
        if "%" in col or "Dist" in col or "MS" in col:
            if df_disp[col].dtype != "O": df_disp[col] = df_disp[col].round(1)
    return df_disp

def wf_table_from_kpi(kpi):
    wf = kpi["WF6"]; prev = kpi["Sales_Prev"]
    rows = []
    for name, key in [("신규", "New"), ("중단", "End"), ("단가", "Price"), ("물량", "Vol"), ("취급율", "Dist"), ("기타", "Etc")]:
        val = wf[key]; contrib = (val / prev * 100) if prev > 0 else 0.0
        rows.append({"요인": name, "증감액(억)": val / 1e8, "전년 대비 기여율(%p)": contrib})
    total_gap = kpi["Sales"] - kpi["Sales_Prev"]
    rows.append({"요인": "합계", "증감액(억)": total_gap / 1e8, "전년 대비 기여율(%p)": (total_gap / prev * 100) if prev > 0 else 0.0})
    df = pd.DataFrame(rows)
    df["증감액(억)"] = df["증감액(억)"].round(1); df["전년 대비 기여율(%p)"] = df["전년 대비 기여율(%p)"].round(1)
    return df

def heimdall_opinion_card(title, content):
    st.markdown(
        f"""
        <div class="heimdall-box">
            <div class="heimdall-header">
                🛡️ HEIMDALL Strategy Opinion
            </div>
            <div class="heimdall-content">{content}</div>
        </div>
        """, unsafe_allow_html=True
    )

def generate_comprehensive_line_report(name, kpi, avg_sales, avg_dist, period_text):
    s = kpi["Sales"]; d = kpi["Dist"]; ms = kpi.get("MS", 0)
    gr = kpi["Gr_YoY"]; dg = kpi["Dist_Gap_YoY"]
    s_status = "상위권" if s >= avg_sales * 1.2 else ("하위권" if s < avg_sales * 0.8 else "평균 수준")
    gr_msg = f"전년 대비 **{gr:+.1f}%** 성장하며" if gr > 0 else f"전년 대비 **{gr:+.1f}%** 역신장하며"
    summary = []
    summary.append(f"**[{period_text}]** 기준, **{name}** 라인은 매출 **{s_status}**, 취급율 **{d:.1f}%**를 기록했습니다.")
    summary.append(f"{gr_msg} 시장 점유율(M/S) **{ms:.1f}%**를 차지하고 있습니다.")
    if gr > 5:
        if dg > 2: summary.append(f"📈 **[성장 요인]** 취급율이 전년 대비 +{dg:.1f}%p 확대되면서 물리적 커버리지가 늘어난 것이 주효했습니다.")
        else: summary.append(f"🌟 **[성장 요인]** 취급율 확대(Gap {dg:+.1f}%p)보다는 점당 회전율 개선이 성장을 견인했습니다.")
    elif gr < -5:
        if dg < -2: summary.append(f"📉 **[하락 원인]** 취급점이 전년 대비 {dg:.1f}%p 축소되며 매출 자연 감소가 발생했습니다.")
        else: summary.append(f"⚠️ **[하락 원인]** 매대는 유지되었으나 점당 효율이 떨어지고 있습니다.")
    return " ".join(summary)

# --------------------------------------------------
# 메인 App 실행 (사이드바 업로드 기능 추가)
# --------------------------------------------------

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown('<div class="main-header">HEIMDALL<br>GT Market<br>POS Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Strategic Intelligence Suite v9.0</div>', unsafe_allow_html=True)
    
    # [NEW] 파일 업로더 추가
    st.header("📂 Data Input")
    uploaded_file = st.file_uploader("Upload Parquet File", type=["parquet"])
    
    if uploaded_file is None:
        st.info("데이터 파일을 업로드해주세요.")
        st.stop()
        
    # 데이터 로드 (업로드된 파일 사용)
    df = load_data(uploaded_file)
    if df is None: st.stop()

    # [FIX] 타임존 및 날짜 표기 수정
    now, iso_year, iso_week, curr_week_str = get_current_time_info()
    st.caption(f"기준일: {curr_week_str}")

    menu_options = ["요약 대시보드", "세부 지표", "전략 브리핑(Line)", "⚖️ 가격 & 커버리지 전략", "📉 가격 시뮬레이터 (Pro)", "🏳️ 화이트 스페이스(Map)", "🌳 SKU 구조(Treemap)", "🚀 신제품 분석(New)", "🔎 제품 추적(Tracker)", "📚 로직 사전"]
    selected_view = st.radio("분석 뷰 (Menu)", menu_options)
    st.markdown("---")

    st.subheader("1. 기간 및 대상 설정")
    years = sorted(df["Year"].unique()); latest_year = int(years[-1])
    prev_year = int(years[-2]) if len(years)>=2 else latest_year-1
    last_week = int(df[df["Year"]==latest_year]["WeekNum"].max())
    sel_week = st.slider("분석 주차", 1, last_week, last_week)
    makers = sorted(df["Maker_Main"].unique())
    my_makers = st.multiselect("자사(제조사) 선택", makers, default=[makers[0]])
    
    st.subheader("2. 필터 설정")
    target_lines = st.multiselect("Line 범위", ["(전체)"]+sorted(df["Line"].unique()), default=["(전체)"])
    if "(전체)" in target_lines or not target_lines: temp_df = df.copy()
    else: temp_df = df[df["Line"].isin(target_lines)].copy()
    avail_brands = sorted(temp_df["Brand"].unique())
    target_brands = st.multiselect("브랜드 선택", ["(전체)"] + avail_brands, default=["(전체)"])
    avail_details = sorted(temp_df["Maker_Detail"].unique())
    target_details = st.multiselect("세부 제조사(Maker Detail) 선택", ["(전체)"] + avail_details, default=["(전체)"])

    st.markdown("---")
    st.subheader("3. 뷰 옵션")
    trend_basis = "매출"
    if selected_view == "요약 대시보드": trend_basis = st.radio("트렌드 기준", ["매출", "수량"], index=0, horizontal=True)
    
    default_univ = int(temp_df["Store_Universe"].max()) if "Store_Universe" in temp_df.columns and temp_df["Store_Universe"].max() > 0 else 50000
    current_universe = st.number_input("전체 점포 수 (Universe)", min_value=1, value=default_univ, step=100, help="취급율 역산에 사용되는 전체 모집단 점포 수")

    wf_mode = "주간 YoY"
    if selected_view in ["요약 대시보드", "세부 지표"]: wf_mode = st.radio("워터폴 기준", ["주간 YoY", "월누계", "분기누계", "연간누계"], horizontal=False)

    if "(전체)" in target_lines or not target_lines: df_step1 = df.copy()
    else: df_step1 = df[df["Line"].isin(target_lines)].copy()
    if "(전체)" in target_brands or not target_brands: df_step2 = df_step1.copy()
    else: df_step2 = df_step1[df_step1["Brand"].isin(target_brands)].copy()
    if "(전체)" in target_details or not target_details: df_mkt = df_step2.copy()
    else: df_mkt = df_step2[df_step2["Maker_Detail"].isin(target_details)].copy()
    
    df_my = df_mkt[df_mkt["Maker_Main"].isin(my_makers)].copy()
    if df_mkt.empty: st.warning("선택된 조건의 시장 데이터가 없습니다."); st.stop()

# ---------------- MAIN CONTENT RENDER (원본 유지) ----------------

trend_mkt = df_mkt.groupby(["Year", "WeekNum"]).agg({"Sales": "sum", "Qty": "sum"}).reset_index()
if not df_my.empty:
    trend_my = df_my.groupby(["Year", "WeekNum"]).agg({"Sales": "sum", "Qty": "sum"}).reset_index()
    trend_my = trend_my.merge(trend_mkt[["Year", "WeekNum", "Sales"]].rename(columns={"Sales": "Sales_Mkt"}), on=["Year", "WeekNum"], how="left")
    trend_my["MS"] = trend_my.apply(lambda x: x["Sales"] / x["Sales_Mkt"] * 100 if x["Sales_Mkt"] > 0 else 0.0, axis=1)
    t_my_curr = trend_my[trend_my["Year"] == latest_year].sort_values("WeekNum")
    t_my_prev = trend_my[trend_my["Year"] == prev_year].sort_values("WeekNum")
    kpi_my, drv_my = kpi_dict_from_scope(df_my, df_mkt, latest_year, prev_year, sel_week, include_ms=True)
    tbl_my = line_level_table(df_my, df_mkt, latest_year, prev_year, sel_week, current_universe)
    tbl_mkt_temp = line_level_table(df_mkt, df_mkt, latest_year, prev_year, sel_week, current_universe)
    tbl_my = tbl_my.merge(tbl_mkt_temp[["Line", "Sales"]].rename(columns={"Sales": "Mkt_Sales"}), on="Line", how="left")
    tbl_my["MS"] = tbl_my.apply(lambda x: x["Sales"] / x["Mkt_Sales"] * 100 if x["Mkt_Sales"] > 0 else 0, axis=1)
    kpi_my_wf, drv_my_wf, period_desc_my = build_wf_kpi(df_my, df_mkt, latest_year, prev_year, sel_week, wf_mode)
else:
    t_my_curr, t_my_prev = pd.DataFrame(), pd.DataFrame()
    kpi_my, tbl_my, kpi_my_wf, drv_my_wf, period_desc_my = {}, pd.DataFrame(), {}, pd.DataFrame(), ""

t_mkt_curr = trend_mkt[trend_mkt["Year"] == latest_year].sort_values("WeekNum")
t_mkt_prev = trend_mkt[trend_mkt["Year"] == prev_year].sort_values("WeekNum")
kpi_mkt, _ = kpi_dict_from_scope(df_mkt, df_mkt, latest_year, prev_year, sel_week, include_ms=False)
tbl_mkt = line_level_table(df_mkt, df_mkt, latest_year, prev_year, sel_week, current_universe)
kpi_mkt_wf, _, period_desc_mkt = build_wf_kpi(df_mkt, df_mkt, latest_year, prev_year, sel_week, wf_mode)

max_week_axis = max(int(t_mkt_prev["WeekNum"].max()) if not t_mkt_prev.empty else 0, int(t_mkt_curr["WeekNum"].max()) if not t_mkt_curr.empty else 0, 1)

if selected_view == "요약 대시보드":
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("전체 시장 (Market) 동주 전년비")
        st.markdown(f"""<div class="insight-box"><div class="insight-title">[Market Diagnosis]</div><div>{insight_from_wf6("시장", kpi_mkt)}</div></div>""", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        kpi_card(c1, "시장 매출", f"{kpi_mkt['Sales']/1e8:.1f}", "억", kpi_mkt["Gr_YoY"], f"{kpi_mkt['Diff_YoY']/1e8:+.1f}억", kpi_mkt["Gr_WoW"], f"{kpi_mkt['Diff_WoW']/1e8:+.1f}억", tooltip="매출 = 단가(ASP) × 수량(Qty)")
        kpi_card(c2, "시장 수량", f"{kpi_mkt['Qty']/1e3:,.0f}", "천개", kpi_mkt["Qty_Gr_YoY"], f"{kpi_mkt['Qty_Diff_YoY']/1e3:+.0f}천", kpi_mkt["Qty_Gr_WoW"], f"{kpi_mkt['Qty_Diff_WoW']/1e3:+.0f}천", tooltip="수량 = 팔린 제품 낱개 총합")
        kpi_card(c3, "시장 취급율", f"{kpi_mkt['Dist']:.1f}", "%", kpi_mkt["Dist_Gap_YoY"], "YoY Gap", kpi_mkt["Dist_Gap_WoW"], "WoW Gap", tooltip="취급율 = 제품이 팔린 점포 수 / 전체 점포 수 × 100")
        kpi_card(c4, "시장 단가", f"{kpi_mkt['ASP']:,.0f}", "원", kpi_mkt["ASP_Gr_YoY"], f"{kpi_mkt['ASP_Diff_YoY']:+,.0f}원", kpi_mkt["ASP_Gr_WoW"], f"{kpi_mkt['ASP_Diff_WoW']:+,.0f}원", tooltip="평균단가(ASP) = 총매출 / 총수량")
        
        metric_col = "Sales" if trend_basis == "매출" else "Qty"
        div = 1e8 if trend_basis == "매출" else 1e3
        y_title = "매출(억)" if trend_basis == "매출" else "수량(천개)"
        
        x_prev, y_prev = expand_week_series(t_mkt_prev, max_week_axis, metric_col)
        x_curr, y_curr = expand_week_series(t_mkt_curr, max_week_axis, metric_col)
        
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=x_prev, y=y_prev/div, name=str(prev_year), line=dict(color="#BDC3C7", width=2, dash="dot"), connectgaps=False))
        fig_m.add_trace(go.Scatter(x=x_curr, y=y_curr/div, name=str(latest_year), line=dict(color="#2C3E50", width=3), connectgaps=False))
        fig_m.update_layout(height=340, title=f"총시장 주간 {trend_basis} 추이", hovermode="x unified", margin=dict(t=80, b=20, l=10, r=10))
        fig_m.update_xaxes(showgrid=False); fig_m.update_yaxes(title=y_title, showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_m, use_container_width=True)
        show_download_button(t_mkt_curr, "market_trend_data")

        st.markdown("##### 라인별 시장 지표")
        if not tbl_mkt.empty: 
            st.dataframe(format_line_table_for_display(tbl_mkt[["Line","Sales","Sales_YoY_%","Dist","Dist_YoY_%p","ASP","ASP_YoY_%"]], False).sort_values("Sales", ascending=False), use_container_width=True, hide_index=True)
            show_download_button(tbl_mkt, "market_line_table")

    with right_col:
        st.subheader(f"제조사 ({', '.join(my_makers)}) 동주 전년비")
        if df_my.empty: st.warning("데이터가 없습니다.")
        else:
            st.markdown(f"""<div class="insight-box"><div class="insight-title">[Company Diagnosis]</div><div>{insight_from_wf6("제조사", kpi_my, True, kpi_mkt)}</div></div>""", unsafe_allow_html=True)
            c1r, c2r, c3r, c4r, c5r = st.columns(5)
            kpi_card(c1r, "제조사 매출", f"{kpi_my['Sales']/1e8:.1f}", "억", kpi_my["Gr_YoY"], f"{kpi_my['Diff_YoY']/1e8:+.1f}억", kpi_my["Gr_WoW"], f"{kpi_my['Diff_WoW']/1e8:+.1f}억", tooltip="자사 매출")
            kpi_card(c2r, "제조사 수량", f"{kpi_my['Qty']/1e3:,.0f}", "천개", kpi_my["Qty_Gr_YoY"], f"{kpi_my['Qty_Diff_YoY']/1e3:+.0f}천", kpi_my["Qty_Gr_WoW"], f"{kpi_my['Qty_Diff_WoW']/1e3:+.0f}천", tooltip="자사 수량")
            kpi_card(c3r, "제조사 취급율", f"{kpi_my['Dist']:.1f}", "%", kpi_my["Dist_Gap_YoY"], "YoY Gap", kpi_my["Dist_Gap_WoW"], "WoW Gap", tooltip="자사 평균 취급율")
            kpi_card(c4r, "제조사 단가", f"{kpi_my['ASP']:,.0f}", "원", kpi_my["ASP_Gr_YoY"], f"{kpi_my['ASP_Diff_YoY']:+,.0f}원", kpi_my["ASP_Gr_WoW"], f"{kpi_my['ASP_Diff_WoW']:+,.0f}원", tooltip="자사 평균 ASP")
            kpi_card(c5r, "M/S", f"{kpi_my['MS']:.1f}", "%", kpi_my["MS_Gap_YoY"], "YoY Gap", kpi_my["MS_Gr_WoW"], "WoW Gap", tooltip="시장 점유율")
            
            x_prev_my, y_prev_my = expand_week_series(t_my_prev, max_week_axis, metric_col)
            x_curr_my, y_curr_my = expand_week_series(t_my_curr, max_week_axis, metric_col)
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=x_prev_my, y=y_prev_my/div, name=f"{prev_year}", line=dict(color="#BDC3C7", width=2, dash="dot"), connectgaps=False))
            fig_s.add_trace(go.Scatter(x=x_curr_my, y=y_curr_my/div, name=f"{latest_year}", line=dict(color="#2980B9", width=3), connectgaps=False))
            fig_s.update_layout(height=340, title=f"제조사 주간 {trend_basis} 추이", hovermode="x unified", margin=dict(t=80, b=20, l=10, r=10))
            fig_s.update_xaxes(showgrid=False); fig_s.update_yaxes(title=y_title, showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig_s, use_container_width=True)
            show_download_button(t_my_curr, "company_trend_data")

            st.plotly_chart(plot_ms_chart(t_my_prev, t_my_curr, prev_year, latest_year), use_container_width=True)
            st.markdown("##### 라인별 제조사 지표")
            if not tbl_my.empty: 
                st.dataframe(format_line_table_for_display(tbl_my[["Line","Sales","Sales_YoY_%","Dist","Dist_YoY_%p","ASP","ASP_YoY_%","MS"]], True).sort_values("Sales", ascending=False), use_container_width=True, hide_index=True)
                show_download_button(tbl_my, "company_line_table")

    st.markdown("---")
    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown(f"##### 시장 6요인 워터폴 ({period_desc_mkt})")
        if kpi_mkt_wf and kpi_mkt_wf["Sales_Prev"] > 0: 
            st.plotly_chart(plot_waterfall_from_kpi(kpi_mkt_wf, drv_my_wf, f"시장 매출 증감 - {period_desc_mkt}", prev_year, latest_year), use_container_width=True)
            show_download_button(wf_table_from_kpi(kpi_mkt_wf), "market_waterfall")
    with wc2:
        st.markdown(f"##### 제조사 6요인 워터폴 ({period_desc_my})")
        if kpi_my_wf and kpi_my_wf["Sales_Prev"] > 0: 
            st.plotly_chart(plot_waterfall_from_kpi(kpi_my_wf, drv_my_wf, f"제조사 매출 증감 - {period_desc_my}", prev_year, latest_year), use_container_width=True)
            show_download_button(wf_table_from_kpi(kpi_my_wf), "company_waterfall")

elif selected_view == "세부 지표":
    st.subheader("KPI 요약 및 워터폴 데이터")
    c_kpi1, c_kpi2 = st.columns(2)
    with c_kpi1:
        st.markdown("##### 시장 워터폴 데이터")
        if kpi_mkt_wf: 
            df_wf_m = wf_table_from_kpi(kpi_mkt_wf)
            st.dataframe(df_wf_m, use_container_width=True, hide_index=True)
            show_download_button(df_wf_m, "market_wf_detail")
    with c_kpi2:
        st.markdown("##### 제조사 워터폴 데이터")
        if kpi_my_wf: 
            df_wf_c = wf_table_from_kpi(kpi_my_wf)
            st.dataframe(df_wf_c, use_container_width=True, hide_index=True)
            show_download_button(df_wf_c, "company_wf_detail")
    st.markdown("---")
    st.subheader("🏆 세부 SKU 동향 (Top/Bottom Performers)")
    if df_my.empty: st.warning("데이터가 없습니다.")
    else:
        if not drv_my_wf.empty:
            drv = drv_my_wf.copy()
            drv["Sales_Diff"] = drv["Sales"] - drv["Sales_LY"]
            drv["Gr_YoY"] = drv.apply(lambda x: pct_change(x["Sales"], x["Sales_LY"]), axis=1)
            drv_disp = drv[["Code","Product_Name","Brand","Line","Sales","Sales_LY","Sales_Diff","Gr_YoY","New","End"]].copy()
            for c in ["Sales","Sales_LY","Sales_Diff","New","End"]: drv_disp[c] = (drv_disp[c]/1e8).round(1)
            
            st.markdown("##### 🔥 급상승(Top Gainers) & ❄️ 급락(Top Losers) SKU")
            c_gain, c_loss = st.columns(2)
            with c_gain:
                st.caption("▲ 매출 증가 상위 10개 (단위: 억)")
                top_gain = drv_disp.sort_values("Sales_Diff", ascending=False).head(10)
                st.dataframe(
                    top_gain,
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Sales_Diff": st.column_config.ProgressColumn("증감액", format="%.1f억", min_value=0, max_value=float(drv_disp["Sales_Diff"].max())),
                        "Gr_YoY": st.column_config.NumberColumn("성장률", format="%.1f%%")
                    }
                )
                show_download_button(top_gain, "top_gainers")
            with c_loss:
                st.caption("▼ 매출 감소 상위 10개 (단위: 억)")
                top_loss = drv_disp.sort_values("Sales_Diff", ascending=True).head(10)
                st.dataframe(
                    top_loss,
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Sales_Diff": st.column_config.ProgressColumn("증감액", format="%.1f억", min_value=float(drv_disp["Sales_Diff"].min()), max_value=0),
                        "Gr_YoY": st.column_config.NumberColumn("성장률", format="%.1f%%")
                    }
                )
                show_download_button(top_loss, "top_losers")
            
            st.markdown("---")
            st.subheader("📦 전체 SKU 리스트 (All Products)")
            st.dataframe(drv_disp.sort_values("Sales", ascending=False), use_container_width=True, hide_index=True)
            show_download_button(drv_disp, "all_sku_performance")

elif selected_view == "전략 브리핑(Line)":
    st.markdown("### 📊 라인별 전략 브리핑 (Strategic Briefing)")
    period_opt_t3 = st.radio("분석 기간 기준", ["연간 누계(YTD)", "최근 4주(L4W)"], horizontal=True)
    
    if not tbl_my.empty:
        st.markdown("#### 1. 포트폴리오 스코어카드 (Portfolio Scorecard)")
        scorecard = tbl_my[["Line", "Sales", "Sales_YoY_%", "Dist", "MS", "Sales_per_Store"]].copy()
        scorecard["Sales(억)"] = (scorecard["Sales"]/1e8).round(1)
        
        st.dataframe(
            scorecard[["Line", "Sales(억)", "Sales_YoY_%", "Dist", "MS", "Sales_per_Store"]].sort_values("Sales(억)", ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                "Sales_YoY_%": st.column_config.NumberColumn("YoY (%)", format="%.1f%%"),
                "Dist": st.column_config.ProgressColumn("취급율 (%)", format="%.1f%%", min_value=0, max_value=100),
                "MS": st.column_config.NumberColumn("M/S (%)", format="%.1f%%"),
                "Sales_per_Store": st.column_config.NumberColumn("점당효율 (원)", format="%d")
            }
        )
        show_download_button(scorecard, "portfolio_scorecard")
        st.markdown("---")
        
        st.subheader("🎯 주간 액션 플랜 (Market Radar)")
        st.info("💡 SKU별 가격 정책과 진단을 입력하여 보고서를 완성하세요. (수정 가능)")

        sel_line_radar = st.selectbox("진단할 Line 선택", sorted(tbl_my["Line"].unique()))
        df_radar = df_my[(df_my["Line"] == sel_line_radar) & (df_my["Year"] == latest_year) & (df_my["WeekNum"] == sel_week)].copy()
        
        df_radar_prev = df_my[(df_my["Line"] == sel_line_radar) & (df_my["Year"] == prev_year) & (df_my["WeekNum"] == sel_week)][["Code", "Sales", "Distribution"]].rename(columns={"Sales": "Sales_LY", "Distribution": "Dist_LY"})
        df_radar = df_radar.merge(df_radar_prev, on="Code", how="left").fillna(0)
        
        df_radar["Week_Growth_Pct"] = df_radar.apply(lambda x: pct_change(x["Sales"], x["Sales_LY"]), axis=1)
        df_radar["Dist_Growth_Pp"] = df_radar["Distribution"] - df_radar["Dist_LY"]
        
        LABEL_MAP = {"Regular": "정상가", "Promo": "행사", "Deep Promo": "초특가", "New": "신제품", "Price Hike": "인상", "Error": "확인필요"}
        
        df_radar["Price_Stat_Code"] = "Regular"
        df_radar["Price_Stat"] = df_radar["Price_Stat_Code"].map(LABEL_MAP)
        
        def auto_diagnose(row):
            if row["Week_Growth_Pct"] < -10: return "📉 경고: 매출 급락 (원인 파악 필요)"
            elif row["Week_Growth_Pct"] > 10: return "🚀 호조: 성장세 지속 (재고 점검)"
            else: return "관망 필요 (특이사항 없음)"
            
        df_radar["Diagnosis"] = df_radar.apply(auto_diagnose, axis=1)
        
        df_editor_input = df_radar[["Product_Name", "Brand", "Sales", "Week_Growth_Pct", "Price_Stat", "Dist_Growth_Pp", "Diagnosis"]].copy()
        df_editor_input.columns = ["제품명", "브랜드", "주간매출", "성장률(%)", "가격정책", "커버리지(∆%p)", "진단 및 액션 플랜"]
        
        column_config = {
            "제품명": st.column_config.TextColumn("제품명", width="medium"),
            "브랜드": st.column_config.TextColumn("브랜드", width="small"),
            "주간매출": st.column_config.NumberColumn("주간매출", format="₩%,d", width="small"),
            "성장률(%)": st.column_config.NumberColumn("성장률(%)", format="%.1f%%", width="small"),
            "가격정책": st.column_config.SelectboxColumn("가격정책", options=["정상가", "행사", "초특가", "신제품"], width="small"),
            "커버리지(∆%p)": st.column_config.NumberColumn("커버리지(∆%p)", format="%.1f", width="small"),
            "진단 및 액션 플랜": st.column_config.TextColumn("진단 및 액션 플랜 (Editable)", width="large"),
        }

        edited_radar = st.data_editor(
            df_editor_input,
            column_config=column_config,
            disabled=["제품명", "브랜드", "주간매출", "성장률(%)", "커버리지(∆%p)"], 
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key=f"radar_editor_{sel_line_radar}"
        )
        
        show_download_button(edited_radar, f"action_plan_{sel_line_radar}")
        
        risk_count = edited_radar["진단 및 액션 플랜"].str.contains("경고|하락", na=False).sum()
        if risk_count > 0:
            st.error(f"⚠️ 현재 {risk_count}개 제품이 리스크 관리 대상으로 식별되었습니다.")

elif selected_view == "⚖️ 가격 & 커버리지 전략":
    st.markdown("### ⚖️ Price & Coverage Strategy Mix")
    st.info("💡 **전략 가이드:** 가격 정책(수요)과 영업 확장(공급)의 상호작용을 통합 진단합니다.")

    c_l, c_b, c_s = st.columns(3)
    with c_l: target_line_mix = st.selectbox("1. Line 선택", sorted(df_my["Line"].unique()), key="mix_l")
    df_line = df_my[df_my["Line"] == target_line_mix]
    with c_b: target_brand_mix = st.selectbox("2. Brand 선택 (Optional)", ["(전체)"] + sorted(df_line["Brand"].unique()), key="mix_b")
    df_brand = df_line if target_brand_mix == "(전체)" else df_line[df_line["Brand"] == target_brand_mix]
    with c_s: target_sku_mix = st.selectbox("3. SKU 선택 (Optional)", ["(전체)"] + sorted(df_brand["Product_Name"].unique()), key="mix_s")
    
    if target_sku_mix != "(전체)": df_final = df_brand[df_brand["Product_Name"] == target_sku_mix]; title_suffix = f"SKU: {target_sku_mix}"
    elif target_brand_mix != "(전체)": df_final = df_brand; title_suffix = f"Brand: {target_brand_mix}"
    else: df_final = df_line; title_suffix = f"Line: {target_line_mix}"

    df_trend = df_final.groupby(["Year", "WeekNum"]).agg({
        "Sales": "sum", "Qty": "sum", "Store_Count": "max", "Distribution": "mean"
    }).reset_index()
    df_trend["WeekIndex"] = df_trend["Year"].astype(str) + "-W" + df_trend["WeekNum"].astype(str)
    df_trend["ASP"] = df_trend.apply(lambda x: x["Sales"]/x["Qty"] if x["Qty"]>0 else 0, axis=1)
    df_trend["Sales_per_Store"] = df_trend.apply(lambda x: x["Sales"]/x["Store_Count"] if x["Store_Count"]>0 else 0, axis=1)

    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("#### 💰 가격 반응성 (Price Sensitivity)")
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Bar(x=df_trend["WeekIndex"], y=df_trend["Qty"], name="판매량", marker_color="#90CAF9", opacity=0.6), secondary_y=False)
        fig1.add_trace(go.Scatter(x=df_trend["WeekIndex"], y=df_trend["ASP"], name="ASP(단가)", line=dict(color="#D32F2F", width=3)), secondary_y=True)
        fig1.update_layout(height=400, showlegend=True, legend=dict(orientation="h", y=1.1), margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig1, use_container_width=True)
        show_download_button(df_trend, "price_elasticity_trend")
        
        corr = df_trend["ASP"].corr(df_trend["Qty"])
        if corr < -0.5: msg1 = "🔴 **민감도 높음:** 가격 인상 시 판매량 감소가 뚜렷합니다."
        elif corr > 0.5: msg1 = "🟢 **프리미엄화:** 가격/판매량이 동반 상승 중입니다."
        else: msg1 = "⚪ **비탄력적:** 가격보다 외부 요인 영향이 큽니다."
        st.caption(f"📢 진단: {msg1}")

    with c_chart2:
        st.markdown("#### 🏗️ 영업 효율성 (Coverage Quality)")
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=df_trend["WeekIndex"], y=df_trend["Distribution"], name="취급율(%)", fill='tozeroy', line=dict(color="#A5D6A7"), marker=dict(opacity=0)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=df_trend["WeekIndex"], y=df_trend["Sales_per_Store"], name="점당 효율", line=dict(color="#2E7D32", width=3)), secondary_y=True)
        fig2.update_layout(height=400, showlegend=True, legend=dict(orientation="h", y=1.1), margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)
        show_download_button(df_trend, "coverage_efficiency_trend")

        if len(df_trend) > 4:
            slope = np.polyfit(np.arange(len(df_trend)), df_trend["Sales_per_Store"], 1)[0]
            if slope > 50: msg2 = "🚀 **질적 성장:** 점당 매출이 상승 추세입니다."
            elif slope < -50: msg2 = "⚠️ **효율 희석:** 점당 매출이 하락하고 있습니다."
            else: msg2 = "➡️ **안정적:** 효율이 일정 수준 유지되고 있습니다."
        else: msg2 = "데이터 부족"
        st.caption(f"📢 진단: {msg2}")

    st.markdown("---")
    st.subheader("🧩 종합 원인 분석 (Efficiency Impact Analysis)")
    
    df_scope_pq = df_final.copy()
    weeks_curr = sorted(df_scope_pq[df_scope_pq["Year"] == latest_year]["WeekNum"].unique())
    if not weeks_curr: 
        st.warning("금년 데이터가 없습니다.")
    else:
        max_w = sel_week
        df_cy = df_scope_pq[(df_scope_pq["Year"] == latest_year) & (df_scope_pq["WeekNum"] <= max_w)]
        df_py = df_scope_pq[(df_scope_pq["Year"] == prev_year) & (df_scope_pq["WeekNum"] <= max_w)]
        
        def agg_pq_mix(d):
            if d.empty: return pd.DataFrame()
            g = d.groupby("Product_Name").agg({"Sales":"sum", "Qty":"sum", "Store_Count":"max"}).reset_index()
            g["Velocity"] = g.apply(lambda x: x["Qty"]/x["Store_Count"] if x["Store_Count"]>0 else 0, axis=1)
            g["ASP"] = g.apply(lambda x: x["Sales"]/x["Qty"] if x["Qty"]>0 else 0, axis=1)
            g["Eff_Sales"] = g.apply(lambda x: x["Sales"]/x["Store_Count"] if x["Store_Count"]>0 else 0, axis=1)
            return g

        agg_cy = agg_pq_mix(df_cy); agg_py = agg_pq_mix(df_py)
        
        if not agg_cy.empty:
            merged_pq = pd.merge(agg_cy, agg_py, on="Product_Name", how="left", suffixes=("", "_LY")).fillna(0)
            merged_pq = merged_pq.sort_values("Sales", ascending=False).head(15) 
            merged_pq["Eff_Gap_%"] = merged_pq.apply(lambda x: pct_change(x["Eff_Sales"], x["Eff_Sales_LY"]), axis=1)

            fig_pq = go.Figure()
            for _, row in merged_pq.iterrows():
                if row["Sales_LY"] > 0:
                    fig_pq.add_trace(go.Scatter(x=[row["Velocity_LY"], row["Velocity"]], y=[row["ASP_LY"], row["ASP"]], mode="lines", line=dict(color="silver", width=1, dash="dot"), showlegend=False, hoverinfo="skip"))
            
            fig_pq.add_trace(go.Scatter(
                x=merged_pq["Velocity"], y=merged_pq["ASP"], mode="markers+text", text=merged_pq["Product_Name"], textposition="top center",
                marker=dict(size=merged_pq["Sales"], sizemode="area", sizeref=2.*max(merged_pq["Sales"])/(40.**2), color=merged_pq["Eff_Gap_%"], colorscale="RdBu", cmid=0, showscale=True, colorbar=dict(title="효율 성장(%)")),
                hovertemplate="<b>%{text}</b><br>효율성장: %{marker.color:.1f}%<br>회전율: %{x:.1f}<br>ASP: %{y:,.0f}<extra></extra>"
            ))
            fig_pq.update_layout(title=f"SKU별 효율성 매트릭스 (X:회전율, Y:단가, Color:효율성장)", xaxis_title="점당 회전율 (Velocity)", yaxis_title="평균 단가 (ASP)", height=500, showlegend=False)
            st.plotly_chart(fig_pq, use_container_width=True)
            show_download_button(merged_pq, "efficiency_matrix_data")
            
            if not merged_pq.empty:
                best = merged_pq.sort_values("Eff_Gap_%", ascending=False).iloc[0]
                worst = merged_pq.sort_values("Eff_Gap_%", ascending=True).iloc[0]
                
                insight_text = f"""
                **[HEIMDALL 종합 진단]**<br>
                데이터 분석 결과, 효율 개선을 주도한 제품은 **'{best['Product_Name']}'** (+{best['Eff_Gap_%']:.1f}%)이며, 
                가장 큰 효율 저하를 보인 제품은 **'{worst['Product_Name']}'** ({worst['Eff_Gap_%']:.1f}%)입니다.<br>
                위 매트릭스에서 **우상단(↗)**으로 이동하는 제품은 '가격/회전율 동반 성장'의 이상적 모델이며, 
                **좌하단(↙)**으로 이동하는 제품은 구조조정이 필요합니다.
                """
                heimdall_opinion_card("Strategic Implication", insight_text)

elif selected_view == "📉 가격 시뮬레이터 (Pro)":
    st.markdown("### 📉 Price Strategy Simulator (Pro)")
    st.info("💡 **가이드:** 가격 할인(Promo)과 인상(Hike)을 모두 시뮬레이션하며, **시장 전체(Total Market) 구조 변화**를 예측합니다.")

    c_sel1, c_sel2, c_sel3, c_sel4 = st.columns(4)
    with c_sel1: 
        sim_maker = st.selectbox("1. 제조사", sorted(df_mkt["Maker_Main"].unique()))
        df_s1 = df_mkt[df_mkt["Maker_Main"] == sim_maker]
    with c_sel2:
        sim_line = st.selectbox("2. 라인", ["(전체)"] + sorted(df_s1["Line"].unique()))
        if sim_line != "(전체)": df_s1 = df_s1[df_s1["Line"] == sim_line]
    with c_sel3:
        sim_brand = st.selectbox("3. 브랜드", ["(전체)"] + sorted(df_s1["Brand"].unique()))
        if sim_brand != "(전체)": df_s1 = df_s1[df_s1["Brand"] == sim_brand]
    with c_sel4:
        sim_sku = st.selectbox("4. 제품(SKU)", sorted(df_s1["Product_Name"].unique()))

    st.divider()
    
    c_ctrl, c_viz = st.columns([1, 2])
    
    sim_range = 24
    df_sim = df_mkt[df_mkt["Product_Name"] == sim_sku].copy()
    df_sim = df_sim.sort_values(["Year", "WeekNum"], ascending=False).head(sim_range)
    
    if len(df_sim) < 4:
        st.warning("데이터 부족으로 분석할 수 없습니다.")
        st.stop()

    df_sim["ASP"] = df_sim["Sales"] / df_sim["Qty"]
    df_sim = df_sim[df_sim["ASP"] > 0]
    
    df_sim["Ln_Sales"] = np.log(df_sim["Qty"])
    df_sim["Ln_Price"] = np.log(df_sim["ASP"])
    try:
        slope, intercept = np.polyfit(df_sim["Ln_Price"], df_sim["Ln_Sales"], 1)
        elasticity = slope
    except: 
        elasticity = 0
        slope, intercept = 0, 0

    base_asp = df_sim["ASP"].mean()
    base_qty_per_week = df_sim["Qty"].mean()
    base_dist = df_sim["Distribution"].mean()
    base_store_count = df_sim["Store_Count"].max()
    if base_store_count == 0: base_store_count = 1
    
    base_qty_per_store = base_qty_per_week / base_store_count
    base_sales_per_week = base_asp * base_qty_per_week
    
    with c_ctrl:
        st.markdown("#### 🎛️ Scenario Builder")
        
        price_change_pct = st.slider("💰 가격 조정률 (Price Change %)", -50, 50, 0, 1, format="%d%%")
        
        st.markdown("**🏗️ 유통 커버리지 가정 (Dist. Assumption)**")
        dist_change_pct = st.slider(
            "가격 변화 시 점포수 증감 예측 (%)", 
            -30, 30, 0, 1, 
            help="예: 가격 인상 시 -5% (퇴점), 행사 시 +10% (행사 매대 확대)"
        )
        
        st.divider()
        st.metric("📊 현재 가격 탄력성 ($E_d$)", f"{elasticity:.2f}",
                 delta="민감함 (Elastic)" if abs(elasticity) > 1 else "둔감함 (Inelastic)",
                 delta_color="inverse")
        st.caption("절대값이 1보다 크면 가격 변화에 물량이 크게 반응합니다.")

    with c_viz:
        new_asp = base_asp * (1 + price_change_pct/100)
        
        qty_change_pct = elasticity * (price_change_pct/100)
        new_qty_per_store = base_qty_per_store * (1 + qty_change_pct)
        
        new_store_count = base_store_count * (1 + dist_change_pct/100)
        
        new_total_sales = new_asp * new_qty_per_store * new_store_count
        
        gap_sales = new_total_sales - base_sales_per_week
        
        effect_price = (new_asp - base_asp) * base_qty_per_week
        vol_change_units = (base_qty_per_week * qty_change_pct)
        effect_vol = new_asp * vol_change_units
        sales_per_store_new = new_asp * new_qty_per_store
        store_diff = new_store_count - base_store_count
        effect_dist = sales_per_store_new * store_diff
        
        calc_check = base_sales_per_week + effect_price + effect_vol + effect_dist
        remainder = new_total_sales - calc_check
        effect_vol += remainder

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=df_sim["ASP"], y=df_sim["Qty"], mode='markers', name='History', 
                                       marker=dict(color='gray', opacity=0.5, size=8)))
        
        if slope != 0:
            min_p = min(df_sim["ASP"].min(), new_asp) * 0.9
            max_p = max(df_sim["ASP"].max(), new_asp) * 1.1
            x_range = np.linspace(min_p, max_p, 100)
            y_pred = np.exp(intercept) * (x_range ** slope)
            fig_curve.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Demand Curve', line=dict(color='blue', width=2)))

        fig_curve.add_trace(go.Scatter(x=[base_asp], y=[base_qty_per_week], mode='markers', name='AS-IS (Current)',
                                       marker=dict(color='green', size=15, symbol='star')))
        
        sim_qty_demand_only = base_qty_per_week * (1 + qty_change_pct)
        fig_curve.add_trace(go.Scatter(x=[new_asp], y=[sim_qty_demand_only], mode='markers', name='TO-BE (Projected)',
                                       marker=dict(color='red', size=15, symbol='star')))

        fig_curve.update_layout(title="📉 수요 곡선 및 시뮬레이션 위치 (Demand Curve)", xaxis_title="가격 (Price)", yaxis_title="판매수량 (Qty)", height=350)
        st.plotly_chart(fig_curve, use_container_width=True)

        st.markdown("#### 📋 상세 지표 변화 (Comparison Table)")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("총 매출 (Total Sales)", 
                      f"{new_total_sales/1e4:,.0f}만원", 
                      f"{gap_sales/1e4:+,.0f}만원 ({pct_change(new_total_sales, base_sales_per_week):+.1f}%)")
        with col_m2:
            st.metric("평균 단가 (ASP)", 
                      f"{new_asp:,.0f}원", 
                      f"{new_asp - base_asp:+,.0f}원 ({price_change_pct:+.1f}%)")
        with col_m3:
            total_qty_new = new_qty_per_store * new_store_count
            st.metric("총 판매량 (Total Qty)", 
                      f"{total_qty_new:,.0f}개", 
                      f"{total_qty_new - base_qty_per_week:+,.0f}개 ({pct_change(total_qty_new, base_qty_per_week):+.1f}%)")
        with col_m4:
            st.metric("점포 커버리지 (Coverage)", 
                      f"{new_store_count:,.0f}점", 
                      f"{store_diff:+,.0f}점 ({dist_change_pct:+.1f}%)")
            
        st.info(f"""
        ℹ️ **진단 요약:**
        가격을 **{price_change_pct}%** 조정할 경우, 탄력성($E_d={elasticity:.2f}$)에 의해 점당 판매량은 **{qty_change_pct*100:+.1f}%** 변동하며,
        여기에 점포 커버리지 변화(**{dist_change_pct}%**)를 반영하면 최종 시장 규모는 **{pct_change(new_total_sales, base_sales_per_week):+.1f}%** 변동할 것으로 예측됩니다.
        """)

elif selected_view == "🏳️ 화이트 스페이스(Map)":
    st.markdown("### 🏳️ White Space Analysis (Price-Volume Map)")
    st.info("💡 **가이드:** 현재 카테고리 내에서 '매출 볼륨이 집중된 가격대'와 '비어있는 기회 영역(White Space)'을 시각화합니다.")

    col_ws1, col_ws2 = st.columns([1, 1])
    with col_ws1: 
        target_ws_line = st.selectbox("분석할 Line 선택", sorted(df_mkt["Line"].unique()))
    with col_ws2:
        available_weeks = sorted(df_mkt[df_mkt["Year"] == latest_year]["WeekNum"].unique())
        if not available_weeks: available_weeks = [1, 52]
        ws_week_range = st.slider("분석 기간 설정 (주차)", min_value=min(available_weeks), max_value=max(available_weeks), value=(min(available_weeks), max(available_weeks)))

    df_ws_curr = df_mkt[(df_mkt["Line"] == target_ws_line) & (df_mkt["Year"] == latest_year) & 
                        (df_mkt["WeekNum"] >= ws_week_range[0]) & (df_mkt["WeekNum"] <= ws_week_range[1])].copy()
    
    df_ws_prev = df_mkt[(df_mkt["Line"] == target_ws_line) & (df_mkt["Year"] == prev_year) & 
                        (df_mkt["WeekNum"] >= ws_week_range[0]) & (df_mkt["WeekNum"] <= ws_week_range[1])].copy()

    if df_ws_curr.empty: st.error("선택한 기간에 데이터가 없습니다."); st.stop()
    
    bins = [0, 1200, 1700, 2500, 2900, 3300, 3500, 4000, 4500, 5500, 6500, 7500, 8500, 9500, 10500, float('inf')]
    labels = ["~1,200", "1,200~1,700", "1,700~2,500", "2,500~2,900", "2,900~3,300", 
              "3,300~3,500", "3,500~4,000", "4,000~4,500", "4,500~5,500", "5,500~6,500", 
              "6,500~7,500", "7,500~8,500", "8,500~9,500", "9,500~10,500", "10,500~"]
    
    def agg_sku_ws(d):
        g = d.groupby(["Maker_Main", "Brand", "Product_Name"]).agg({"Sales": "sum", "Qty": "sum", "Store_Count": "max"}).reset_index()
        g["ASP"] = g.apply(lambda x: x["Sales"]/x["Qty"] if x["Qty"]>0 else 0, axis=1)
        g["Price_Range"] = pd.cut(g["ASP"], bins=bins, labels=labels, right=False).astype(str)
        return g

    ws_curr_agg = agg_sku_ws(df_ws_curr)
    
    total_cat_sales = ws_curr_agg["Sales"].sum()
    ws_curr_agg["Share_Pct"] = ws_curr_agg["Sales"] / total_cat_sales * 100
    ws_curr_agg["Qty_per_Store"] = ws_curr_agg.apply(lambda x: x["Qty"]/x["Store_Count"] if x["Store_Count"]>0 else 0, axis=1)
    ws_curr_agg["Sales_per_Store"] = ws_curr_agg.apply(lambda x: x["Sales"]/x["Store_Count"] if x["Store_Count"]>0 else 0, axis=1)

    ws_curr_agg["Is_My_Maker"] = ws_curr_agg["Maker_Main"].apply(lambda x: "🟦 자사" if x in my_makers else "⬜ 경쟁사")

    ws_prev_agg = agg_sku_ws(df_ws_prev)[["Product_Name", "Sales"]].rename(columns={"Sales": "Sales_LY"})
    
    ws_final = pd.merge(ws_curr_agg, ws_prev_agg, on="Product_Name", how="left")
    ws_final["Sales_LY"] = ws_final["Sales_LY"].fillna(0)
    ws_final["YoY_Pct"] = ws_final.apply(lambda x: pct_change(x["Sales"], x["Sales_LY"]), axis=1)
    
    ws_final = ws_final[ws_final["Price_Range"] != "nan"]
    ws_final = ws_final[ws_final["Sales"] > 0]
    
    if not ws_final.empty:
        fig_ws = px.treemap(
            ws_final, 
            path=[px.Constant(target_ws_line), "Price_Range", "Is_My_Maker", "Brand", "Product_Name"], 
            values="Sales",
            color="Is_My_Maker",
            color_discrete_map={"🟦 자사": "#2E86C1", "⬜ 경쟁사": "#BDC3C7"}, 
            title=f"가격대별 매출 구조도 (Price-Volume Map): {target_ws_line} (W{ws_week_range[0]}~W{ws_week_range[1]})",
            custom_data=["ASP", "Qty", "Sales", "Qty_per_Store", "Sales_per_Store", "Share_Pct"]
        )
        fig_ws.update_traces(hovertemplate="<b>%{label}</b><br>--------------<br>단가: %{customdata[0]:,.0f}원<br>판매량: %{customdata[1]:,.0f}<br>매출액: %{customdata[2]:,.0f}원<br>회전량(Q/S): %{customdata[3]:.1f}<br>회전액(S/S): %{customdata[4]:,.0f}<br>비중: %{customdata[5]:.1f}%<extra></extra>")
        fig_ws.update_layout(height=700)
        st.plotly_chart(fig_ws, use_container_width=True)
        show_download_button(ws_final, "white_space_treemap_data")
    else:
        st.warning("조건에 맞는 데이터가 없어 차트를 표시할 수 없습니다.")
    
    st.markdown("#### 📋 세부 데이터 테이블")
    
    dist_map = df_ws_curr.groupby("Product_Name")["Distribution"].mean().reset_index()
    ws_table = pd.merge(ws_final, dist_map, on="Product_Name", how="left")
    
    ws_table_disp = ws_table[[
        "Price_Range", "Maker_Main", "Brand", "Product_Name", 
        "ASP", "Qty", "Sales", "YoY_Pct", "Distribution", "Qty_per_Store", "Sales_per_Store", "Share_Pct"
    ]].copy()
    
    ws_table_disp.columns = ["가격대", "제조사", "브랜드", "제품명", "평균단가", "총판매수량", "총판매금액", "YoY(%)", "취급율(%)", "점당회전량", "점당회전액", "매출비중(%)"]
    
    st.dataframe(
        ws_table_disp.sort_values(["가격대", "총판매금액"], ascending=[True, False]),
        use_container_width=True, hide_index=True,
        column_config={
            "평균단가": st.column_config.NumberColumn("단가", format="%d원"),
            "총판매수량": st.column_config.NumberColumn("수량", format="%d"),
            "총판매금액": st.column_config.NumberColumn("매출", format="%d"),
            "YoY(%)": st.column_config.NumberColumn("YoY", format="%.1f%%"),
            "취급율(%)": st.column_config.ProgressColumn("취급율", format="%.1f%%", min_value=0, max_value=100),
            "점당회전량": st.column_config.NumberColumn("회전량", format="%.1f"),
            "점당회전액": st.column_config.NumberColumn("회전액", format="%d"),
            "매출비중(%)": st.column_config.NumberColumn("비중", format="%.1f%%"),
        }
    )
    show_download_button(ws_table_disp, "white_space_detailed_table")

elif selected_view == "🌳 SKU 구조(Treemap)":
    st.markdown("### 🌳 SKU Contribution Map (성장성 vs 규모)")
    c_sel, c_opt = st.columns([2, 1])
    with c_sel: target_line_t6 = st.selectbox("분석할 라인을 선택하세요", sorted(tbl_my["Line"].unique()))
    with c_opt: period_opt = st.selectbox("집계 기간", ["해당 주차", "최근 4주", "연간 누계(YTD)"])
    
    if period_opt == "해당 주차":
        sku_curr = df_my[(df_my["Year"]==latest_year) & (df_my["WeekNum"]==sel_week) & (df_my["Line"]==target_line_t6)]
        sku_prev = df_my[(df_my["Year"]==prev_year) & (df_my["WeekNum"]==sel_week) & (df_my["Line"]==target_line_t6)]
    elif period_opt == "최근 4주":
        target_weeks = sorted(df_my[df_my["Year"]==latest_year]["WeekNum"].unique())[-4:]
        sku_curr = df_my[(df_my["Year"]==latest_year) & (df_my["WeekNum"].isin(target_weeks)) & (df_my["Line"]==target_line_t6)]
        sku_prev = df_my[(df_my["Year"]==prev_year) & (df_my["WeekNum"].isin(target_weeks)) & (df_my["Line"]==target_line_t6)]
    else:
        sku_curr = df_my[(df_my["Year"]==latest_year) & (df_my["WeekNum"]<=sel_week) & (df_my["Line"]==target_line_t6)]
        sku_prev = df_my[(df_my["Year"]==prev_year) & (df_my["WeekNum"]<=sel_week) & (df_my["Line"]==target_line_t6)]
    
    sku_agg_c = sku_curr.groupby(["Code", "Product_Name"]).agg({"Sales": "sum", "Qty": "sum", "Store_Count": "max"}).reset_index()
    sku_agg_p = sku_prev.groupby(["Code", "Product_Name"]).agg({"Sales": "sum"}).reset_index().rename(columns={"Sales": "Sales_LY"})
    sku_merged = pd.merge(sku_agg_c, sku_agg_p, on=["Code", "Product_Name"], how="left").fillna(0)
    sku_merged["Qty_per_Store"] = sku_merged.apply(lambda x: x["Qty"]/x["Store_Count"] if x["Store_Count"]>0 else 0, axis=1)
    sku_merged["ASP"] = sku_merged.apply(lambda x: x["Sales"]/x["Qty"] if x["Qty"]>0 else 0, axis=1)
    def calc_gr_color(row):
        if row["Sales_LY"] == 0: return 999 
        return (row["Sales"] - row["Sales_LY"]) / row["Sales_LY"] * 100
    sku_merged["Gr_YoY"] = sku_merged.apply(calc_gr_color, axis=1)
    sku_merged["Color_Val"] = sku_merged["Gr_YoY"].clip(-50, 50)
    sku_merged.loc[sku_merged["Gr_YoY"] == 999, "Color_Val"] = 60 
    sku_merged = sku_merged[sku_merged["Sales"] > 0]
    
    fig = px.treemap(
        sku_merged, path=[px.Constant(target_line_t6), "Product_Name"], values="Sales", color="Color_Val",
        color_continuous_scale="RdBu", color_continuous_midpoint=0,
        custom_data=["Sales", "Qty", "Qty_per_Store", "ASP", "Gr_YoY"],
        title=f"{target_line_t6} SKU별 매출 기여도 ({period_opt})"
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br><br>매출: %{customdata[0]:,.0f}원<br>수량: %{customdata[1]:,.0f}개<br>점당회전: %{customdata[2]:.1f}개<br>단가: %{customdata[3]:,.0f}원<br>전년비: %{customdata[4]:.1f}%<extra></extra>")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    show_download_button(sku_merged, "sku_treemap_data")
    
    pos_share = (sku_merged[sku_merged["Gr_YoY"] > 0]["Sales"].sum() / sku_merged["Sales"].sum() * 100) if sku_merged["Sales"].sum() > 0 else 0
    if pos_share > 60: op = f"🟢 **긍정적 포트폴리오:** 성장 중인 SKU가 매출의 {pos_share:.1f}%를 견인하고 있습니다."
    elif pos_share < 30: op = f"🔴 **구조적 위험:** 매출의 {100-pos_share:.1f}%가 역신장 SKU에 의존하고 있습니다. Trouble Maker 구조조정이 시급합니다."
    else: op = f"⚪ **혼조세:** 성장/하락 제품이 혼재되어 있습니다. 신제품의 안착 여부가 중요합니다."
    heimdall_opinion_card("Portfolio Structure Diagnosis", op)

elif selected_view == "🚀 신제품 분석(New)":
    st.markdown("### 🚀 New Product Launch Tracker")
    df_new = df_mkt.dropna(subset=["Launch_WeekIdx"]).copy()
    limit_idx = get_cutoff_week_idx(latest_year, sel_week, 12)
    recent_new_skus = df_new[df_new["Launch_WeekIdx"] >= limit_idx]
    
    if recent_new_skus.empty: st.warning("최근 12주 내 출시된 신제품 데이터가 없습니다.")
    else:
        c1, c2 = st.columns(2)
        with c1: target_sku_name = st.selectbox("1. 분석할 신제품 선택", sorted(recent_new_skus["Product_Name"].unique()))
        target_info = recent_new_skus[recent_new_skus["Product_Name"] == target_sku_name].iloc[0]
        target_line = target_info["Line"]; target_launch_wk = int(target_info["Launch_WeekIdx"])
        st.markdown(f"**Target Info:** {target_sku_name} (Line: {target_line}, Launch: {target_launch_wk})")
        
        same_line_skus = df_mkt[df_mkt["Line"] == target_line]
        top_sellers = same_line_skus.groupby("Product_Name")["Sales"].sum().sort_values(ascending=False).head(10).index.tolist()
        if target_sku_name in top_sellers: top_sellers.remove(target_sku_name)
        with c2: rivals = st.multiselect("2. 비교할 라이벌 선택 (Max 2)", top_sellers + sorted(same_line_skus["Product_Name"].unique()), default=top_sellers[:1], max_selections=2)

        compare_list = [target_sku_name] + rivals
        comp_df = df_mkt[df_mkt["Product_Name"].isin(compare_list)].copy()
        comp_df["Weeks_Since_Launch"] = comp_df.apply(lambda r: (int(str(r["Year"])+"{:02d}".format(r["WeekNum"])) - int(r["Launch_WeekIdx"])) if pd.notnull(r["Launch_WeekIdx"]) else 0, axis=1)
        
        viz_df_list = []
        for sku in compare_list:
            sku_data = comp_df[comp_df["Product_Name"] == sku]
            max_w = sku_data["Weeks_Since_Launch"].max()
            if pd.isna(max_w): max_w = 0
            full_range = pd.DataFrame({"Weeks_Since_Launch": range(int(max_w) + 1)})
            merged_sku = full_range.merge(sku_data, on="Weeks_Since_Launch", how="left")
            merged_sku["Product_Name"] = sku
            merged_sku[["Sales", "Qty", "Store_Count", "Distribution"]] = merged_sku[["Sales", "Qty", "Store_Count", "Distribution"]].fillna(0)
            viz_df_list.append(merged_sku)
        
        if viz_df_list:
            viz_df = pd.concat(viz_df_list)
            viz_df = viz_df[viz_df["Weeks_Since_Launch"] <= 12]
            viz_df["Qty_per_Store"] = viz_df.apply(lambda x: x["Qty"]/x["Store_Count"] if x["Store_Count"]>0 else 0, axis=1)
            viz_df["Sales_per_Store"] = viz_df.apply(lambda x: x["Sales"]/x["Store_Count"] if x["Store_Count"]>0 else 0, axis=1)
            viz_df["ASP"] = viz_df.apply(lambda x: x["Sales"]/x["Qty"] if x["Qty"]>0 else 0, axis=1)
            
            def plot_line(df, y, title):
                fig = px.line(df, x="Weeks_Since_Launch", y=y, color="Product_Name", markers=True, title=title)
                fig.update_traces(line=dict(width=3)); fig.update_xaxes(tickprefix="W+")
                return fig

            t1, t2, t3, t4, t5, t6 = st.tabs(["판매량(회전율)", "매출액(효율)", "총 매출액(규모)", "총 판매량(규모)", "커버리지(취급율)", "평균단가(ASP)"])
            with t1: st.plotly_chart(plot_line(viz_df, "Qty_per_Store", "점당 주간 판매량 (회전율)"), use_container_width=True)
            with t2: st.plotly_chart(plot_line(viz_df, "Sales_per_Store", "점당 주간 매출액 (효율)"), use_container_width=True)
            with t3: st.plotly_chart(plot_line(viz_df, "Sales", "주간 총 매출액 (절대규모)"), use_container_width=True)
            with t4: st.plotly_chart(plot_line(viz_df, "Qty", "주간 총 판매량 (절대규모)"), use_container_width=True)
            with t5: st.plotly_chart(plot_line(viz_df, "Distribution", "주간 취급율"), use_container_width=True)
            with t6: st.plotly_chart(plot_line(viz_df, "ASP", "주간 평균단가"), use_container_width=True)
            
            show_download_button(viz_df, "new_product_tracking_data")

            last_wk = int(viz_df[viz_df["Product_Name"] == target_sku_name]["Weeks_Since_Launch"].max())
            t_qty = viz_df[(viz_df["Product_Name"] == target_sku_name) & (viz_df["Weeks_Since_Launch"] == last_wk)]["Qty_per_Store"].values[0] if not viz_df[viz_df["Product_Name"] == target_sku_name].empty else 0
            rival_data = viz_df[(viz_df["Product_Name"] != target_sku_name) & (viz_df["Weeks_Since_Launch"] == last_wk)]
            if not rival_data.empty:
                r_qty = rival_data["Qty_per_Store"].mean()
                if r_qty > 0:
                    ratio = t_qty / r_qty
                    if ratio > 1.2: op = f"🚀 **초기 돌풍 (W+{last_wk}):** 점당 판매량이 경쟁작 대비 **{ratio:.1f}배** 높습니다. 시장 안착에 성공했습니다."
                    elif ratio < 0.8: op = f"⚠️ **반응 저조 (W+{last_wk}):** 초기 회전율이 경쟁작 대비 낮습니다. 가격 저항이나 맛/품질 이슈를 점검하십시오."
                    else: op = f"⚪ **무난한 출발 (W+{last_wk}):** 경쟁작과 유사한 수준의 초기 성과를 보이고 있습니다."
                else: op = "비교 대상의 데이터가 없습니다."
            else: op = f"출시 {last_wk}주차 데이터가 확인됩니다. 지속적인 모니터링이 필요합니다."
            heimdall_opinion_card("Launch Performance Diagnosis", op)

elif selected_view == "🔎 제품 추적(Tracker)":
    st.markdown("### 🔎 Product Lifecycle Tracker")
    c1, c2 = st.columns(2)
    all_sku_list = sorted(df_mkt["Product_Name"].unique())
    with c1: target_sku_name = st.selectbox("1. 분석할 제품 선택", all_sku_list)
    target_info = df_mkt[df_mkt["Product_Name"] == target_sku_name].iloc[0]
    target_line = target_info["Line"]
    same_line_skus = df_mkt[df_mkt["Line"] == target_line]
    top_sellers = same_line_skus.groupby("Product_Name")["Sales"].sum().sort_values(ascending=False).head(10).index.tolist()
    if target_sku_name in top_sellers: top_sellers.remove(target_sku_name)
    with c2: rivals = st.multiselect("2. 비교할 라이벌 선택 (Max 2)", top_sellers + sorted(same_line_skus["Product_Name"].unique()), default=top_sellers[:1], max_selections=2)
    compare_list = [target_sku_name] + rivals
    viz_df = df_mkt[df_mkt["Product_Name"].isin(compare_list)].copy()
    viz_df["Qty_per_Store"] = viz_df.apply(lambda x: x["Qty"]/x["Store_Count"] if x["Store_Count"]>0 else 0, axis=1)
    viz_df["Sales_per_Store"] = viz_df.apply(lambda x: x["Sales"]/x["Store_Count"] if x["Store_Count"]>0 else 0, axis=1)
    viz_df["ASP"] = viz_df.apply(lambda x: x["Sales"]/x["Qty"] if x["Qty"]>0 else 0, axis=1)
    viz_df = viz_df.sort_values("WeekIndex")
    def plot_line_abs(df, y, title):
        fig = px.line(df, x="WeekIndex", y=y, color="Product_Name", markers=True, title=title)
        fig.update_traces(line=dict(width=2)); fig.update_xaxes(type='category')
        return fig

    t1, t2, t3, t4, t5, t6 = st.tabs(["판매량(회전율)", "매출액(효율)", "총 매출액(규모)", "총 판매량(규모)", "커버리지(취급율)", "평균단가(ASP)"])
    with t1: st.plotly_chart(plot_line_abs(viz_df, "Qty_per_Store", "점당 주간 판매량 추이"), use_container_width=True)
    with t2: st.plotly_chart(plot_line_abs(viz_df, "Sales_per_Store", "점당 주간 매출액 추이"), use_container_width=True)
    with t3: st.plotly_chart(plot_line_abs(viz_df, "Sales", "주간 총 매출액 추이"), use_container_width=True)
    with t4: st.plotly_chart(plot_line_abs(viz_df, "Qty", "주간 총 판매량 추이"), use_container_width=True)
    with t5: st.plotly_chart(plot_line_abs(viz_df, "Distribution", "주간 취급율 추이"), use_container_width=True)
    with t6: st.plotly_chart(plot_line_abs(viz_df, "ASP", "주간 평균단가 추이"), use_container_width=True)
    
    show_download_button(viz_df, "product_lifecycle_data")

    target_trend = viz_df[viz_df["Product_Name"] == target_sku_name]
    if len(target_trend) > 8:
        recent = target_trend.iloc[-4:]["Sales"].mean()
        past = target_trend.iloc[-8:-4]["Sales"].mean()
        if recent > past * 1.05: op = "📈 **상승세:** 최근 4주 평균 매출이 직전 대비 증가 추세입니다."
        elif recent < past * 0.95: op = "📉 **하락세:** 최근 매출 흐름이 둔화되고 있습니다. 원인 파악이 필요합니다."
        else: op = "➡️ **보합세:** 뚜렷한 등락 없이 안정적인 흐름을 유지 중입니다."
    else: op = "데이터 기간이 짧아 장기 추세를 판단하기 어렵습니다."
    heimdall_opinion_card("Lifecycle Trend Diagnosis", op)

elif selected_view == "📚 로직 사전(Appendix)":
    st.markdown("### 📚 HEIMDALL Logic Dictionary")
    
    with st.expander("1. 핵심 KPI 정의 (Key Performance Indicators)", expanded=True):
        st.markdown("""
        - **매출 (Sales):** POS 데이터 상 판매 금액의 총합 (소비자가).
        - **수량 (Qty):** 판매된 제품의 낱개 수량 총합.
        - **취급율 (Distribution):** `(제품이 판매된 점포 수 / 전체 모집단 점포 수) * 100`. 
          > *주의: 제품이 한 개라도 팔린 점포를 '취급 점포'로 간주합니다.*
        - **평균단가 (ASP):** `총 매출 / 총 수량`. 제품의 평균 판매 가격.
        - **점당 회전량 (Velocity):** `총 수량 / 취급 점포 수`. 점포 하나당 평균 판매 개수.
        """)
        
    with st.expander("2. 워터폴(Waterfall) 분석 로직"):
        st.markdown("""
        - **신규 (New):** 작년에는 없었는데 올해 새로 매출이 발생한 제품.
        - **중단 (End):** 작년에는 있었는데 올해 매출이 0인 제품 (이탈).
        - **단가 (Price):** 가격 변동으로 인한 매출 증감분. `(올해 단가 - 작년 단가) * 작년 물량`
        - **물량 (Volume):** 순수 판매량 변화로 인한 매출 증감분. `(올해 물량 - 작년 물량) * 작년 단가`
        - **취급율 (Dist):** 점포 수 확대/축소로 인한 구조적 매출 변동분.
        """)
        
    with st.expander("3. 🆕 [Advanced] 가격 시뮬레이션 및 예측 모델", expanded=True):
        st.markdown(r"""
        #### A. 가격 탄력성 추정 (Estimation Method)
        경제학 표준인 **로그-로그 회귀 (Log-Log Regression)** 모형을 사용하여 탄력성($\beta$)을 추정합니다.
        
        $$
        \ln(Quantity) = \alpha + \beta \cdot \ln(Price) + \epsilon
        $$
        
        * 데이터 포인트($P, Q$)를 로그 스케일로 변환하여 선형 회귀를 수행합니다.
        * 이때 구해진 기울기 $\beta$가 바로 **가격 탄력성 ($E_d$)**입니다.
        * $E_d = -2.0$이면, 가격을 1% 인상할 때 물량은 2% 감소한다는 의미입니다.

        #### B. 시장 시뮬레이션 (Market Simulation)
        사용자가 설정한 가격 변동률($\Delta P$)과 유통 커버리지 가정($\Delta Dist$)을 대입하여 미래를 예측합니다.
        
        1.  **점당 판매량 변화:** 탄력성을 적용하여 점포당 회전율을 재계산합니다.
            $$ Q_{new} = Q_{base} \times (1 + E_d \times \Delta P) $$
        2.  **커버리지 변화:** 사용자의 가정을 반영하여 취급 점포 수를 조정합니다.
            $$ Stores_{new} = Stores_{base} \times (1 + \Delta Dist) $$
        3.  **최종 매출 예측:** $$ Sales_{new} = P_{new} \times Q_{new} \times Stores_{new} $$
        """)
