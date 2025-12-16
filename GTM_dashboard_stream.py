import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import io

# --------------------------------------------------
# 1. 시스템 설정 (Identity)
# --------------------------------------------------
st.set_page_config(
    page_title="HEIMDALL GT Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }
        .main-header { font-size: 2.0rem; font-weight: 800; color: #2C3E50; margin-bottom: 0.5rem; }
        .sub-header { font-size: 1.0rem; color: #7F8C8D; margin-bottom: 2rem; border-bottom: 2px solid #ECF0F1; padding-bottom: 10px; }
        .diagnosis-box { background-color: #ECF0F1; padding: 20px; border-radius: 10px; border: 1px solid #BDC3C7; font-family: 'Courier New'; }
        .stMetric { background-color: #F8F9F9; padding: 10px; border-radius: 5px; border-left: 5px solid #3498DB; }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 2. 데이터 표준화 및 로드 (Core Engine)
# --------------------------------------------------

def normalize_data(df):
    """
    [Data Integrity Protocol]
    다양한 형태의 컬럼명을 시스템 표준으로 강제 변환합니다.
    """
    # 1. 컬럼명 매핑 (입력 변수 -> 표준 변수)
    col_map = {
        'date': 'Date', 'DATE': 'Date', 'iso_date': 'Date',
        'brand': 'Brand', 'BRAND': 'Brand',
        'product_name': 'Product_Name', 'PRODUCT_NAME': 'Product_Name', 'sku': 'Product_Name',
        'sales': 'Sales', 'SALES': 'Sales', 'amt': 'Sales',
        'qty': 'Qty', 'QTY': 'Qty', 'quantity': 'Qty',
        'asp': 'ASP', 'ASP': 'ASP', 'price': 'ASP',
        'store_count': 'Store_Count', 'STORE_COUNT': 'Store_Count', 'store': 'Store_Count',
        'barcode': 'Barcode', 'BARCODE': 'Barcode'
    }
    
    # 컬럼명 변경 적용
    df = df.rename(columns=col_map)
    
    # 2. 필수 컬럼 존재 여부 체크
    required_cols = ['Date', 'Brand', 'Product_Name', 'Sales', 'Qty', 'ASP', 'Store_Count']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        st.error(f"🚨 데이터 형식이 맞지 않습니다. 다음 컬럼이 누락되었습니다: {missing}")
        st.stop()
        
    # 3. 데이터 타입 강제 변환
    # 날짜: datetime으로 변환
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            st.error("🚨 'Date' 컬럼을 날짜 형식으로 변환할 수 없습니다.")
            st.stop()
            
    # 수치형 데이터: 숫자 외 문자 제거 후 변환
    for col in ['Sales', 'Qty', 'ASP', 'Store_Count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df

@st.cache_data
def load_parquet(file):
    try:
        df = pd.read_parquet(file)
        return normalize_data(df)
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        return None

# --------------------------------------------------
# 3. 비즈니스 로직 (Price & Diagnosis)
# --------------------------------------------------

def determine_price_status(current_asp, history_asps):
    """[HEIMDALL Price Logic v2.0]"""
    if not history_asps or len(history_asps) < 4:
        return "New/Unknown"

    # 최빈값(Mode) 계산 (10원 단위 반올림)
    rounded_history = [round(p, -1) for p in history_asps]
    if not rounded_history: return "Error"
    
    count = Counter(rounded_history)
    mode_price = count.most_common(1)[0][0]
    
    if mode_price == 0: return "Check Data"

    ratio = current_asp / mode_price

    if 0.96 <= ratio <= 1.04: return "Regular (정상)"
    elif ratio < 0.96:
        return "Deep Promo (초특가)" if ratio < 0.85 else "Promo (행사)"
    else: return "Price Hike (인상)"

def generate_diagnosis(row_current, row_prev, price_status):
    """[Automated Diagnosis Algorithm]"""
    if row_current is None: return "데이터 없음"
    
    # 전년 데이터가 없는 경우 (신제품 등)
    if row_prev is None:
        return f"**[신규 진입]** 금주 매출 {row_current['Sales']:,.0f}원. 전년 동기 데이터가 없어 비교 불가."

    sales_diff = row_current['Sales'] - row_prev['Sales']
    sales_growth = (sales_diff / row_prev['Sales'] * 100) if row_prev['Sales'] > 0 else 0
    
    diagnosis = []
    diagnosis.append(f"**[종합 진단]** 매출 {sales_growth:+.1f}% (YoY 변동액: {sales_diff:,.0f}원)")

    # 1. Existence
    if row_prev['Sales'] == 0 and row_current['Sales'] > 500000:
        diagnosis.append("- **🚨 New Entry:** 신규 진입 제품.")
    elif row_prev['Sales'] > 100000 and row_current['Sales'] == 0:
        diagnosis.append("- **⚠️ Discontinued:** 주력 제품 이탈 의심.")
    
    # 2. Pricing
    asp_diff = row_current['ASP'] - row_prev['ASP']
    diagnosis.append(f"- **Pricing:** 현재상태 **[{price_status}]**. (YoY {asp_diff:+.0f}원)")
    
    # 3. Volume & Dist
    qty_growth = (row_current['Qty'] - row_prev['Qty']) / row_prev['Qty'] * 100 if row_prev['Qty'] > 0 else 0
    store_growth = (row_current['Store_Count'] - row_prev['Store_Count']) / row_prev['Store_Count'] * 100 if row_prev['Store_Count'] > 0 else 0

    diagnosis.append(f"- **Volume:** 판매량 {qty_growth:+.1f}%")
    diagnosis.append(f"- **Coverage:** 취급점 {store_growth:+.1f}%")

    return "\n".join(diagnosis)

# --------------------------------------------------
# 4. UI 레이아웃
# --------------------------------------------------
st.markdown('<div class="main-header">🛡️ HEIMDALL GT Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Market Intelligence System for Lotte Wellfood</div>', unsafe_allow_html=True)

# [Sidebar]
with st.sidebar:
    st.header("📂 Data Interface")
    uploaded_file = st.file_uploader("Drop Parquet File Here", type=["parquet"])
    
    if uploaded_file is not None:
        df = load_parquet(uploaded_file)
        if df is not None:
            st.success(f"데이터 로드 완료 ({len(df):,} rows)")
            
            # 필터링
            all_brands = sorted(df['Brand'].unique())
            selected_brand = st.selectbox("Brand", all_brands)
            
            brand_df = df[df['Brand'] == selected_brand]
            all_skus = sorted(brand_df['Product_Name'].unique())
            selected_sku = st.selectbox("SKU", all_skus)
            
            sku_df = brand_df[brand_df['Product_Name'] == selected_sku].sort_values('Date')
        else:
            st.stop()
    else:
        st.info("분석할 Parquet 파일을 업로드해주세요.")
        st.stop()

# [Main Dashboard]
if not sku_df.empty:
    # 최신 주차 및 전년 동기 주차 찾기
    current_row = sku_df.iloc[-1]
    
    # 단순히 52주 전 인덱스로 찾지 않고, 실제 Date 기준으로 1년 전 데이터를 찾음 (더 정확함)
    target_date_1y_ago = current_row['Date'] - pd.DateOffset(weeks=52)
    
    # 1년 전과 가장 가까운 날짜의 데이터 찾기 (오차 범위 7일 이내)
    prev_year_df = sku_df[
        (sku_df['Date'] >= target_date_1y_ago - pd.Timedelta(days=3)) & 
        (sku_df['Date'] <= target_date_1y_ago + pd.Timedelta(days=3))
    ]
    
    prev_row = prev_year_df.iloc[0] if not prev_year_df.empty else None

    # 가격 상태 진단
    recent_asps = sku_df.tail(12)['ASP'].tolist()
    price_status = determine_price_status(current_row['ASP'], recent_asps)

    # 1. KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        delta_val = (current_row['Sales'] - prev_row['Sales']) if prev_row is not None else 0
        st.metric("Weekly Sales", f"{current_row['Sales']:,.0f} 원", f"{delta_val:,.0f} (YoY)")
    with k2:
        delta_qty = (current_row['Qty'] - prev_row['Qty']) if prev_row is not None else 0
        st.metric("Weekly Qty", f"{current_row['Qty']:,.0f} 개", f"{delta_qty:,.0f} (YoY)")
    with k3:
        st.metric("ASP (Avg Price)", f"{current_row['ASP']:,.0f} 원", price_status, delta_color="off")
    with k4:
        delta_store = (current_row['Store_Count'] - prev_row['Store_Count']) if prev_row is not None else 0
        st.metric("Store Count", f"{current_row['Store_Count']:,.0f} 점", f"{delta_store:,.0f} (YoY)")

    # 2. Chart (Volume-Quantity Gap)
    st.subheader(f"📊 {selected_sku} Trend Analysis")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(x=sku_df['Date'], y=sku_df['Sales'], name="매출(Sales)", marker_color='#3498DB', opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=sku_df['Date'], y=sku_df['Qty'], name="수량(Qty)", line=dict(color='#E74C3C', width=3)), secondary_y=True)
    
    fig.update_layout(height=400, template='plotly_white', hovermode="x unified")
    fig.update_yaxes(title_text="매출 (원)", secondary_y=False)
    fig.update_yaxes(title_text="수량 (개)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Diagnosis
    st.subheader("📝 Heimdall Diagnosis")
    diag_text = generate_diagnosis(current_row, prev_row, price_status)
    st.markdown(f'<div class="diagnosis-box">{diag_text.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    # 4. Data Grid
    with st.expander("🔎 Raw Data View"):
        st.dataframe(sku_df.sort_values('Date', ascending=False).style.format({
            'Sales': '{:,.0f}', 'Qty': '{:,.0f}', 'ASP': '{:,.0f}', 'Store_Count': '{:,.0f}', 'Date': '{:%Y-%m-%d}'
        }))

else:
    st.warning("선택된 데이터가 없습니다.")

st.markdown("---")
st.caption("System: HEIMDALL GT (Web Ver 2.2 Stable) | Powered by Streamlit")
