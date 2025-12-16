import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter

# --------------------------------------------------
# 1. 시스템 설정 & 스타일 (Identity)
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
        .metric-card { background-color: #F8F9F9; border-left: 5px solid #3498DB; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .diagnosis-box { background-color: #ECF0F1; padding: 20px; border-radius: 10px; border: 1px solid #BDC3C7; font-family: 'Courier New'; }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 2. 핵심 로직: 가격 분석 및 데이터 처리
# --------------------------------------------------

def determine_price_status(current_asp, history_asps):
    """
    [HEIMDALL Price Logic v2.0]
    - current_asp: 이번 주 평균 판매 단가
    - history_asps: 지난 12주간의 단가 리스트 (List of floats)
    """
    # 1. 데이터 부족 시 판단 유보
    if not history_asps or len(history_asps) < 4:
        return "New/Unknown"

    # 2. 최빈값(Mode) 계산 - 10원 단위 반올림하여 노이즈 제거
    rounded_history = [round(p, -1) for p in history_asps]
    if not rounded_history:
        return "Error"
        
    count = Counter(rounded_history)
    if not count:
        return "Error"
        
    mode_price = count.most_common(1)[0][0]
    
    if mode_price == 0: return "Error"

    # 3. 변동률 계산
    ratio = current_asp / mode_price

    # 4. 상태 판정 (The 4% Rule)
    if 0.96 <= ratio <= 1.04:
        return "Regular (정상)"
    elif ratio < 0.96:
        if ratio < 0.85:
            return "Deep Promo (초특가)" # 15% 이상 하락
        else:
            return "Promo (행사)"       # 4% ~ 15% 하락
    else: # ratio > 1.04
        return "Price Hike (인상)"

@st.cache_data
def load_parquet(file):
    """업로드된 Parquet 파일을 로드하고 전처리합니다."""
    try:
        df = pd.read_parquet(file)
        # Barcode Standard: 문자열 변환
        if 'Barcode' in df.columns:
            df['Barcode'] = df['Barcode'].astype(str)
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

def generate_diagnosis(row_current, row_prev, price_status):
    """
    [Automated Diagnosis Algorithm]
    매출 변동의 원인을 4단계로 진단하여 텍스트 리포트 생성
    """
    if row_current is None or row_prev is None:
        return "분석할 데이터가 충분하지 않습니다."

    sales_diff = row_current['Sales'] - row_prev['Sales']
    sales_growth = (sales_diff / row_prev['Sales'] * 100) if row_prev['Sales'] > 0 else 0
    
    diagnosis = []
    diagnosis.append(f"**[종합 진단]** 매출 {sales_growth:+.1f}% (YoY 변동액: {sales_diff:,.0f}원)")

    # 1. Existence Check
    if row_prev['Sales'] == 0 and row_current['Sales'] > 500000:
        diagnosis.append("- **🚨 New Entry:** 신규 진입 제품으로 매출 순증 발생.")
    elif row_prev['Sales'] > 100000 and row_current['Sales'] == 0:
        diagnosis.append("- **⚠️ Discontinued:** 기존 주력 제품 이탈(단종/미취급) 발생.")
    
    # 2. Pricing Check
    asp_diff_ratio = (row_current['ASP'] - row_prev['ASP']) / row_prev['ASP'] * 100 if row_prev['ASP'] > 0 else 0
    diagnosis.append(f"- **Pricing:** 가격 상태 **[{price_status}]**. 전년 대비 단가 변동 {asp_diff_ratio:+.1f}%.")
    
    if price_status in ["Promo (행사)", "Deep Promo (초특가)"] and sales_growth > 0:
        diagnosis.append("  → 가격 인하가 매출 볼륨 확대를 성공적으로 견인함.")
    elif price_status == "Price Hike (인상)" and sales_growth < 0:
        diagnosis.append("  → 가격 인상에 따른 물량 저항(Volume Resistance) 발생.")

    # 3. Velocity & Distribution Check
    qty_growth = (row_current['Qty'] - row_prev['Qty']) / row_prev['Qty'] * 100 if row_prev['Qty'] > 0 else 0
    store_growth = (row_current['Store_Count'] - row_prev['Store_Count']) / row_prev['Store_Count'] * 100 if row_prev['Store_Count'] > 0 else 0

    diagnosis.append(f"- **Volume:** 판매 수량 {qty_growth:+.1f}% 변동.")
    diagnosis.append(f"- **Distribution:** 취급 점포수 {store_growth:+.1f}% 변동.")

    return "\n".join(diagnosis)

# --------------------------------------------------
# 3. UI 레이아웃
# --------------------------------------------------
st.markdown('<div class="main-header">🛡️ HEIMDALL GT Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Market Intelligence System for Lotte Wellfood</div>', unsafe_allow_html=True)

# [Sidebar] 파일 업로드 및 필터
with st.sidebar:
    st.header("📂 Data Interface")
    uploaded_file = st.file_uploader("Drop Parquet File Here", type=["parquet"])
    
    if uploaded_file is not None:
        df = load_parquet(uploaded_file)
        if df is not None:
            st.success("데이터 로드 완료")
            
            # 필터링 UI
            all_brands = sorted(df['Brand'].unique())
            selected_brand = st.selectbox("Brand Selection", all_brands)
            
            brand_df = df[df['Brand'] == selected_brand]
            all_skus = sorted(brand_df['Product_Name'].unique())
            selected_sku = st.selectbox("SKU Selection", all_skus)
            
            # 분석 데이터 추출
            sku_df = df[df['Product_Name'] == selected_sku].sort_values('Date')
    else:
        st.info("분석할 Parquet 파일을 업로드해주세요.")
        st.stop()

# --------------------------------------------------
# 4. 메인 대시보드
# --------------------------------------------------

# [Data Prep] 선택된 SKU의 시계열 데이터 준비
if not sku_df.empty:
    current_week_row = sku_df.iloc[-1]
    prev_year_row = sku_df.iloc[-53] if len(sku_df) >= 53 else None # 전년 동기 (약식)
    
    # 가격 로직 적용 (최근 12주)
    recent_12_weeks = sku_df.tail(12)['ASP'].tolist()
    current_price_status = determine_price_status(current_week_row['ASP'], recent_12_weeks)

    # 상단 KPI 지표
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Weekly Sales", f"{current_week_row['Sales']:,.0f} 원", 
                  delta=f"{(current_week_row['Sales'] - prev_year_row['Sales']):,.0f}" if prev_year_row is not None else None)
    with c2:
        st.metric("Weekly Qty", f"{current_week_row['Qty']:,.0f} 개",
                  delta=f"{(current_week_row['Qty'] - prev_year_row['Qty']):,.0f}" if prev_year_row is not None else None)
    with c3:
        st.metric("ASP (Avg Price)", f"{current_week_row['ASP']:,.0f} 원",
                  delta=current_price_status, delta_color="off")
    with c4:
        st.metric("Store Count", f"{current_week_row['Store_Count']:,.0f} 점",
                  delta=f"{(current_week_row['Store_Count'] - prev_year_row['Store_Count']):,.0f}" if prev_year_row is not None else None)

    # [Chart] Volume-Quantity Gap Analysis
    st.subheader("📊 Volume-Quantity Gap Analysis")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 매출 (Bar)
    fig.add_trace(
        go.Bar(x=sku_df['Date'], y=sku_df['Sales'], name="매출(Sales)", marker_color='#3498DB', opacity=0.6),
        secondary_y=False
    )
    # 수량 (Line)
    fig.add_trace(
        go.Scatter(x=sku_df['Date'], y=sku_df['Qty'], name="수량(Qty)", line=dict(color='#E74C3C', width=3)),
        secondary_y=True
    )
    
    fig.update_layout(title_text=f"{selected_sku} 주간 트렌드", template='plotly_white', hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # [Diagnosis] 자동 진단 리포트
    st.subheader("📝 Heimdall Diagnosis")
    with st.container():
        diagnosis_text = generate_diagnosis(current_week_row, prev_year_row, current_price_status)
        st.markdown(f"""
        <div class="diagnosis-box">
            {diagnosis_text.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

    # [Table] 상세 데이터 보기
    with st.expander("🔎 Raw Data View"):
        st.dataframe(sku_df.sort_values('Date', ascending=False).style.format({
            'Sales': '{:,.0f}', 'Qty': '{:,.0f}', 'ASP': '{:,.0f}', 'Store_Count': '{:,.0f}'
        }))

else:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")

# Footer
st.markdown("---")
st.markdown("**System:** HEIMDALL GT (Web Ver.) | **Security:** Local Processing Only | **Version:** 2.1")