import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import os
import numpy as np

# -----------------------------------------------------------------------------
# 1. 아이패드/모바일 최적화 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="GTM Dashboard", page_icon="📈", layout="wide")

# CSS: 아이패드 가독성을 위한 폰트 확대 및 여백 조정
st.markdown("""
<style>
    /* 상단 여백 줄이기 (모바일 화면 확보) */
    .block-container {padding-top: 1.5rem !important; padding-bottom: 3rem !important;}
    
    /* KPI 라벨 (좀 더 진하게) */
    div[data-testid="stMetricLabel"] {
        font-size: 18px !important; 
        font-weight: 600 !important;
        color: #666666 !important;
    }
    /* KPI 숫자 (크고 시원하게) */
    div[data-testid="stMetricValue"] {
        font-size: 36px !important; 
        font-weight: 800 !important;
        color: #2C3E50 !important;
    }
    /* 탭 글씨 키우기 */
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 10px !important;
    }
    /* 데이터프레임 헤더 강조 */
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 로드 (보안 업로드 방식)
# -----------------------------------------------------------------------------
st.sidebar.title("🔐 Secure Loader")
st.sidebar.info("보안을 위해 서버에 데이터를 저장하지 않습니다.\n**'일자별실적_Master.parquet'** 파일을 아래에 넣어주세요.")

uploaded_file = st.sidebar.file_uploader("데이터 파일 업로드", type=["parquet"])

# 파일이 없으면 대기 화면 표시
if uploaded_file is None:
    st.header("👋 GTM Sales Dashboard (iPad Ver.)")
    st.markdown("""
    ### 📲 사용 가이드
    1. 왼쪽 사이드바를 여세요 (**>** 버튼).
    2. **`일자별실적_Master.parquet`** 파일을 업로드 박스에 넣으세요.
    3. (아이패드 팁) **파일 앱**에서 파일을 끌어다 놓으면(Drag & Drop) 편합니다.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2906/2906274.png", width=150)
    st.stop()

# 파일 로드 및 전처리 (로직 유지)
@st.cache_data(ttl="1h")
def load_data(file):
    try:
        df = pd.read_parquet(file)
        
        num_cols = ['sales_price', 'gross_sales', 'sales_box_qty']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        str_cols = ['channel_name','customer_name','category','brand','sku_name']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace(['','nan','None'], '미지정')
                
        return df
    except Exception as e:
        return pd.DataFrame()

df_master = load_data(uploaded_file)

if df_master.empty:
    st.error("🚨 파일 형식이 올바르지 않습니다. Refinery로 생성된 Parquet 파일을 넣어주세요.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 사이드바 (컨트롤 타워)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.markdown("### 📅 조회 기간 설정")
    
    min_d, max_d = df_master['date'].min().date(), df_master['date'].max().date()
    def_start = max_d - timedelta(days=30)
    if def_start < min_d: def_start = min_d
    
    # 모바일에서는 달력 입력이 작으므로 컬럼 없이 한 줄씩 배치
    s_date = st.date_input("시작일 (Start)", def_start, min_value=min_d, max_value=max_d)
    e_date = st.date_input("종료일 (End)", max_d, min_value=min_d, max_value=max_d)
    
    st.divider()
    view_mode = st.radio("트리맵 기준", ["1. 매출 성장성 (매출증감%)", "2. 매출&D/C 추이 (증감%p)"])
    
    st.caption("※ 아이패드 가로 모드를 권장합니다.")

# -----------------------------------------------------------------------------
# 4. 데이터 엔진 & KPI (로직 100% 유지)
# -----------------------------------------------------------------------------
mask_c = (df_master['date'].dt.date >= s_date) & (df_master['date'].dt.date <= e_date)
mask_l = (df_master['date'].dt.date >= (s_date - timedelta(weeks=52))) & (df_master['date'].dt.date <= (e_date - timedelta(weeks=52)))

df_cy = df_master[mask_c].copy()
df_ly = df_master[mask_l].copy()

# 전사 실적 집계
tc_amt = df_cy['sales_price'].sum()
tl_amt = df_ly['sales_price'].sum()
tc_qty = df_cy['sales_box_qty'].sum()
tl_qty = df_ly['sales_box_qty'].sum()

# D/C율
dc_cy = (df_cy['gross_sales'].sum() - tc_amt)/df_cy['gross_sales'].sum()*100 if df_cy['gross_sales'].sum() else 0
dc_ly = (df_ly['gross_sales'].sum() - tl_amt)/df_ly['gross_sales'].sum()*100 if df_ly['gross_sales'].sum() else 0

st.markdown(f"### 🚀 Summary ({s_date} ~ {e_date})")

# KPI 배치 (모바일 호환성을 위해 컨테이너 활용)
k1, k2, k3, k4 = st.columns(4)

k1.metric("총 납품매출", f"{tc_amt/1e8:,.1f}억", f"{(tc_amt-tl_amt)/tl_amt*100:+.1f}%" if tl_amt else "0%")
k2.metric("평균 D/C율", f"{dc_cy:.1f}%", f"{dc_cy-dc_ly:+.1f}%p", delta_color="inverse")
qty_growth = ((tc_qty - tl_qty) / tl_qty * 100) if tl_qty else 0
k3.metric("판매수량", f"{tc_qty/1000:,.1f}천Box", f"{qty_growth:+.1f}%")
asp_cy = tc_amt/tc_qty if tc_qty else 0
asp_ly = tl_amt/tl_qty if tl_qty else 0
asp_growth = ((asp_cy - asp_ly) / asp_ly * 100) if asp_ly else 0
k4.metric("ASP (단가)", f"{asp_cy:,.0f}원", f"{asp_growth:+.1f}%")

st.divider()

# -----------------------------------------------------------------------------
# 5. 집계 함수 (로직 유지)
# -----------------------------------------------------------------------------
def get_agg(d1, d2, grp):
    if d1.empty and d2.empty: return pd.DataFrame()
    c = d1.groupby(grp)[['sales_price','gross_sales']].sum().reset_index()
    l = d2.groupby(grp)[['sales_price','gross_sales']].sum().reset_index()
    m = pd.merge(c, l, on=grp, how='outer', suffixes=('_c','_l')).fillna(0)
    
    m['grw'] = m.apply(lambda x: ((x['sales_price_c'] - x['sales_price_l']) / x['sales_price_l'] * 100) if x['sales_price_l']!=0 else 0, axis=1)
    m['grw_gross'] = m.apply(lambda x: ((x['gross_sales_c'] - x['gross_sales_l']) / x['gross_sales_l'] * 100) if x['gross_sales_l']!=0 else 0, axis=1)
    
    m['dc_c'] = m.apply(lambda x: (x['gross_sales_c'] - x['sales_price_c'])/x['gross_sales_c']*100 if x['gross_sales_c']!=0 else 0, axis=1)
    m['dc_l'] = m.apply(lambda x: (x['gross_sales_l'] - x['sales_price_l'])/x['gross_sales_l']*100 if x['gross_sales_l']!=0 else 0, axis=1)
    m['dc_diff'] = m['dc_c'] - m['dc_l']
    
    total_gross_c = m['gross_sales_c'].sum()
    m['gross_share'] = m['gross_sales_c'] / total_gross_c if total_gross_c != 0 else 0
    m['dc_contrib'] = m['dc_diff'] * m['gross_share']
    
    m['amt_100m'] = m['sales_price_c'] / 1e8
    
    if '1.' in view_mode:
        m['label_txt'] = m.apply(lambda x: f"{x[grp[-1]]}<br>{x['amt_100m']:.1f}억<br>{x['grw']:+.1f}%", axis=1)
    else:
        m['label_txt'] = m.apply(lambda x: f"{x[grp[-1]]}<br>{x['amt_100m']:.1f}억<br>{x['dc_diff']:+.1f}%p", axis=1)
    return m

# -----------------------------------------------------------------------------
# 6. 상단 트리맵
# -----------------------------------------------------------------------------
st.subheader("1️⃣ Market Map (Category ➝ Brand)")
df_top = get_agg(df_cy, df_ly, ['category','brand'])
df_top = df_top[df_top['sales_price_c'] > 0]

if "1." in view_mode:
    val, col, clr, rn = 'sales_price_c', 'grw', 'RdBu', [-30, 30]
else:
    val, col, clr, rn = 'sales_price_c', 'dc_diff', 'RdYlGn_r', [-5, 5]

if not df_top.empty:
    fig = px.treemap(df_top, path=['category','brand'], values=val, color=col, color_continuous_scale=clr, range_color=rn, color_continuous_midpoint=0)
    fig.update_traces(text=df_top['label_txt'], textinfo="text", textfont=dict(size=20)) # 폰트 사이즈 업
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=350) # 높이 확보
    st.plotly_chart(fig, use_container_width=True)
else: st.info("데이터가 없습니다.")

st.divider()

# -----------------------------------------------------------------------------
# 7. 상세 분석 필터 (모바일 배치 최적화)
# -----------------------------------------------------------------------------
st.markdown("#### 🔍 Detail Analysis")
cats_list = ['전체'] + sorted(df_master['category'].unique().tolist())

# 모바일에서는 3단 컬럼이 너무 좁을 수 있어 비율 조정
c1, c2, c3 = st.columns([1,1,1])
sel_cat = c1.selectbox("라인 (Category)", cats_list)

if sel_cat == '전체': 
    brands_list = ['전체'] + sorted(df_master['brand'].unique().tolist())
else: 
    brands_list = ['전체'] + sorted(df_master[df_master['category'] == sel_cat]['brand'].unique().tolist())
sel_brand = c2.selectbox("브랜드 (Brand)", brands_list)

if sel_brand == '전체':
    if sel_cat == '전체': skus_list = ['전체']
    else: skus_list = ['전체'] + sorted(df_master[df_master['category'] == sel_cat]['sku_name'].unique().tolist())
else:
    skus_list = ['전체'] + sorted(df_master[df_master['brand'] == sel_brand]['sku_name'].unique().tolist())
sel_sku = c3.selectbox("제품 (SKU)", skus_list)

# Filtering logic
target_name = "전사 (Total)"
sc = df_cy; sl = df_ly

if sel_sku != '전체':
    target_name = sel_sku
    sc = df_cy[df_cy['sku_name'] == sel_sku]
    sl = df_ly[df_ly['sku_name'] == sel_sku]
elif sel_brand != '전체':
    target_name = sel_brand
    sc = df_cy[df_cy['brand'] == sel_brand]
    sl = df_ly[df_ly['brand'] == sel_brand]
elif sel_cat != '전체':
    target_name = sel_cat
    sc = df_cy[df_cy['category'] == sel_cat]
    sl = df_ly[df_ly['category'] == sel_cat]

st.subheader(f"📊 '{target_name}' 분석 결과")

# -----------------------------------------------------------------------------
# 8. 하단 상세 분석 (탭 방식)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📋 상세 리스트", "🌊 원인 분석 (Waterfall)"])

with tab1:
    if sel_sku != '전체': 
        grp_col = 'customer_name'; col_kor = '거래처'
    elif sel_brand != '전체': 
        grp_col = 'sku_name'; col_kor = '제품명'
    elif sel_cat != '전체': 
        grp_col = 'brand'; col_kor = '브랜드'
    else: 
        grp_col = 'category'; col_kor = '라인'
    
    t = get_agg(sc, sl, [grp_col])
    if not t.empty:
        # 아이패드에서 보기 좋게 컬럼 수 줄이기 (핵심만)
        tbl = t[[grp_col,'amt_100m','grw','dc_c','dc_diff']].copy()
        tbl.columns=[col_kor,'매출(억)','성장(%)','D/C(%)','D/C변동']
        
        st.dataframe(
            tbl.sort_values('매출(억)', ascending=False).style
            .format({'매출(억)':'{:,.1f}', '성장(%)':'{:+.1f}%', 'D/C(%)':'{:.1f}%', 'D/C변동':'{:+.1f}%p'})
            .background_gradient(subset=['성장(%)'], cmap='RdYlGn', vmin=-20, vmax=20)
            .bar(subset=['D/C변동'], align='mid', color=['#FF6B6B', '#009688']),
            use_container_width=True, hide_index=True, height=500
        )
    else: st.info("데이터가 없습니다.")

with tab2:
    if not sc.empty or not sl.empty:
        # [cite_start]PVM 로직 (Source 0과 동일) [cite: 18, 19]
        pvm_c = sc.groupby('sku_name')[['sales_price','sales_box_qty']].sum()
        pvm_l = sl.groupby('sku_name')[['sales_price','sales_box_qty']].sum()
        m_pvm = pd.merge(pvm_c, pvm_l, on='sku_name', how='outer', suffixes=('_c','_l')).fillna(0)
        
        m_pvm['asp_c'] = np.where(m_pvm['sales_box_qty_c']==0, 0, m_pvm['sales_price_c']/m_pvm['sales_box_qty_c'])
        m_pvm['asp_l'] = np.where(m_pvm['sales_box_qty_l']==0, 0, m_pvm['sales_price_l']/m_pvm['sales_box_qty_l'])
        
        new_cond = (m_pvm['sales_price_l'] == 0) & (m_pvm['sales_price_c'] > 0)
        lost_cond = (m_pvm['sales_price_l'] > 0) & (m_pvm['sales_price_c'] == 0)
        
        val_new = m_pvm.loc[new_cond, 'sales_price_c'].sum() / 1e8
        val_lost = -1 * m_pvm.loc[lost_cond, 'sales_price_l'].sum() / 1e8
        
        exist = m_pvm[~new_cond & ~lost_cond].copy()
        vol_eff = ((exist['sales_box_qty_c'] - exist['sales_box_qty_l']) * exist['asp_l']).sum() / 1e8
        price_eff = ((exist['asp_c'] - exist['asp_l']) * exist['sales_box_qty_c']).sum() / 1e8
        
        tot_ly = sl['sales_price'].sum() / 1e8
        tot_cy = sc['sales_price'].sum() / 1e8
        
        calc_sum = tot_ly + val_new + val_lost + vol_eff + price_eff
        resid = tot_cy - calc_sum
        price_eff += resid 
        
        x_vals = ["작년", "단종📉", "신규🚀", "물량📦", "가격/DC🏷️", "올해"]
        y_vals = [tot_ly, val_lost, val_new, vol_eff, price_eff, tot_cy]
        measure_vals = ["relative", "relative", "relative", "relative", "relative", "total"]
        
        fig_w = go.Figure(go.Waterfall(
            orientation = "v", measure = measure_vals, x = x_vals, y = y_vals,
            text = [f"{v:+.1f}" if i not in [0,5] else f"{v:.1f}" for i, v in enumerate(y_vals)],
            textposition = "outside",
            connector = {"line":{"color":"#555555"}},
            decreasing = {"marker":{"color":"#FF6B6B"}},
            increasing = {"marker":{"color":"#009688"}},
            totals = {"marker":{"color":"#2C3E50"}},
            textfont = dict(size=18, color="black") # 모바일용 폰트 조절
        ))
        
        fig_w.update_layout(title="매출 증감 원인 (단위: 억)", margin=dict(t=50), height=400)
        st.plotly_chart(fig_w, use_container_width=True)
        
        st.info(f"""
        💡 **Insight Note:**
        작년 매출 **{tot_ly:.1f}억**에서 올해 **{tot_cy:.1f}억**이 되었습니다.
        가장 큰 영향은 **{'물량(Box)' if abs(vol_eff) > abs(price_eff) else '할인/단가'}** 요인입니다.
        """)
    else: st.info("분석할 데이터가 없습니다.")