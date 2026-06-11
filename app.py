"""
app.py - AI 주식 리서치 에이전트
실행: streamlit run app.py --server.port 8502
"""
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/quant_project.db"

st.set_page_config(
    page_title="AI 주식 리서치 에이전트",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 공통 유틸 ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_connection():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def safe_read(table):
    con = get_connection()
    if con is None:
        return pd.DataFrame()
    try:
        return pd.read_sql(f"SELECT * FROM {table}", con)
    except Exception:
        return pd.DataFrame()


def db_exists():
    return os.path.exists(DB_PATH)


def table_exists(table):
    con = get_connection()
    if con is None:
        return False
    try:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        return table in tables
    except Exception:
        return False


def get_all_tables():
    con = get_connection()
    if con is None:
        return []
    try:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [r[0] for r in cur.fetchall()]
    except Exception:
        return []


REGIME_COLOR = {"강세장": "#00aa44", "중립장": "#ddaa00", "약세장": "#ee6600", "하락장": "#cc0000"}

# 국면별 팩터 가중치 (06_market_regime.py 기준값)
ALL_REGIME_WEIGHTS = {
    "강세장": {"수익성": 1.2, "밸류에이션": 1.0, "모멘텀": 1.3, "안정성": 0.8, "유동성": 1.0},
    "중립장": {"수익성": 1.0, "밸류에이션": 1.1, "모멘텀": 1.0, "안정성": 1.0, "유동성": 1.0},
    "약세장": {"수익성": 0.9, "밸류에이션": 1.1, "모멘텀": 0.7, "안정성": 1.4, "유동성": 1.2},
    "하락장": {"수익성": 0.8, "밸류에이션": 1.0, "모멘텀": 0.5, "안정성": 1.5, "유동성": 1.3},
}

FACTORS = ["수익성", "밸류에이션", "모멘텀", "안정성", "유동성"]


# ── 사이드바 ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 AI 주식 리서치 에이전트")
    st.divider()

    db_ok = db_exists()
    if db_ok:
        st.success("DB 연결됨")
    else:
        st.error("DB 없음 — run_pipeline.py 먼저 실행")
        st.code("python run_pipeline.py", language="bash")

    st.divider()
    st.markdown("**파이프라인 실행**")
    st.code("python run_pipeline.py", language="bash")
    st.markdown("**대시보드 실행**")
    st.code("streamlit run app.py --server.port 8502", language="bash")
    st.divider()

    # 데이터 신선도
    st.markdown("**데이터 업데이트 현황**")
    _freshness_tables = [
        ("price_data",                        "주가"),
        ("market_regime",                     "시장국면"),
        ("score_result_final_ai_by_model_all","AI점수"),
        ("research_consensus_factor_all",     "컨센서스"),
        ("ml_prediction_score_by_model_all",  "ML예측"),
    ]
    _con_f = get_connection()
    for _tbl, _label in _freshness_tables:
        if _con_f:
            try:
                _cnt = _con_f.execute(f"SELECT COUNT(*) FROM {_tbl}").fetchone()[0]
                st.caption(f"✅ {_label}: {_cnt}행")
            except Exception:
                st.caption(f"❌ {_label}: 없음")
        else:
            st.caption(f"❌ {_label}: DB없음")

    st.divider()
    st.caption("⚠️ 본 도구는 교육·분석 목적 참고자료이며 투자 권유가 아닙니다.")


# ── 메인 탭 ──────────────────────────────────────────────────────────────────

tab_overview, tab_regime, tab_stock, tab_portfolio, tab_report = st.tabs(
    ["📋 개요", "🌐 시장 국면", "🔍 종목 선정", "💼 포트폴리오", "🤖 AI 리포트"]
)


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: 개요
# ═══════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("📋 프로젝트 개요")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **AI 기반 한국 주식 리서치 & 포트폴리오 의사결정 에이전트**

        본 시스템은 한국 주식 데이터를 기반으로 다음을 통합합니다:
        - **퀀트 팩터 스코어링**: 수익성·밸류에이션·모멘텀·안정성·유동성
        - **시장 국면 판단**: KOSPI 기반 국면(강세/중립/약세/하락)에 따른 팩터 가중치 조정
        - **리서치/컨센서스 팩터**: 투자의견·목표주가·리포트 수 (FnGuide 실제 데이터)
        - **ML 예측**: LogisticRegression·RandomForest·GradientBoosting
        - **포트폴리오 최적화**: 동일비중·최소분산·최대샤프·Risk Parity·AI점수가중
        - **AI 리포트**: Evidence JSON 기반 자동 리포트 (OpenAI 선택)

        **최종 AI 점수 = 리서치 강화 점수 70% + ML 점수 20% + 위험방어 점수 10%**
        """)

    with col2:
        st.markdown("**분석 흐름**")
        st.markdown("""
        ```
        01 데이터 수집
            ↓
        02 퀀트 스코어링
            ↓
        03 포트폴리오 백테스트
            ↓
        06 시장 국면 판단
            ↓
        07 리서치/컨센서스
            ↓
        09 ML 예측
            ↓
        08 고도화 포트폴리오
            ↓
        04 AI 리포트 생성
        ```
        """)

    st.divider()
    st.subheader("DB 테이블 상태")

    required_tables = [
        ("price_data", "가격 데이터"),
        ("stock_master", "종목 마스터"),
        ("benchmark_price", "벤치마크(KOSPI)"),
        ("score_result_all", "퀀트 스코어링"),
        ("market_regime", "시장 국면"),
        ("score_result_regime_adjusted", "국면 반영 점수"),
        ("research_consensus_factor_all", "리서치/컨센서스"),
        ("score_result_research_enhanced_all", "리서치 강화 점수"),
        ("ml_prediction_score_by_model_all", "ML 예측"),
        ("score_result_final_ai_by_model_all", "최종 AI 점수"),
        ("advanced_portfolio_performance", "고도화 포트폴리오"),
        ("portfolio_risk_contribution", "위험기여도"),
        ("ai_report_result_final_enhanced", "AI 리포트"),
    ]

    all_tables = get_all_tables()
    status_data = []
    for table, desc in required_tables:
        exists = table in all_tables
        count = 0
        if exists:
            try:
                con = get_connection()
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                pass
        status_data.append({
            "테이블": table,
            "설명": desc,
            "상태": "✅ 존재" if exists else "❌ 없음",
            "행 수": count if exists else "-",
        })

    st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)

    if not db_exists():
        st.warning("파이프라인을 먼저 실행하세요: `python run_pipeline.py`")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: 시장 국면
# ═══════════════════════════════════════════════════════════════════════
with tab_regime:
    st.header("🌐 시장 국면 판단")

    regime_df = safe_read("market_regime")
    if regime_df.empty:
        st.warning("시장 국면 데이터 없음. 파이프라인을 실행하세요.")
        st.stop()

    r = regime_df.iloc[0]
    regime_name = r.get("regime", "N/A")
    color = REGIME_COLOR.get(regime_name, "#888888")

    # 핵심 지표 요약
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("KOSPI", f"{r['kospi_close']:,.0f}pt")
    col2.metric("1M 수익률", f"{r['ret_1m']:.2f}%")
    col3.metric("3M 수익률", f"{r['ret_3m']:.2f}%")
    col4.metric("20일 변동성", f"{r['vol_20']:.2f}%")
    col5.metric("1년 MDD", f"{r['mdd_1y']:.2f}%")

    st.divider()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown(
            f"### 현재 국면: <span style='color:{color}; font-size:1.4em'>{regime_name}</span>",
            unsafe_allow_html=True,
        )
        st.metric("종합 점수", f"{r['total_score']:.1f} / 100")
        st.caption("판단 기준: ≥65 강세장 / ≥45 중립장 / ≥30 약세장 / <30 하락장")
        st.markdown(f"**추천 전략**: {r['strategy_rec']}")
        st.markdown(
            f"**200일선 대비**: {'위 📈' if r['above_ma200'] else '아래 📉'} "
            f"(MA200: {r['ma200']:,.0f}pt)"
        )

        st.divider()
        st.markdown("**세부 점수 분석** (종합점수 산출 근거)")

        # 모멘텀 신호 (06_market_regime.py 동일 공식)
        ret_6m = float(r.get("ret_6m", 0))
        mom_signal = (float(r["ret_1m"]) + float(r["ret_3m"]) * 0.5 + ret_6m * 0.3) / 3
        mom_signal_c = float(np.clip(mom_signal, -20, 20))

        score_details = [
            {
                "label": "모멘텀 점수",
                "val": float(r["momentum_score"]),
                "weight": 35,
                "raw": f"1M={float(r['ret_1m']):+.2f}%, 3M={float(r['ret_3m']):+.2f}%, 6M={ret_6m:+.2f}%",
                "logic": f"신호={mom_signal_c:.2f} → (신호+20)/40×100",
            },
            {
                "label": "변동성 점수",
                "val": float(r["vol_score"]),
                "weight": 25,
                "raw": f"20일 변동성={float(r['vol_20']):.2f}% (클램핑 범위: 5~40%)",
                "logic": f"낮을수록 高점수 → (40−{float(r['vol_20']):.1f})/35×100",
            },
            {
                "label": "MDD 점수",
                "val": float(r["mdd_score"]),
                "weight": 25,
                "raw": f"1년 최대낙폭={float(r['mdd_1y']):.2f}% (클램핑 범위: 0~30%)",
                "logic": f"낙폭 작을수록 高점수 → (30−{abs(float(r['mdd_1y'])):.1f})/30×100",
            },
            {
                "label": "200일선 점수",
                "val": float(r["ma_score"]),
                "weight": 15,
                "raw": f"KOSPI {float(r['kospi_close']):,.0f}pt  vs  MA200 {float(r['ma200']):,.0f}pt",
                "logic": f"200일선 {'위 → 100점' if r['above_ma200'] else '아래 → 0점'}",
            },
        ]

        for sd in score_details:
            contrib = sd["val"] * sd["weight"] / 100
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{sd['label']}** (가중 {sd['weight']}%)")
            c2.markdown(f"**{sd['val']:.1f}점** / 기여 {contrib:.1f}점")
            st.progress(int(min(sd["val"], 100)) / 100)
            st.caption(f"📊 {sd['raw']}\n🧮 {sd['logic']}")
            if sd["val"] <= 0:
                st.warning(
                    f"⚠️ {sd['label']} 0점: {sd['raw']}가 클램핑 범위 최대치에 도달하여 최저점 처리됩니다. "
                    f"이는 버그가 아니라 현재 시장 지표값이 정규화 범위를 벗어난 것입니다."
                )

        total_check = sum(sd["val"] * sd["weight"] / 100 for sd in score_details)
        st.markdown(
            f"**종합점수** = {score_details[0]['val']:.1f}×35% + {score_details[1]['val']:.1f}×25% "
            f"+ {score_details[2]['val']:.1f}×25% + {score_details[3]['val']:.1f}×15% "
            f"= **{r['total_score']:.1f}점**"
        )

    with col_right:
        # KOSPI 가격 차트
        bm_df = safe_read("benchmark_price")
        if not bm_df.empty:
            bm_df["date"] = pd.to_datetime(bm_df["date"])
            bm_df = bm_df.sort_values("date")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bm_df["date"], y=bm_df["close"],
                name="KOSPI", line=dict(color="#1976D2", width=2),
                fill="tozeroy", fillcolor="rgba(25,118,210,0.1)"
            ))
            if len(bm_df) >= 200:
                bm_df["ma200"] = bm_df["close"].rolling(200).mean()
                fig.add_trace(go.Scatter(
                    x=bm_df["date"], y=bm_df["ma200"],
                    name="200일선", line=dict(color="orange", dash="dash", width=1.5)
                ))

            fig.update_layout(
                title="KOSPI 지수 (최근 3년)",
                xaxis_title="날짜", yaxis_title="지수",
                height=380, margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── 국면별 팩터 가중치 ─────────────────────────────────────────────────
    st.divider()
    st.subheader("국면별 팩터 가중치")
    st.caption(
        "각 국면 진입 시 퀀트 팩터에 적용되는 승수입니다. "
        "1.0 = 기준, >1.0 = 비중 확대, <1.0 = 비중 축소. "
        f"현재 국면 **{regime_name}** 행이 강조됩니다."
    )

    regime_rows = []
    for rname, wmap in ALL_REGIME_WEIGHTS.items():
        row = {
            "국면": rname,
            "판단 기준(종합점수)": "≥65" if rname == "강세장" else (
                "≥45" if rname == "중립장" else ("≥30" if rname == "약세장" else "<30")
            ),
        }
        row.update(wmap)
        regime_rows.append(row)
    regime_weight_df = pd.DataFrame(regime_rows)

    def _highlight_regime(row):
        if row["국면"] == regime_name:
            return ["background-color:#fff9c4; font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        regime_weight_df.style.apply(_highlight_regime, axis=1),
        use_container_width=True, hide_index=True,
    )

    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        current_w = ALL_REGIME_WEIGHTS.get(regime_name, {f: 1.0 for f in FACTORS})
        fig_radar = go.Figure(go.Scatterpolar(
            r=[current_w[f] for f in FACTORS] + [current_w[FACTORS[0]]],
            theta=FACTORS + [FACTORS[0]],
            fill="toself", name=regime_name,
            line_color=color,
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[1.0] * (len(FACTORS) + 1),
            theta=FACTORS + [FACTORS[0]],
            fill=None, name="기준(1.0)", line=dict(color="gray", dash="dot"),
        ))
        fig_radar.update_layout(
            title=f"{regime_name} 팩터 가중치 레이더",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.7])),
            height=350, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r2:
        st.markdown(f"#### {regime_name} 투자 전략")
        st.info(r["strategy_rec"])
        st.markdown("**팩터 가중치 적용 공식**")
        st.markdown("""
        ```
        조정 퀀트점수 =
          수익성 × w_p × 20%
        + 밸류에이션 × w_v × 20%
        + 모멘텀 × w_m × 25%
        + 안정성 × w_s × 20%
        + 유동성 × w_l × 15%
        (→ min-max 정규화 후 0~100 점수화)
        ```
        """)
        cw_df = pd.DataFrame([
            {
                "팩터": f,
                "현재 국면 승수": v,
                "해석": "▲ 비중 확대" if v > 1.05 else ("▼ 비중 축소" if v < 0.95 else "— 기준"),
            }
            for f, v in current_w.items()
        ])
        st.dataframe(cw_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: 종목 선정
# ═══════════════════════════════════════════════════════════════════════
with tab_stock:
    st.header("🔍 종목 선정")

    sub_tabs = st.tabs([
        "📐 점수 산출 기준",
        "📊 최종 순위",
        "🎯 시나리오 분석",
        "📰 리서치/컨센서스",
        "🤖 ML 예측·검증",
        "🔎 종목 상세",
    ])

    # ── 서브탭 1: 점수 산출 기준 ────────────────────────────────────────
    with sub_tabs[0]:
        st.subheader("최종 AI 점수 구조")
        st.markdown("""
        ```
        최종 AI 점수 = 리서치 강화 점수 × 70%
                     + ML 상승 점수 × 20%
                     + 위험 방어 점수 × 10%

        리서치 강화 점수 = 퀀트 점수 × 70% + 리서치/컨센서스 점수 × 30%

        퀀트 점수 = 수익성(20%) + 밸류에이션(20%) + 모멘텀(25%)
                  + 안정성(20%) + 유동성(15%)
        ```
        """)

        score_table = pd.DataFrame([
            ("수익성 점수", "DART ROE(60%) + 영업이익률(40%) / proxy: 수익률/변동성",
             "자기자본 대비 이익 창출 능력. 높을수록 고점수"),
            ("밸류에이션 점수", "NAVER PER(60%) + PBR(40%) / proxy: 52주 가격 위치",
             "이익 대비 주가 수준. 저평가일수록 고점수"),
            ("모멘텀 점수", "3M·6M·12M 수익률",
             "최근 주가 추세. 상승 추세가 강할수록 고점수"),
            ("안정성 점수", "20일·60일 변동성, 1년 MDD",
             "변동성·낙폭이 낮을수록 고점수"),
            ("유동성 점수", "60일 평균 거래대금",
             "거래 가능성. 유동성이 높을수록 고점수"),
            ("시장 국면 반영", "KOSPI 모멘텀·변동성·MDD·200일선",
             "국면에 따라 팩터 가중치 동적 조정"),
            ("리서치/컨센서스 점수",
             "목표주가 상승여력(40%) + 투자의견(35%) + 리포트 수(10%) + 감성(15%)",
             "FnGuide 실제 증권사 컨센서스 (NAVER 금융)"),
            ("ML 상승 점수", "LogReg·RF·GBM·ExtraTrees 앙상블 상승확률 (4개 모델 평균)",
             "KOSPI 대비 초과수익 가능성"),
            ("위험 방어 점수", "ML 위험점수 (=1-상승확률)",
             "위험이 높은 종목 최종 점수 보수적 조정"),
        ], columns=["점수 구분", "주요 원천 지표", "해석"])
        st.dataframe(score_table, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("데이터 소스")
        source_table = pd.DataFrame([
            ("주가 데이터", "FinanceDataReader", "40개 KOSPI 종목 3년치 일별 가격"),
            ("KOSPI 벤치마크", "NAVER 금융 fchart API", "실제 지수 데이터"),
            ("PER/PBR/ROE/EPS", "NAVER 금융 종목 페이지 스크래핑", "실제 재무 데이터"),
            ("ROE/영업이익률/부채비율", "DART OpenAPI (사업보고서)", "2025년 연간보고서 기준"),
            ("목표주가/투자의견", "NAVER 금융 / FnGuide 컨센서스", "실제 애널리스트 컨센서스"),
        ], columns=["데이터 종류", "출처", "비고"])
        st.dataframe(source_table, use_container_width=True, hide_index=True)

    # ── 서브탭 2: 최종 순위 ────────────────────────────────────────────
    with sub_tabs[1]:
        final_df = safe_read("score_result_final_ai_by_model_all")
        if final_df.empty:
            st.warning("최종 AI 점수 데이터 없음. 파이프라인을 실행하세요.")
        else:
            col1, col2, col3 = st.columns(3)
            top_n = col1.slider("상위 N개 표시", 5, len(final_df), 15, key="top_n_slider")

            sort_options = {
                "최종 AI 점수": "final_ai_score",
                "퀀트 점수": "quant_score",
                "리서치 강화 점수": "research_enhanced_score",
                "모멘텀 점수": "momentum_score",
            }
            sort_label = col2.selectbox("정렬 기준", list(sort_options.keys()), key="sort_by_sel")
            sort_by = sort_options[sort_label]
            min_score = col3.slider("최소 AI 점수 필터", 0, 100, 0, key="min_score_sl")

            display_df = (
                final_df[final_df["final_ai_score"] >= min_score]
                .sort_values(sort_by, ascending=False)
                .head(top_n)
            )

            show_cols = ["final_ai_rank", "code", "name", "final_ai_score",
                         "quant_score", "research_enhanced_score", "ml_upside_prob",
                         "momentum_score", "stability_score", "liquidity_score"]
            show_cols = [c for c in show_cols if c in display_df.columns]

            _rank_renamed = display_df[show_cols].rename(columns={
                "final_ai_rank": "순위", "code": "종목코드", "name": "종목명",
                "final_ai_score": "최종AI점수", "quant_score": "퀀트점수",
                "research_enhanced_score": "리서치강화점수",
                "ml_upside_prob": "ML상승확률",
                "momentum_score": "모멘텀", "stability_score": "안정성",
                "liquidity_score": "유동성",
            })
            _sel_event = st.dataframe(
                _rank_renamed.style.background_gradient(subset=["최종AI점수"], cmap="Blues"),
                use_container_width=True, hide_index=True,
                on_select="rerun", selection_mode="single-row",
            )
            # 행 선택 시 종목 상세 탭으로 자동 이동
            if _sel_event and _sel_event.selection and _sel_event.selection.rows:
                _sel_row_idx = _sel_event.selection.rows[0]
                _sel_code = display_df.iloc[_sel_row_idx]["code"]
                _sel_name = display_df.iloc[_sel_row_idx]["name"]
                _detail_label = f"{_sel_name} ({_sel_code})"

                # rerun 시 테이블 선택이 유지되므로, 실제로 새 행을 클릭한 경우에만 덮어쓰기
                if st.session_state.get("_last_table_clicked_code") != _sel_code:
                    st.session_state["_last_table_clicked_code"] = _sel_code
                    st.session_state["detail_stock_sel"] = _detail_label

                # 새 종목을 선택한 경우에만 JS로 탭 이동 (루프 방지)
                if st.session_state.get("_last_nav_label") != _detail_label:
                    st.session_state["_last_nav_label"] = _detail_label
                    import streamlit.components.v1 as _stc
                    _stc.html("""
                    <script>
                    setTimeout(function() {
                        var btns = window.parent.document.querySelectorAll('button[data-testid="stTab"]');
                        for (var i = 0; i < btns.length; i++) {
                            if (btns[i].innerText.indexOf('종목 상세') !== -1) {
                                btns[i].click();
                                break;
                            }
                        }
                    }, 100);
                    </script>
                    """, height=0)

            fig = px.bar(
                display_df.sort_values("final_ai_rank"),
                x="name", y=["quant_score", "research_enhanced_score"],
                title="종목별 점수 구성",
                barmode="group",
                labels={"name": "종목명", "value": "점수", "variable": "구분"},
                color_discrete_map={
                    "quant_score": "#1976D2",
                    "research_enhanced_score": "#00897B",
                },
            )
            fig.update_layout(height=380, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # ── 서브탭 3: 시나리오 분석 ─────────────────────────────────────────
    with sub_tabs[2]:
        st.subheader("팩터 가중치 시나리오 분석")
        with st.expander("이 탭의 역할"):
            st.markdown("""
            **팩터 가중치를 바꾸면 종목 순위가 어떻게 달라지는가?** 를 즉시 확인하는 탭입니다.

            - 기본 AI 점수는 수익성·밸류에이션·모멘텀·안정성·유동성을 **고정 비율**로 합산합니다.
            - 이 탭에서 가중치를 조정하면 **"내가 모멘텀보다 가치를 더 중시한다면?"** 같은
              가상 시나리오에서 순위가 어떻게 바뀌는지 확인할 수 있습니다.
            - 프리셋(공격형/방어형/가치형/균형형)은 전형적인 투자 스타일을 미리 정의한 것입니다.
            - **순위 변화(↑양수=상승)** 컬럼으로 어떤 종목이 해당 스타일에서 유리한지 바로 파악할 수 있습니다.
            """)


        final_df_s = safe_read("score_result_final_ai_by_model_all")
        if final_df_s.empty:
            st.warning("데이터 없음")
        else:
            preset = st.selectbox("프리셋 선택", [
                "사용자 정의", "공격형/모멘텀", "방어형/저변동성", "가치형", "균형형"
            ], key="scenario_preset")

            presets = {
                "공격형/모멘텀":   {"수익성": 0.15, "밸류에이션": 0.10, "모멘텀": 0.45, "안정성": 0.10, "유동성": 0.20},
                "방어형/저변동성": {"수익성": 0.15, "밸류에이션": 0.20, "모멘텀": 0.10, "안정성": 0.40, "유동성": 0.15},
                "가치형":         {"수익성": 0.25, "밸류에이션": 0.40, "모멘텀": 0.10, "안정성": 0.15, "유동성": 0.10},
                "균형형":         {"수익성": 0.20, "밸류에이션": 0.20, "모멘텀": 0.25, "안정성": 0.20, "유동성": 0.15},
            }

            is_custom = (preset == "사용자 정의")
            pw = presets.get(preset, {"수익성": 0.20, "밸류에이션": 0.20, "모멘텀": 0.25, "안정성": 0.20, "유동성": 0.15})

            # 프리셋별로 키를 다르게 해서 슬라이더가 preset 변경 시 정확한 값으로 초기화되도록 함
            ks = preset.replace("/", "_").replace(" ", "_")

            col1, col2 = st.columns(2)
            if is_custom:
                with col1:
                    w_prof = st.slider("수익성 가중치",    0.0, 1.0, pw["수익성"],    0.05, key=f"w_prof_{ks}")
                    w_val  = st.slider("밸류에이션 가중치", 0.0, 1.0, pw["밸류에이션"], 0.05, key=f"w_val_{ks}")
                    w_mom  = st.slider("모멘텀 가중치",    0.0, 1.0, pw["모멘텀"],    0.05, key=f"w_mom_{ks}")
                with col2:
                    w_stab = st.slider("안정성 가중치",    0.0, 1.0, pw["안정성"],    0.05, key=f"w_stab_{ks}")
                    w_liq  = st.slider("유동성 가중치",    0.0, 1.0, pw["유동성"],    0.05, key=f"w_liq_{ks}")
            else:
                st.info(f"**{preset}** 프리셋은 가중치가 고정됩니다. 아래 슬라이더는 참고용 표시이며, 조정하려면 '사용자 정의'를 선택하세요.")
                with col1:
                    st.slider("수익성 가중치",    0.0, 1.0, pw["수익성"],    0.05, key=f"w_prof_{ks}", disabled=True)
                    st.slider("밸류에이션 가중치", 0.0, 1.0, pw["밸류에이션"], 0.05, key=f"w_val_{ks}",  disabled=True)
                    st.slider("모멘텀 가중치",    0.0, 1.0, pw["모멘텀"],    0.05, key=f"w_mom_{ks}",  disabled=True)
                with col2:
                    st.slider("안정성 가중치",    0.0, 1.0, pw["안정성"],    0.05, key=f"w_stab_{ks}", disabled=True)
                    st.slider("유동성 가중치",    0.0, 1.0, pw["유동성"],    0.05, key=f"w_liq_{ks}",  disabled=True)
                # 프리셋 고정값으로 직접 할당 (세션 상태 무시)
                w_prof = pw["수익성"]
                w_val  = pw["밸류에이션"]
                w_mom  = pw["모멘텀"]
                w_stab = pw["안정성"]
                w_liq  = pw["유동성"]

            total_w = w_prof + w_val + w_mom + w_stab + w_liq
            if total_w <= 0:
                st.error("가중치 합이 0입니다.")
            else:
                scenario_score = (
                    final_df_s.get("profitability_score", pd.Series(50, index=final_df_s.index)) * w_prof / total_w
                    + final_df_s.get("valuation_score",    pd.Series(50, index=final_df_s.index)) * w_val  / total_w
                    + final_df_s.get("momentum_score",     pd.Series(50, index=final_df_s.index)) * w_mom  / total_w
                    + final_df_s.get("stability_score",    pd.Series(50, index=final_df_s.index)) * w_stab / total_w
                    + final_df_s.get("liquidity_score",    pd.Series(50, index=final_df_s.index)) * w_liq  / total_w
                )
                tmp = final_df_s.copy()
                tmp["scenario_score"] = scenario_score
                tmp["scenario_rank"]  = scenario_score.rank(ascending=False).astype(int)
                tmp["rank_change"]    = tmp["final_ai_rank"] - tmp["scenario_rank"]

                _scen_top_n = st.session_state.get("top_n_slider", 15)
                top15 = tmp.nsmallest(_scen_top_n, "scenario_rank")[
                    ["scenario_rank", "code", "name", "scenario_score", "final_ai_rank", "rank_change"]
                ].rename(columns={
                    "scenario_rank": "시나리오 순위", "code": "종목코드", "name": "종목명",
                    "scenario_score": "시나리오 점수", "final_ai_rank": "기존 순위",
                    "rank_change": "순위 변화(↑양수=상승)",
                })
                st.dataframe(
                    top15.style.background_gradient(subset=["시나리오 점수"], cmap="Greens"),
                    use_container_width=True, hide_index=True,
                )
                st.caption(f"가중치 합계: {total_w:.2f} (자동 정규화 적용)")

    # ── 서브탭 4: 리서치/컨센서스 ─────────────────────────────────────
    with sub_tabs[3]:
        st.subheader("리서치/컨센서스 팩터")
        st.caption(
            "NAVER 금융 / FnGuide 실제 애널리스트 컨센서스 데이터 (data_type=real) · "
            "수집 실패 시 proxy(52주 고점 기반) 자동 적용"
        )

        cons_df = safe_read("research_consensus_factor_all")
        if cons_df.empty:
            st.warning("컨센서스 데이터 없음")
        else:
            real_count  = (cons_df.get("data_type", pd.Series()) == "real").sum()
            proxy_count = len(cons_df) - real_count
            c1, c2 = st.columns(2)
            c1.metric("실제 컨센서스", f"{real_count}개 종목")
            c2.metric("Proxy 사용", f"{proxy_count}개 종목")

            show_cols = [c for c in ["code", "name", "current_price", "target_price",
                                      "upside_ratio", "opinion", "opinion_score",
                                      "report_count", "sentiment_score",
                                      "research_score", "data_type"] if c in cons_df.columns]
            st.dataframe(
                cons_df[show_cols].rename(columns={
                    "code": "종목코드", "name": "종목명",
                    "current_price": "현재가", "target_price": "목표주가",
                    "upside_ratio": "상승여력(%)", "opinion": "투자의견",
                    "opinion_score": "의견점수(1~5)", "report_count": "리포트 수",
                    "sentiment_score": "감성점수", "research_score": "리서치점수",
                    "data_type": "데이터유형",
                }).style.background_gradient(subset=["리서치점수"], cmap="YlOrRd"),
                use_container_width=True, hide_index=True,
            )

            st.divider()
            st.subheader("실제 컨센서스 데이터 직접 업로드 (선택)")
            uploaded = st.file_uploader(
                "CSV/Excel 업로드 (컬럼: code, name, current_price, target_price, opinion, report_count, sentiment_score)",
                type=["csv", "xlsx"], key="consensus_upload",
            )
            if uploaded:
                try:
                    udf = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                    udf["data_type"] = "uploaded"
                    con = get_connection()
                    udf.to_sql("research_consensus_factor_uploaded", con, if_exists="replace", index=False)
                    st.success(f"업로드 완료: {len(udf)}개 종목 저장")
                    st.dataframe(udf.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"업로드 오류: {e}")

    # ── 서브탭 5: ML 예측·검증 ────────────────────────────────────────
    with sub_tabs[4]:
        st.subheader("ML 예측·검증")

        with st.expander("모델 선택 근거 및 주식 데이터 적합성"):
            st.markdown("""
            #### 사용 모델 선택 근거

            주가 데이터는 **노이즈가 많고**, **샘플 수가 제한적**이며, **비선형 패턴**이 존재합니다.
            이런 특성에 맞춰 서로 다른 귀납적 편향(inductive bias)을 가진 4개 모델을 앙상블합니다.

            | 모델 | 특성 | 선택 이유 |
            |------|------|----------|
            | **LogisticRegression** | 선형 모델, L2 정규화 | 과적합 위험 최소화. 가격·모멘텀의 선형 관계를 빠르게 포착. 앙상블의 편향 방어막 역할 |
            | **RandomForest** | 배깅 기반 결정트리 | 비선형 패턴 포착, 특성 중요도 제공, 분산 감소 효과. 개별 트리 노이즈를 평균화 |
            | **GradientBoosting** | 순차적 부스팅 | 이전 모델의 잔차를 순차적으로 보정. 복잡한 비선형 패턴에 강함. 단, 과적합 주의 |
            | **ExtraTrees** | 극도 무작위화 트리 | RF보다 분기점을 더 무작위로 선택 → 주가 노이즈에 더 강건. 학습 속도 빠름 |

            #### 앙상블 효과
            - 4개 모델 상승확률을 **단순 평균**하여 최종 ML 점수 산출
            - 개별 모델의 편향/분산을 상호 보완
            - 최종 AI 점수에서 **ML 상승 점수 20% + 위험 방어 점수 10%** 반영

            #### 현 시스템 한계 및 개선 방향
            - 학습 데이터: 최근 3년 일별 가격·거래량 기반 기술적 지표
            - 종목당 샘플 ~600개(거래일), train 80% / test 20% 시계열 분할
            - 향후 고려 가능 모델: XGBoost(별도 패키지), LSTM(시계열 딥러닝), 멀티팩터 회귀
            """)

        ml_df = safe_read("ml_prediction_score_by_model_all")
        if ml_df.empty:
            st.warning("ML 결과 없음")
        else:
            models_available = ml_df["model"].unique().tolist()
            selected_model = st.selectbox("모델 선택", ["전체 평균"] + models_available, key="ml_model_sel")

            if selected_model == "전체 평균":
                disp = ml_df.groupby("code").agg(
                    upside_prob=("upside_prob", "mean"),
                    accuracy=("accuracy", "mean"),
                    roc_auc=("roc_auc", "mean"),
                ).reset_index()
                disp["model"] = "전체 평균"
            else:
                disp = ml_df[ml_df["model"] == selected_model].copy()

            final_df_ml = safe_read("score_result_final_ai_by_model_all")
            if not final_df_ml.empty:
                disp = disp.merge(final_df_ml[["code", "name"]].drop_duplicates(), on="code", how="left")

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(
                    disp[["code", "name", "upside_prob", "accuracy", "roc_auc"]].rename(columns={
                        "code": "종목코드", "name": "종목명",
                        "upside_prob": "상승확률", "accuracy": "정확도", "roc_auc": "ROC-AUC",
                    }).sort_values("상승확률", ascending=False).style.background_gradient(
                        subset=["상승확률"], cmap="RdYlGn"
                    ),
                    use_container_width=True, hide_index=True,
                )
            with col2:
                fig_ml = px.histogram(
                    disp, x="upside_prob", nbins=15,
                    title="상승확률 분포",
                    labels={"upside_prob": "상승확률"},
                    color_discrete_sequence=["#1976D2"],
                )
                fig_ml.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="기준선 0.5")
                fig_ml.update_layout(height=350, margin=dict(t=40, b=0))
                st.plotly_chart(fig_ml, use_container_width=True)

            st.divider()
            st.subheader("모델별 성능 비교")
            perf = ml_df.groupby("model").agg(
                avg_accuracy=("accuracy", "mean"),
                avg_roc_auc=("roc_auc", "mean"),
                avg_upside_prob=("upside_prob", "mean"),
            ).reset_index()
            st.dataframe(perf.rename(columns={
                "model": "모델", "avg_accuracy": "평균 정확도",
                "avg_roc_auc": "평균 ROC-AUC", "avg_upside_prob": "평균 상승확률",
            }), use_container_width=True, hide_index=True)

    # ── 서브탭 6: 종목 상세 드릴다운 ─────────────────────────────────
    with sub_tabs[5]:
        st.subheader("종목 상세 분석")

        detail_final_df = safe_read("score_result_final_ai_by_model_all")
        detail_price_df = safe_read("price_data")
        detail_ml_df    = safe_read("ml_prediction_score_by_model_all")
        detail_cons_df  = safe_read("research_consensus_factor_all")

        if detail_final_df.empty:
            st.warning("데이터 없음. 파이프라인을 실행하세요.")
        else:
            name_code_map = {
                f"{row['name']} ({row['code']})": row["code"]
                for _, row in detail_final_df.sort_values("final_ai_rank").iterrows()
            }
            sel_label = st.selectbox("종목 선택", list(name_code_map.keys()), key="detail_stock_sel")
            sel_code  = name_code_map[sel_label]
            row_d = detail_final_df[detail_final_df["code"] == sel_code].iloc[0]

            # 상단 핵심 지표
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("AI 순위",    f"{int(row_d.get('final_ai_rank', 0))}위")
            d2.metric("최종 AI 점수", f"{row_d.get('final_ai_score', 0):.1f}")
            d3.metric("퀀트 점수",   f"{row_d.get('quant_score', 0):.1f}")
            d4.metric("ML 상승확률", f"{row_d.get('ml_upside_prob', 0.5)*100:.1f}%")
            cons_row_d = detail_cons_df[detail_cons_df["code"] == sel_code] if not detail_cons_df.empty else pd.DataFrame()
            upside_d = cons_row_d.iloc[0].get("upside_ratio", 0) if not cons_row_d.empty else 0
            d5.metric("목표가 상승여력", f"{upside_d:.1f}%")

            st.divider()
            col_chart, col_radar = st.columns([3, 2])

            # ── 주가 차트 ──────────────────────────────────────────────
            with col_chart:
                period_sel = st.radio("기간", ["1년", "3년"], horizontal=True, key="detail_period")
                if not detail_price_df.empty:
                    pf = detail_price_df[detail_price_df["code"] == sel_code].copy()
                    pf["date"] = pd.to_datetime(pf["date"])
                    pf = pf.sort_values("date")
                    if period_sel == "1년":
                        cutoff = pf["date"].max() - pd.DateOffset(years=1)
                        pf = pf[pf["date"] >= cutoff]
                    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                              row_heights=[0.75, 0.25], vertical_spacing=0.03)
                    fig_price.add_trace(
                        go.Scatter(x=pf["date"], y=pf["close"],
                                   name="종가", line=dict(color="#1976D2", width=1.5)),
                        row=1, col=1
                    )
                    ma20 = pf["close"].rolling(20).mean()
                    ma60 = pf["close"].rolling(60).mean()
                    fig_price.add_trace(go.Scatter(x=pf["date"], y=ma20, name="MA20",
                                                   line=dict(color="orange", width=1, dash="dot")), row=1, col=1)
                    fig_price.add_trace(go.Scatter(x=pf["date"], y=ma60, name="MA60",
                                                   line=dict(color="green", width=1, dash="dot")), row=1, col=1)
                    fig_price.add_trace(
                        go.Bar(x=pf["date"], y=pf["volume"], name="거래량",
                               marker_color="#90CAF9", opacity=0.6),
                        row=2, col=1
                    )
                    fig_price.update_layout(
                        title=f"{row_d.get('name', sel_code)} 주가 ({period_sel})",
                        height=400, margin=dict(t=40, b=0), showlegend=True,
                    )
                    st.plotly_chart(fig_price, use_container_width=True)

            # ── 팩터 레이더 차트 ───────────────────────────────────────
            with col_radar:
                score_cols = ["profitability_score", "valuation_score", "momentum_score",
                              "stability_score", "liquidity_score"]
                score_labels = ["수익성", "밸류에이션", "모멘텀", "안정성", "유동성"]
                radar_vals = [float(row_d.get(c, 50)) for c in score_cols]
                fig_radar = go.Figure(go.Scatterpolar(
                    r=radar_vals + [radar_vals[0]],
                    theta=score_labels + [score_labels[0]],
                    fill="toself",
                    line=dict(color="#1976D2"),
                    fillcolor="rgba(25,118,210,0.2)",
                    name=row_d.get("name", sel_code),
                ))
                avg_vals = [float(detail_final_df[c].mean()) for c in score_cols]
                fig_radar.add_trace(go.Scatterpolar(
                    r=avg_vals + [avg_vals[0]],
                    theta=score_labels + [score_labels[0]],
                    line=dict(color="gray", dash="dot"),
                    name="전체 평균",
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="팩터 점수 레이더",
                    height=380, margin=dict(t=50, b=10),
                    showlegend=True,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # ── 재무 핵심 지표 ─────────────────────────────────────────
            st.divider()
            col_fin, col_ml = st.columns(2)
            with col_fin:
                st.markdown("**재무 핵심 지표**")
                fin_rows = [
                    ("PER",           row_d.get("PER",  "N/A")),
                    ("PBR",           row_d.get("PBR",  "N/A")),
                    ("ROE",           row_d.get("ROE",  "N/A")),
                    ("DART ROE",      row_d.get("dart_roe", "N/A")),
                    ("영업이익률",     row_d.get("dart_opm",  "N/A")),
                    ("부채비율",       row_d.get("dart_debt", "N/A")),
                    ("3M 수익률",     f"{row_d.get('ret_3m',  0):.2f}%"),
                    ("6M 수익률",     f"{row_d.get('ret_6m',  0):.2f}%"),
                    ("12M 수익률",    f"{row_d.get('ret_12m', 0):.2f}%"),
                    ("20일 변동성",   f"{row_d.get('vol_20',  0):.2f}%"),
                    ("1년 MDD",       f"{row_d.get('mdd_1y',  0):.2f}%"),
                ]
                st.dataframe(
                    pd.DataFrame(fin_rows, columns=["지표", "값"]),
                    use_container_width=True, hide_index=True,
                )

            # ── ML 모델별 상승확률 ─────────────────────────────────────
            with col_ml:
                st.markdown("**ML 모델별 상승확률**")
                if not detail_ml_df.empty:
                    ml_stock = detail_ml_df[detail_ml_df["code"] == sel_code].copy()
                    if not ml_stock.empty:
                        fig_ml_bar = px.bar(
                            ml_stock.sort_values("upside_prob", ascending=False),
                            x="model", y="upside_prob",
                            color="upside_prob",
                            color_continuous_scale="RdYlGn",
                            range_color=[0, 1],
                            labels={"model": "모델", "upside_prob": "상승확률"},
                            title="모델별 상승확률",
                        )
                        fig_ml_bar.add_hline(y=0.5, line_dash="dash", line_color="red",
                                              annotation_text="기준선 0.5")
                        fig_ml_bar.update_layout(height=320, margin=dict(t=40, b=0),
                                                  showlegend=False)
                        st.plotly_chart(fig_ml_bar, use_container_width=True)

                        ml_perf = ml_stock[["model", "accuracy", "roc_auc"]].rename(columns={
                            "model": "모델", "accuracy": "정확도", "roc_auc": "ROC-AUC"
                        })
                        st.dataframe(ml_perf, use_container_width=True, hide_index=True)

            # ── 컨센서스 요약 ──────────────────────────────────────────
            if not cons_row_d.empty:
                st.divider()
                cr = cons_row_d.iloc[0]
                st.markdown("**애널리스트 컨센서스**")
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("현재가",    f"{int(cr.get('current_price', 0)):,}원")
                cc2.metric("목표주가",  f"{int(cr.get('target_price', 0)):,}원")
                cc3.metric("투자의견",  str(cr.get("opinion", "N/A")))
                cc4.metric("리포트 수", f"{int(cr.get('report_count', 0))}건")
                data_badge = "✅ 실제" if cr.get("data_type") == "real" else "⚠️ Proxy"
                st.caption(f"데이터 유형: {data_badge}")



# ═══════════════════════════════════════════════════════════════════════
# TAB 4: 포트폴리오
# ═══════════════════════════════════════════════════════════════════════
with tab_portfolio:
    st.header("💼 포트폴리오 분석")

    port_tabs = st.tabs(["📈 추천 포트폴리오", "🏥 보유 포트폴리오 진단"])

    # ── 포트 서브탭 1: 추천 포트폴리오 ────────────────────────────────
    with port_tabs[0]:
        selected_top_n = st.session_state.get("top_n_slider", 15)
        st.info(
            f"'종목 선정 → 최종 순위' 탭에서 선택한 상위 **{selected_top_n}개** 종목을 기반으로 포트폴리오를 구성합니다. "
            f"종목 수를 바꾸려면 해당 탭의 슬라이더를 조정하세요."
        )

        final_df_p = safe_read("score_result_final_ai_by_model_all")
        price_df_p = safe_read("price_data")

        if final_df_p.empty:
            st.warning("최종 AI 점수 데이터 없음. 파이프라인을 실행하세요.")
        else:
            top_stocks = final_df_p.nsmallest(selected_top_n, "final_ai_rank")

            st.markdown(f"#### 선정 종목 (AI 점수 상위 {selected_top_n}개)")
            show_port_cols = [c for c in ["final_ai_rank", "code", "name", "final_ai_score",
                                           "quant_score", "momentum_score", "stability_score"]
                              if c in top_stocks.columns]
            st.dataframe(
                top_stocks[show_port_cols].rename(columns={
                    "final_ai_rank": "순위", "code": "종목코드", "name": "종목명",
                    "final_ai_score": "AI점수", "quant_score": "퀀트점수",
                    "momentum_score": "모멘텀", "stability_score": "안정성",
                }).style.background_gradient(subset=["AI점수"], cmap="Blues"),
                use_container_width=True, hide_index=True,
            )

            st.divider()

            # ── 선정 종목 기반 실시간 전략 계산 ──────────────────────────
            st.subheader(f"전략별 성과 비교 (선정 {selected_top_n}개 종목 기준)")

            _live_perf_df = pd.DataFrame()
            _live_weight_df = pd.DataFrame()
            _live_risk_df = pd.DataFrame()
            _live_cum = {}

            if not price_df_p.empty and not top_stocks.empty:
                from scipy.optimize import minimize as _spmin

                _ts_codes = top_stocks["code"].tolist()
                _ts_names = dict(zip(top_stocks["code"], top_stocks["name"]))
                _ts_ai    = dict(zip(top_stocks["code"],
                                     top_stocks.get("final_ai_score", pd.Series(50, index=top_stocks.index))))

                price_df_p["date"] = pd.to_datetime(price_df_p["date"])
                _pivot_live = price_df_p[price_df_p["code"].isin(_ts_codes)].pivot(
                    index="date", columns="code", values="close"
                ).sort_index().ffill()
                _avail = [c for c in _ts_codes if c in _pivot_live.columns]

                if len(_avail) >= 2:
                    _rets  = _pivot_live[_avail].pct_change().dropna()
                    _n     = len(_avail)
                    _cov   = _rets.cov().values * 252
                    _mu    = _rets.mean().values * 252
                    _bnd   = [(0.0, 1.0)] * _n
                    _ceq   = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
                    _x0    = np.ones(_n) / _n

                    _live_w = {}
                    _live_w["동일비중"] = np.ones(_n) / _n

                    _r1 = _spmin(lambda w: w @ _cov @ w, _x0, bounds=_bnd, constraints=_ceq, method="SLSQP")
                    _wmv = _r1.x if _r1.success else _x0.copy()
                    _wmv[_wmv < 0.005] = 0; _wmv /= _wmv.sum()
                    _live_w["최소분산"] = _wmv

                    def _neg_sh(w): return -(w @ _mu) / (np.sqrt(w @ _cov @ w) + 1e-9)
                    _r2 = _spmin(_neg_sh, _x0, bounds=_bnd, constraints=_ceq, method="SLSQP")
                    _wms = _r2.x if _r2.success else _x0.copy()
                    _wms[_wms < 0.005] = 0; _wms /= _wms.sum()
                    _live_w["최대샤프"] = _wms

                    def _rp_obj(w):
                        _pv = w @ _cov @ w
                        _rc = w * (_cov @ w) / (np.sqrt(_pv) + 1e-9)
                        return np.sum((_rc - _rc.mean()) ** 2)
                    _r3 = _spmin(_rp_obj, _x0, bounds=_bnd, constraints=_ceq, method="SLSQP")
                    _wrp = _r3.x if _r3.success else _x0.copy()
                    _wrp[_wrp < 0.001] = 0; _wrp /= _wrp.sum()
                    _live_w["Risk Parity"] = _wrp

                    _wai = np.array([_ts_ai.get(c, 50) for c in _avail], dtype=float)
                    _wai /= _wai.sum()
                    _live_w["AI점수가중"] = _wai

                    _perf_rows, _wt_rows, _rk_rows = [], [], []
                    for _sn, _w in _live_w.items():
                        _pr  = (_rets * _w).sum(axis=1)
                        _cum = (1 + _pr).cumprod()
                        _live_cum[_sn] = _cum
                        _ar  = _pr.mean() * 252 * 100
                        _av  = _pr.std()  * np.sqrt(252) * 100
                        _sh  = _ar / (_av + 1e-9)
                        _mdd = ((_cum - _cum.cummax()) / _cum.cummax()).min() * 100
                        _perf_rows.append({"strategy": _sn, "ann_return": round(_ar, 2),
                                           "ann_vol": round(_av, 2), "sharpe": round(_sh, 3),
                                           "mdd": round(_mdd, 2)})
                        _pv  = _w @ _cov @ _w
                        _rc  = _w * (_cov @ _w) / (np.sqrt(_pv) + 1e-9)
                        _rcp = _rc / (_rc.sum() + 1e-9) * 100
                        for _i, _c in enumerate(_avail):
                            if _w[_i] > 0.001:
                                _wt_rows.append({"strategy": _sn, "code": _c,
                                                 "name": _ts_names.get(_c, _c), "weight": round(_w[_i], 4)})
                                _rk_rows.append({"strategy": _sn, "code": _c,
                                                 "name": _ts_names.get(_c, _c), "risk_contrib_pct": round(_rcp[_i], 2)})

                    _live_perf_df  = pd.DataFrame(_perf_rows)
                    _live_weight_df = pd.DataFrame(_wt_rows)
                    _live_risk_df  = pd.DataFrame(_rk_rows)

            if not _live_perf_df.empty:
                bm_df_p = safe_read("benchmark_price")
                if not bm_df_p.empty:
                    bm_df_p["date"] = pd.to_datetime(bm_df_p["date"])
                    bm_close = bm_df_p.set_index("date").sort_index()["close"]
                    bm_ret   = bm_close.pct_change().dropna()
                    bm_ann   = bm_ret.mean() * 252 * 100
                    bm_vol   = bm_ret.std() * np.sqrt(252) * 100
                    bm_sharpe = bm_ann / (bm_vol + 1e-9)
                    cum_bm   = (1 + bm_ret).cumprod()
                    bm_mdd   = ((cum_bm - cum_bm.cummax()) / cum_bm.cummax()).min() * 100
                    _live_cum["KOSPI(벤치마크)"] = cum_bm
                    _bm_row = pd.DataFrame([{"strategy": "KOSPI(벤치마크)",
                                             "ann_return": round(bm_ann, 2), "ann_vol": round(bm_vol, 2),
                                             "sharpe": round(bm_sharpe, 3), "mdd": round(bm_mdd, 2)}])
                    display_perf = pd.concat([_live_perf_df, _bm_row], ignore_index=True)
                else:
                    display_perf = _live_perf_df.copy()

                st.dataframe(
                    display_perf.rename(columns={
                        "strategy": "전략", "ann_return": "연수익률(%)",
                        "ann_vol": "연변동성(%)", "sharpe": "Sharpe", "mdd": "MDD(%)",
                    }).style.background_gradient(subset=["연수익률(%)"], cmap="Blues"),
                    use_container_width=True, hide_index=True,
                )

                # ── 시장 국면 vs 전략 해석 안내 ───────────────────────────
                regime_df_p = safe_read("market_regime")
                if not regime_df_p.empty:
                    rp = regime_df_p.iloc[0]
                    rp_name = rp.get("regime", "N/A")
                    rp_rec  = rp.get("strategy_rec", "")
                    with st.expander("시장 국면 추천 전략 vs 백테스트 결과 해석", expanded=False):
                        st.markdown(f"""
                        **현재 국면**: {rp_name} → 추천: {rp_rec}

                        > 백테스트에서 **최대 샤프 전략**이 가장 높은 수익률을 보이더라도,
                        > 이는 **과거 3년 데이터에 최적화된 결과**입니다 (in-sample fit).
                        >
                        > **시장 국면 추천 전략**은 *현재의 시장 환경*에서 향후 위험을 관리하기 위한
                        > 전방위(forward-looking) 권고입니다. 두 가지는 서로 다른 목적입니다:
                        >
                        > | 구분 | 목적 | 시점 |
                        > |------|------|------|
                        > | 백테스트 최적 전략 | 과거 성과 최대화 | 후향적 (backward-looking) |
                        > | 국면 추천 전략 | 현재 위험 관리 | 전향적 (forward-looking) |
                        >
                        > **{rp_name}** 국면에서는 {rp_rec.split('(')[0].strip()}가 적합합니다.
                        > 과거 최대 샤프 전략을 그대로 따르면 현재 시장 상황에 맞지 않는 팩터에
                        > 과도하게 노출될 수 있습니다.
                        """)

                _strat_options = _live_perf_df["strategy"].tolist()
                selected_strat = st.selectbox("전략 선택 (비중 및 수익률 확인)", _strat_options, key="strat_sel_pie")

                _strat_w = _live_weight_df[_live_weight_df["strategy"] == selected_strat]
                _strat_r = _live_risk_df[_live_risk_df["strategy"] == selected_strat]
                if not _strat_w.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_pie = px.pie(_strat_w, values="weight", names="name",
                                         title=f"{selected_strat} 종목 비중")
                        fig_pie.update_layout(height=380)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        if not _strat_r.empty:
                            fig_risk = px.bar(
                                _strat_r.sort_values("risk_contrib_pct", ascending=False),
                                x="name", y="risk_contrib_pct",
                                title=f"{selected_strat} 위험기여도(%)",
                                color="risk_contrib_pct", color_continuous_scale="Reds",
                            )
                            fig_risk.update_layout(height=380, margin=dict(t=40, b=0))
                            st.plotly_chart(fig_risk, use_container_width=True)
                    n_used  = len(_strat_w)
                    if n_used < selected_top_n:
                        st.info(f"이 전략은 선정 {selected_top_n}개 종목 중 **{n_used}개**를 사용합니다. "
                                f"최적화 과정에서 {selected_top_n - n_used}개 종목의 비중이 0으로 설정되었습니다.")

                # ── 선택 전략 누적 수익률 vs KOSPI ─────────────────────
                st.divider()
                st.subheader(f"{selected_strat} 전략 누적 수익률 vs KOSPI")
                if selected_strat in _live_cum:
                    _cum_sel = _live_cum[selected_strat]
                    _pr_sel  = _cum_sel.pct_change().fillna(0)
                    _ar_sel  = _pr_sel.mean() * 252 * 100
                    _av_sel  = _pr_sel.std()  * np.sqrt(252) * 100
                    _sh_sel  = _ar_sel / (_av_sel + 1e-9)
                    _md_sel  = ((_cum_sel - _cum_sel.cummax()) / _cum_sel.cummax()).min() * 100
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("연수익률", f"{_ar_sel:.2f}%")
                    c2.metric("연변동성", f"{_av_sel:.2f}%")
                    c3.metric("Sharpe",  f"{_sh_sel:.3f}")
                    c4.metric("MDD",     f"{_md_sel:.2f}%")

                    fig_sw = go.Figure()
                    fig_sw.add_trace(go.Scatter(x=_cum_sel.index, y=_cum_sel.values,
                                                name=selected_strat, line=dict(color="#1976D2")))
                    if "KOSPI(벤치마크)" in _live_cum:
                        _bm_c = _live_cum["KOSPI(벤치마크)"].reindex(_cum_sel.index).ffill()
                        fig_sw.add_trace(go.Scatter(x=_bm_c.index, y=_bm_c.values,
                                                    name="KOSPI", line=dict(color="orange", dash="dash")))
                    fig_sw.update_layout(title=f"{selected_strat} vs KOSPI 누적 수익률",
                                         height=350, margin=dict(t=40, b=0))
                    st.plotly_chart(fig_sw, use_container_width=True)

            # ── 상관관계 히트맵 ───────────────────────────────────────
            st.divider()
            st.subheader("종목 간 상관관계 히트맵")
            st.caption("선정된 종목들의 일별 수익률 기반 상관계수 (-1 ~ +1). 붉을수록 동조, 파랄수록 역행.")
            if not price_df_p.empty and not top_stocks.empty:
                corr_codes = top_stocks["code"].tolist()
                corr_names = dict(zip(top_stocks["code"], top_stocks["name"]))
                price_df_p["date"] = pd.to_datetime(price_df_p["date"])
                pivot_corr = price_df_p[price_df_p["code"].isin(corr_codes)].pivot(
                    index="date", columns="code", values="close"
                ).sort_index().ffill()
                avail_codes = [c for c in corr_codes if c in pivot_corr.columns]
                if len(avail_codes) >= 2:
                    ret_corr = pivot_corr[avail_codes].pct_change().dropna()
                    corr_mat = ret_corr.corr()
                    corr_mat.index   = [corr_names.get(c, c) for c in corr_mat.index]
                    corr_mat.columns = [corr_names.get(c, c) for c in corr_mat.columns]

                    fig_hm = px.imshow(
                        corr_mat,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        text_auto=".2f",
                        title=f"종목 수익률 상관관계 (선정 {len(avail_codes)}개 종목)",
                    )
                    fig_hm.update_layout(
                        height=max(400, len(avail_codes) * 30 + 80),
                        margin=dict(t=50, b=0),
                        coloraxis_colorbar=dict(title="상관계수"),
                    )
                    fig_hm.update_traces(textfont_size=9)
                    st.plotly_chart(fig_hm, use_container_width=True)

                    avg_off_diag = corr_mat.values[np.triu_indices_from(corr_mat.values, k=1)].mean()
                    if avg_off_diag > 0.7:
                        st.warning(f"평균 상관계수 {avg_off_diag:.2f} — 종목 간 동조화가 강해 분산 효과가 제한적입니다.")
                    elif avg_off_diag < 0.3:
                        st.success(f"평균 상관계수 {avg_off_diag:.2f} — 분산 효과가 우수한 포트폴리오입니다.")
                    else:
                        st.info(f"평균 상관계수 {avg_off_diag:.2f} — 적절한 분산 수준입니다.")

            # ── 인터랙티브 포트폴리오 최적화 ─────────────────────────────
            st.divider()
            st.subheader("인터랙티브 포트폴리오 최적화")
            st.caption(
                "선정된 종목으로 직접 최적화를 실행합니다. "
                "'전체 종목 강제 포함'은 모든 종목에 최소 비중을 부여하고, "
                "'최적 서브셋'은 최적화가 일부 종목 비중을 0으로 설정할 수 있습니다."
            )

            opt_col1, opt_col2 = st.columns(2)
            opt_method   = opt_col1.selectbox(
                "최적화 전략", ["최소분산 (Min Variance)", "최대샤프 (Max Sharpe)", "동일비중 (Equal Weight)"],
                key="opt_method_sel"
            )
            force_all    = opt_col2.radio(
                "종목 포함 방식",
                ["전체 종목 강제 포함 (최소 비중 2%)", "최적 서브셋 자동 선택 (비중 0 허용)"],
                key="force_all_radio"
            ) == "전체 종목 강제 포함 (최소 비중 2%)"

            if st.button("최적화 실행", key="opt_run_btn"):
                codes_top = top_stocks["code"].tolist()
                if not price_df_p.empty and len(codes_top) >= 2:
                    price_df_p["date"] = pd.to_datetime(price_df_p["date"])
                    pivot_opt = price_df_p[price_df_p["code"].isin(codes_top)].pivot(
                        index="date", columns="code", values="close"
                    ).sort_index().ffill()
                    opt_codes = [c for c in codes_top if c in pivot_opt.columns]
                    name_map  = dict(zip(top_stocks["code"], top_stocks["name"]))

                    if len(opt_codes) >= 2:
                        rets_opt = pivot_opt[opt_codes].pct_change().dropna()
                        n_opt    = len(opt_codes)
                        cov_opt  = rets_opt.cov().values * 252
                        mu_opt   = rets_opt.mean().values * 252
                        lb_opt   = 0.02 if force_all else 0.0

                        try:
                            from scipy.optimize import minimize as _minimize

                            bounds_opt = [(lb_opt, 1.0)] * n_opt
                            cons_opt   = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
                            x0_opt     = np.ones(n_opt) / n_opt

                            if "최소분산" in opt_method:
                                def _obj(w): return w @ cov_opt @ w
                            elif "최대샤프" in opt_method:
                                def _obj(w):
                                    r = w @ mu_opt
                                    v = np.sqrt(w @ cov_opt @ w)
                                    return -(r / (v + 1e-9))
                            else:
                                opt_w = np.ones(n_opt) / n_opt
                                _obj  = None

                            if _obj is not None:
                                res = _minimize(_obj, x0_opt, bounds=bounds_opt,
                                                constraints=cons_opt, method="SLSQP")
                                opt_w = res.x if res.success else x0_opt
                                if not force_all:
                                    opt_w[opt_w < 0.01] = 0
                                if opt_w.sum() > 0:
                                    opt_w = opt_w / opt_w.sum()

                            used_idx   = [i for i, w in enumerate(opt_w) if w > 0.001]
                            used_codes = [opt_codes[i] for i in used_idx]
                            used_w     = opt_w[used_idx]

                            st.markdown(f"**사용 종목: {len(used_codes)}/{n_opt}개** (비중 0.1% 이상 기준)")
                            opt_result_df = pd.DataFrame({
                                "종목코드": used_codes,
                                "종목명":  [name_map.get(c, c) for c in used_codes],
                                "비중(%)": (used_w * 100).round(2),
                            }).sort_values("비중(%)", ascending=False)
                            st.dataframe(opt_result_df, use_container_width=True, hide_index=True)

                            w_full  = np.zeros(len(opt_codes))
                            for i, c in enumerate(opt_codes):
                                if c in used_codes:
                                    w_full[i] = used_w[list(used_codes).index(c)]
                            port_opt_ret = (rets_opt * w_full).sum(axis=1)
                            cum_opt      = (1 + port_opt_ret).cumprod()
                            ann_ret_o    = port_opt_ret.mean() * 252 * 100
                            ann_vol_o    = port_opt_ret.std() * np.sqrt(252) * 100
                            sharpe_o     = ann_ret_o / (ann_vol_o + 1e-9)
                            mdd_o        = ((cum_opt - cum_opt.cummax()) / cum_opt.cummax()).min() * 100

                            oc1, oc2, oc3, oc4 = st.columns(4)
                            oc1.metric("연수익률", f"{ann_ret_o:.2f}%")
                            oc2.metric("연변동성", f"{ann_vol_o:.2f}%")
                            oc3.metric("Sharpe",  f"{sharpe_o:.3f}")
                            oc4.metric("MDD",     f"{mdd_o:.2f}%")

                            fig_opt = go.Figure()
                            fig_opt.add_trace(go.Scatter(
                                x=cum_opt.index, y=cum_opt.values,
                                name=opt_method.split(" ")[0], line=dict(color="#7B1FA2")
                            ))
                            bm_opt = safe_read("benchmark_price")
                            if not bm_opt.empty:
                                bm_opt["date"] = pd.to_datetime(bm_opt["date"])
                                bm_co = bm_opt.set_index("date")["close"].reindex(cum_opt.index).ffill()
                                bm_co_cum = (1 + bm_co.pct_change().fillna(0)).cumprod()
                                fig_opt.add_trace(go.Scatter(
                                    x=bm_co_cum.index, y=bm_co_cum.values,
                                    name="KOSPI", line=dict(color="orange", dash="dash")
                                ))
                            fig_opt.update_layout(
                                title=f"{opt_method} 최적화 누적 수익률 vs KOSPI",
                                height=350, margin=dict(t=40, b=0)
                            )
                            st.plotly_chart(fig_opt, use_container_width=True)

                        except Exception as e:
                            st.error(f"최적화 오류: {e}")

    # ── 포트 서브탭 2: 보유 포트폴리오 진단 ───────────────────────────
    with port_tabs[1]:
        st.subheader("보유 포트폴리오 진단")
        st.caption("종목코드 또는 종목명, 비중(%)을 한 줄에 하나씩 입력하세요. CASH 또는 현금 사용 가능.")

        sample = "삼성전자,30\nSK하이닉스,20\nNAVER,20\nCASH,30"
        user_input = st.text_area("보유 포트폴리오 입력", sample, height=150, key="user_port_input")

        if st.button("진단 실행", key="diag_btn"):
            master_df   = safe_read("stock_master")
            final_df_d  = safe_read("score_result_final_ai_by_model_all")
            price_df_d  = safe_read("price_data")
            cons_df_d   = safe_read("research_consensus_factor_all")

            # 국면 정보
            regime_df_diag = safe_read("market_regime")
            regime_name_diag = regime_df_diag.iloc[0].get("regime", "N/A") if not regime_df_diag.empty else "N/A"

            lines = [l.strip() for l in user_input.strip().split("\n") if l.strip()]
            parsed = []
            errors = []
            for line in lines:
                parts = line.split(",")
                if len(parts) != 2:
                    errors.append(f"형식 오류: {line}")
                    continue
                identifier, weight_str = parts[0].strip(), parts[1].strip()
                try:
                    weight = float(weight_str)
                except ValueError:
                    errors.append(f"비중 오류: {line}")
                    continue

                if identifier.upper() in ("CASH", "현금"):
                    parsed.append({"code": "CASH", "name": "현금", "weight": weight})
                    continue

                matched = None
                if not master_df.empty:
                    by_code = master_df[master_df["code"] == identifier]
                    by_name = master_df[master_df["name"] == identifier]
                    if not by_code.empty:
                        matched = by_code.iloc[0]
                    elif not by_name.empty:
                        matched = by_name.iloc[0]

                parsed.append({
                    "code": matched["code"] if matched is not None else identifier,
                    "name": matched["name"] if matched is not None else identifier,
                    "weight": weight,
                })

            for e in errors:
                st.error(e)

            if parsed:
                port_df_user = pd.DataFrame(parsed)
                total_w_user = port_df_user["weight"].sum()
                port_df_user["weight_pct"] = port_df_user["weight"] / total_w_user * 100

                cash_pct = port_df_user[port_df_user["code"] == "CASH"]["weight_pct"].sum()
                non_cash = port_df_user[port_df_user["code"] != "CASH"]
                codes = non_cash["code"].tolist()

                # ── 섹션 1: 구성 현황 ──────────────────────────────────
                st.markdown("### 1. 포트폴리오 구성 현황")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("총 종목 수", f"{len(non_cash)}개")
                col_b.metric("현금 비중", f"{cash_pct:.1f}%")
                if len(non_cash) > 0:
                    max_w_val = non_cash["weight_pct"].max()
                    col_c.metric("최대 단일 비중", f"{max_w_val:.1f}%")

                st.dataframe(
                    port_df_user[["code", "name", "weight_pct"]].rename(columns={
                        "code": "코드", "name": "종목명", "weight_pct": "비중(%)",
                    }),
                    use_container_width=True, hide_index=True,
                )

                # ── 섹션 2: 수익·위험 진단 ─────────────────────────────
                ann_ret = ann_vol = sharpe = mdd = None
                cum = None
                actual_codes = []

                if len(codes) >= 2 and not price_df_d.empty:
                    price_df_d["date"] = pd.to_datetime(price_df_d["date"])
                    pivot = price_df_d[price_df_d["code"].isin(codes)].pivot(
                        index="date", columns="code", values="close"
                    ).sort_index().ffill()

                    actual_codes = [c for c in codes if c in pivot.columns]
                    if len(actual_codes) >= 2:
                        actual_w = non_cash[non_cash["code"].isin(actual_codes)]["weight"].values
                        actual_w = actual_w / actual_w.sum()

                        rets = pivot[actual_codes].pct_change().dropna()
                        port_ret = (rets * actual_w).sum(axis=1)
                        cum = (1 + port_ret).cumprod()
                        ann_ret = port_ret.mean() * 252 * 100
                        ann_vol = port_ret.std() * np.sqrt(252) * 100
                        sharpe  = ann_ret / (ann_vol + 1e-9)
                        mdd     = ((cum - cum.cummax()) / cum.cummax()).min() * 100

                        st.divider()
                        st.markdown("### 2. 수익·위험 진단")

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("연수익률",    f"{ann_ret:.2f}%",
                                  delta="양호" if ann_ret > 5 else ("주의" if ann_ret > 0 else "위험"))
                        c2.metric("연변동성",    f"{ann_vol:.2f}%",
                                  delta="낮음" if ann_vol < 15 else ("보통" if ann_vol < 25 else "높음"),
                                  delta_color="inverse")
                        c3.metric("Sharpe Ratio", f"{sharpe:.3f}",
                                  delta="우수" if sharpe > 1 else ("양호" if sharpe > 0.5 else "미흡"))
                        c4.metric("최대 낙폭(MDD)", f"{mdd:.2f}%",
                                  delta="낮음" if mdd > -15 else ("보통" if mdd > -30 else "높음"),
                                  delta_color="inverse")

                        if ann_vol < 12 and mdd > -15:
                            risk_grade, risk_color = "저위험", "#00aa44"
                        elif ann_vol < 20 and mdd > -25:
                            risk_grade, risk_color = "중위험", "#ddaa00"
                        else:
                            risk_grade, risk_color = "고위험", "#cc0000"

                        st.markdown(
                            f"**종합 위험 등급**: "
                            f"<span style='color:{risk_color}; font-size:1.2em; font-weight:bold'>"
                            f"{risk_grade}</span>",
                            unsafe_allow_html=True,
                        )

                        fig_cum_d = go.Figure()
                        fig_cum_d.add_trace(go.Scatter(
                            x=cum.index, y=cum.values,
                            name="보유 포트폴리오", line=dict(color="#1976D2"),
                        ))
                        bm_df_d = safe_read("benchmark_price")
                        if not bm_df_d.empty:
                            bm_df_d["date"] = pd.to_datetime(bm_df_d["date"])
                            bm_c = bm_df_d.set_index("date")["close"].reindex(cum.index).ffill()
                            bm_cum_d = (1 + bm_c.pct_change().fillna(0)).cumprod()
                            fig_cum_d.add_trace(go.Scatter(
                                x=bm_cum_d.index, y=bm_cum_d.values,
                                name="KOSPI", line=dict(color="orange", dash="dash"),
                            ))
                        fig_cum_d.update_layout(
                            title="누적 수익률 vs KOSPI",
                            height=320, margin=dict(t=40, b=0),
                        )
                        st.plotly_chart(fig_cum_d, use_container_width=True)

                # ── 섹션 3: AI 점수 기반 종목 평가 ────────────────────
                if not final_df_d.empty:
                    st.divider()
                    st.markdown("### 3. 종목별 AI 점수 평가")

                    diag = pd.merge(
                        port_df_user,
                        final_df_d[["code", "final_ai_score", "final_ai_rank",
                                    "momentum_score", "stability_score"]].drop_duplicates(),
                        on="code", how="left",
                    )
                    if not cons_df_d.empty:
                        diag = pd.merge(
                            diag,
                            cons_df_d[["code", "target_price", "upside_ratio", "opinion"]].drop_duplicates(),
                            on="code", how="left",
                        )

                    diag["AI평가"] = diag["final_ai_score"].apply(
                        lambda v: ("✅ 양호" if v >= 60 else ("⚠️ 보통" if v >= 40 else "❌ 재검토"))
                        if pd.notna(v) else "N/A"
                    )

                    show_d_cols = ["name", "weight_pct", "final_ai_score", "final_ai_rank",
                                   "momentum_score", "stability_score", "AI평가"]
                    if "upside_ratio" in diag.columns:
                        show_d_cols += ["upside_ratio", "opinion"]

                    st.dataframe(
                        diag[[c for c in show_d_cols if c in diag.columns]].rename(columns={
                            "name": "종목명", "weight_pct": "비중(%)",
                            "final_ai_score": "AI점수", "final_ai_rank": "AI순위",
                            "momentum_score": "모멘텀", "stability_score": "안정성",
                            "AI평가": "AI평가", "upside_ratio": "상승여력(%)",
                            "opinion": "투자의견",
                        }),
                        use_container_width=True, hide_index=True,
                    )

                # ── 섹션 4: 처방전 ─────────────────────────────────────
                st.divider()
                st.markdown("### 4. 처방전 — 리밸런싱 제안")
                st.caption("각 항목은 AI 점수·위험 지표 기반 분석입니다. 실제 투자 권유가 아닙니다.")

                prescriptions = []

                # 집중도 위험
                nc = port_df_user[port_df_user["code"] != "CASH"]
                if not nc.empty:
                    max_row = nc.loc[nc["weight_pct"].idxmax()]
                    if max_row["weight_pct"] > 30:
                        prescriptions.append({
                            "우선순위": "🔴 긴급",
                            "진단": "집중도 위험",
                            "소견": f"**{max_row['name']}** 비중({max_row['weight_pct']:.1f}%)이 30%를 초과합니다. "
                                    f"단일 종목에 과도하게 집중되면 해당 종목의 악재가 포트폴리오 전체에 큰 손실을 줄 수 있습니다.",
                            "처방": f"{max_row['name']} 비중을 20~25%로 낮추고, 줄인 비중은 AI 점수 상위 종목에 분산 배분하세요.",
                        })

                # 현금 비중
                if cash_pct < 5:
                    prescriptions.append({
                        "우선순위": "🟡 권장",
                        "진단": "현금 부족",
                        "소견": f"현금 비중({cash_pct:.1f}%)이 5% 미만입니다. "
                                f"비상 상황 대응이나 신규 매수 기회를 잡기 어렵습니다.",
                        "처방": "수익률이 가장 낮은 보유 종목 일부를 매도하여 현금 비중을 최소 5~10%로 유지하세요.",
                    })
                elif cash_pct > 30:
                    prescriptions.append({
                        "우선순위": "🟡 권장",
                        "진단": "현금 과다",
                        "소견": f"현금 비중({cash_pct:.1f}%)이 30%를 초과합니다. "
                                f"현재 시장 국면({regime_name_diag})에서는 기회 비용이 발생하고 있습니다.",
                        "처방": f"AI 점수 상위 종목에 현금 일부를 단계적으로 투자하세요. "
                                f"시장 국면이 {regime_name_diag}이므로 "
                                + ("모멘텀 위주로 투자하세요." if regime_name_diag == "강세장"
                                   else "저변동성·배당주 위주로 투자하세요."),
                    })

                # AI 점수 낮은 종목
                if not final_df_d.empty:
                    for _, row_d in nc.iterrows():
                        ai_row = final_df_d[final_df_d["code"] == row_d["code"]]
                        if not ai_row.empty:
                            ai_score = ai_row.iloc[0]["final_ai_score"]
                            if pd.notna(ai_score) and ai_score < 40:
                                prescriptions.append({
                                    "우선순위": "🟠 검토",
                                    "진단": "저평가 종목 보유",
                                    "소견": f"**{row_d['name']}** AI점수({ai_score:.1f})가 40점 미만으로, "
                                            f"현재 비중({row_d['weight_pct']:.1f}%)이 포트폴리오 성과를 끌어내릴 수 있습니다.",
                                    "처방": f"{row_d['name']} 비중을 단계적으로 축소하고, AI 점수 상위 종목으로 교체를 검토하세요.",
                                })

                # 변동성·MDD 위험
                if ann_vol is not None and ann_vol >= 25:
                    prescriptions.append({
                        "우선순위": "🟠 검토",
                        "진단": "고변동성 포트폴리오",
                        "소견": f"포트폴리오 연간 변동성({ann_vol:.1f}%)이 25%를 초과합니다. "
                                f"손실 발생 시 심리적 부담이 커질 수 있습니다.",
                        "처방": "안정성 점수 상위 종목(저변동성·낮은 MDD) 비중을 확대하거나, 현금 비중을 늘려 리스크를 낮추세요.",
                    })

                if mdd is not None and mdd < -30:
                    prescriptions.append({
                        "우선순위": "🔴 긴급",
                        "진단": "과도한 최대낙폭",
                        "소견": f"1년 MDD({mdd:.1f}%)가 -30%를 초과합니다. "
                                f"대형 손실이 발생했거나 발생 위험이 높은 구성입니다.",
                        "처방": "개별 종목 및 포트폴리오 단위의 손절 기준(예: -15%)을 설정하고, "
                                "채권·현금 등 헤지 자산 비중을 확대하세요.",
                    })

                if prescriptions:
                    # 우선순위 정렬: 긴급 → 권장 → 검토
                    priority_order = {"🔴 긴급": 0, "🟡 권장": 1, "🟠 검토": 2}
                    prescriptions.sort(key=lambda x: priority_order.get(x["우선순위"], 9))

                    for presc in prescriptions:
                        with st.expander(f"{presc['우선순위']} {presc['진단']}"):
                            st.markdown(f"**소견**: {presc['소견']}")
                            st.markdown(f"**처방**: {presc['처방']}")
                else:
                    st.success(
                        "진단 결과 특별한 리밸런싱이 필요하지 않습니다. "
                        "현재 포트폴리오 구성을 유지하면서 정기적으로 모니터링하세요."
                    )



# ═══════════════════════════════════════════════════════════════════════
# TAB 5: AI 리포트
# ═══════════════════════════════════════════════════════════════════════
with tab_report:
    st.header("🤖 AI 리포트")

    report_tabs = st.tabs(["📄 저장된 리포트", "✍️ AI 리포트 생성"])

    # ── 리포트 서브탭 1: 저장된 리포트 ────────────────────────────────
    with report_tabs[0]:
        report_df = safe_read("ai_report_result_final_enhanced")
        if report_df.empty:
            st.warning("저장된 리포트 없음. 파이프라인을 실행하거나 아래에서 재생성하세요.")
        else:
            r_rep = report_df.iloc[-1]

            with st.expander("리포트 작성 기준", expanded=True):
                meta_col1, meta_col2 = st.columns(2)
                with meta_col1:
                    report_type_label = (
                        "OpenAI GPT 생성" if r_rep.get("report_type") == "openai"
                        else "템플릿 자동 생성"
                    )
                    st.markdown(
                        f"**생성일**: {r_rep.get('report_date', 'N/A')}  \n"
                        f"**유형**: {r_rep.get('report_type', 'N/A')} ({report_type_label})"
                    )
                    try:
                        ev = json.loads(r_rep.get("evidence_json", "{}"))
                        regime_info = ev.get("market_regime", {})
                        top_stocks_ev = ev.get("top_stocks", [])
                        kospi_val = regime_info.get("kospi_close", "N/A")
                        kospi_str = f"{kospi_val:,.0f}pt" if isinstance(kospi_val, (int, float)) else str(kospi_val)
                        ts_val = regime_info.get("total_score", "N/A")
                        ts_str = f"{ts_val:.1f}" if isinstance(ts_val, (int, float)) else str(ts_val)
                        st.markdown(
                            f"**분석 기준 국면**: {regime_info.get('regime', 'N/A')} "
                            f"(종합점수 {ts_str})  \n"
                            f"**KOSPI (분석 시점)**: {kospi_str}  \n"
                            f"**분석 대상**: 상위 {len(top_stocks_ev)}개 종목"
                        )
                    except Exception:
                        pass

                with meta_col2:
                    st.markdown("""
                    **데이터 소스**
                    - 주가: FinanceDataReader (실제 데이터)
                    - KOSPI: NAVER 금융 fchart API (실제 데이터)
                    - 재무: NAVER 금융 + DART OpenAPI (실제 데이터)
                    - 컨센서스: NAVER 금융 / FnGuide (실제 데이터)
                    - AI 점수: 퀀트×70% + ML×20% + 위험방어×10%
                    """)

            st.divider()
            st.markdown(r_rep.get("report_text", "내용 없음"))

            with st.expander("Evidence JSON 원본 보기"):
                try:
                    ev = json.loads(r_rep.get("evidence_json", "{}"))
                    st.json(ev)
                except Exception:
                    st.text(r_rep.get("evidence_json", ""))

    # ── 리포트 서브탭 2: AI 리포트 생성 ────────────────────────────────
    with report_tabs[1]:
        st.subheader("AI 리포트 생성")

        report_type_sel = st.radio(
            "리포트 유형", ["템플릿 리포트", "OpenAI 생성형 리포트"], key="report_type_radio"
        )
        api_key_input = ""
        if report_type_sel == "OpenAI 생성형 리포트":
            api_key_input = st.text_input("OpenAI API Key", type="password", key="openai_key")
            st.caption("입력한 API Key는 이 세션에서만 사용됩니다.")

        st.markdown("**포함할 섹션 선택**")
        with st.expander("각 섹션 설명"):
            st.markdown("""
            | 섹션 | 내용 |
            |------|------|
            | **시장 브리프** | 현재 KOSPI 지수, 시장 국면(강세/중립/약세/하락), 1M·3M 수익률, 20일 변동성, 1년 MDD, 200일선 위치, 추천 전략을 요약합니다 |
            | **종목 리포트** | 종목선정 탭에서 선택한 상위 N개 종목의 최종 AI 점수, 퀀트 점수, 리서치 점수, ML 상승확률, 모멘텀·안정성, 목표주가 상승여력, 투자의견을 정리합니다 |
            | **포트폴리오 진단** | 종목선정 탭에서 선택한 상위 N개 종목을 기반으로 각 최적화 전략(동일비중·최소분산·최대샤프·Risk Parity·AI점수가중)의 연수익률, Sharpe 비율, MDD를 비교하고 현재 시장 국면에 맞는 전략을 제안합니다 |
            | **리밸런싱 제안** | 현재 포트폴리오 구성에서 집중도 위험, 현금 비중, AI 점수 낮은 종목, 변동성 위험에 대한 구체적 처방을 제공합니다 |
            """)

        report_focus = st.multiselect("섹션 선택", [
            "시장 브리프", "종목 리포트", "포트폴리오 진단", "리밸런싱 제안"
        ], default=["시장 브리프", "종목 리포트", "포트폴리오 진단"], key="report_focus_sel")

        selected_top_n_r = st.session_state.get("top_n_slider", 15)
        st.info(f"종목선정 탭에서 선택한 상위 **{selected_top_n_r}개** 종목을 기반으로 리포트를 생성합니다. 종목 수를 변경하려면 종목선정 탭의 슬라이더를 먼저 조정하세요.")

        if st.button("리포트 생성", key="gen_report_btn"):
            with st.spinner("리포트 생성 중..."):
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location("ai_report_mod", "pipeline/04_ai_report.py")
                _mod  = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)

                final_df_r    = safe_read("score_result_final_ai_by_model_all")
                regime_df_r   = safe_read("market_regime")
                portfolio_df_r = safe_read("advanced_portfolio_performance")
                consensus_df_r = safe_read("research_consensus_factor_all")

                evidence = _mod.build_evidence(final_df_r, regime_df_r, portfolio_df_r, consensus_df_r, top_n=selected_top_n_r)

                if report_type_sel == "OpenAI 생성형 리포트" and api_key_input:
                    report_text = _mod.generate_openai_report(evidence, api_key_input)
                else:
                    report_text = _mod.generate_template_report(evidence)

                st.markdown(report_text)

                con = get_connection()
                if con:
                    pd.DataFrame([{
                        "report_date": evidence["report_date"],
                        "report_type": "openai" if (
                            report_type_sel == "OpenAI 생성형 리포트" and api_key_input
                        ) else "template",
                        "evidence_json": json.dumps(evidence, ensure_ascii=False),
                        "report_text": report_text,
                    }]).to_sql("ai_report_result_final_enhanced", con, if_exists="replace", index=False)
                    st.success("리포트 저장 완료")
