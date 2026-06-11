"""
sector_analysis_module.py
────────────────────────────────────────────────────────────────────────
다른 Streamlit 프로젝트에서 섹터 분석 탭을 재사용하기 위한 모듈.

사용 방법 (다른 app.py에서):
    from sector_analysis_module import SECTOR_MAP, code_to_sector, render_sector_tab
    tab_sector = st.tabs(["🏭 섹터 분석"])[0]
    with tab_sector:
        render_sector_tab(safe_read_fn)

의존성:
    streamlit, pandas, numpy, plotly

필요한 DB 테이블:
    score_result_final_ai_by_model_all  (final_ai_score, ret_3m, ret_6m, ret_12m, vol_20, mdd_1y, ...)
    price_data                          (code, date, close, volume)
    benchmark_price                     (date, close)

종목 코드 기준: KOSPI 40개 종목 (2026년 기준)
────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ── 섹터 매핑 ─────────────────────────────────────────────────────────────────

SECTOR_MAP = {
    "반도체/IT":    ["005930", "000660", "009150", "018260"],
    "배터리/소재":  ["051910", "006400", "003670", "010130", "005490"],
    "자동차":       ["000270", "005380", "012330"],
    "금융":         ["086790", "105560", "055550", "024110", "000810", "032830"],
    "바이오/헬스":  ["207940", "000100", "068270"],
    "플랫폼/통신":  ["035420", "035720", "030200", "017670"],
    "에너지/화학":  ["096770", "011170", "034730"],
    "소비재/유통":  ["282330", "271560", "001040", "097950", "051900", "090430"],
    "산업재":       ["047050", "028260", "011200", "003490", "066570", "003550"],
}


def code_to_sector(code: str) -> str:
    """종목 코드를 받아 섹터명을 반환합니다. 매핑 없으면 '기타'."""
    for sec, codes in SECTOR_MAP.items():
        if code in codes:
            return sec
    return "기타"


# ── 메인 렌더 함수 ────────────────────────────────────────────────────────────

def render_sector_tab(safe_read_fn):
    """
    섹터 분석 탭 전체를 렌더링합니다.

    Parameters
    ----------
    safe_read_fn : callable
        테이블명을 받아 DataFrame을 반환하는 함수.
        예) lambda tbl: pd.read_sql(f"SELECT * FROM {tbl}", con)
    """
    st.header("🏭 섹터 분석")

    sec_final_df = safe_read_fn("score_result_final_ai_by_model_all")
    sec_price_df = safe_read_fn("price_data")
    sec_bm_df    = safe_read_fn("benchmark_price")

    if sec_final_df.empty:
        st.warning("데이터 없음. 파이프라인을 실행하세요.")
        return

    sec_final_df["sector"] = sec_final_df["code"].apply(code_to_sector)

    sector_tabs = st.tabs([
        "📊 섹터 성과",
        "🔄 섹터 로테이션",
        "🏆 섹터별 AI 점수",
        "📋 섹터 종목 현황",
    ])

    # ── 섹터 탭 1: 섹터별 성과 지표 ─────────────────────────────────────────
    with sector_tabs[0]:
        st.subheader("섹터별 평균 수익률 & 변동성")

        sec_perf = sec_final_df.groupby("sector").agg(
            종목수=("code", "count"),
            평균AI점수=("final_ai_score", "mean"),
            평균1M수익률=("ret_3m", lambda x: x.mean() / 3),
            평균3M수익률=("ret_3m", "mean"),
            평균6M수익률=("ret_6m", "mean"),
            평균12M수익률=(
                "ret_12m" if "ret_12m" in sec_final_df.columns else "ret_6m", "mean"
            ),
            평균변동성=("vol_20", "mean"),
            평균MDD=("mdd_1y", "mean"),
        ).reset_index().rename(columns={"sector": "섹터"})

        for col in ["평균AI점수", "평균3M수익률", "평균6M수익률",
                    "평균12M수익률", "평균변동성", "평균MDD"]:
            if col in sec_perf.columns:
                sec_perf[col] = sec_perf[col].round(2)

        st.dataframe(
            sec_perf.sort_values("평균AI점수", ascending=False)
                .style.background_gradient(subset=["평균AI점수"], cmap="Blues")
                .background_gradient(subset=["평균3M수익률"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True,
        )

        fig_sec_scatter = px.scatter(
            sec_perf,
            x="평균변동성", y="평균AI점수",
            size="종목수", color="섹터",
            text="섹터",
            title="섹터별 AI 점수 vs 변동성 (버블 크기 = 종목 수)",
            labels={"평균변동성": "평균 20일 변동성(%)", "평균AI점수": "평균 AI 점수"},
        )
        fig_sec_scatter.update_traces(textposition="top center")
        fig_sec_scatter.update_layout(height=420, margin=dict(t=50, b=0))
        st.plotly_chart(fig_sec_scatter, use_container_width=True)

    # ── 섹터 탭 2: 섹터 로테이션 ─────────────────────────────────────────────
    with sector_tabs[1]:
        st.subheader("섹터 로테이션 분석")
        st.caption("기간별 평균 수익률을 비교해 어느 섹터가 최근 강세인지 확인합니다.")

        ret_cols = {}
        if "ret_3m" in sec_final_df.columns:
            ret_cols["1M(근사)"] = sec_final_df.groupby("sector")["ret_3m"].mean() / 3
            ret_cols["3M"]      = sec_final_df.groupby("sector")["ret_3m"].mean()
        if "ret_6m" in sec_final_df.columns:
            ret_cols["6M"]      = sec_final_df.groupby("sector")["ret_6m"].mean()
        if "ret_12m" in sec_final_df.columns:
            ret_cols["12M"]     = sec_final_df.groupby("sector")["ret_12m"].mean()

        if ret_cols:
            rotation_df = (
                pd.DataFrame(ret_cols)
                .reset_index()
                .rename(columns={"sector": "섹터"})
            )
            melt_df = rotation_df.melt(id_vars="섹터", var_name="기간", value_name="수익률(%)")

            fig_rot = px.bar(
                melt_df,
                x="섹터", y="수익률(%)", color="기간",
                barmode="group",
                title="섹터별 기간별 평균 수익률",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_rot.update_layout(height=450, margin=dict(t=50, b=0))
            st.plotly_chart(fig_rot, use_container_width=True)

            if "3M" in rotation_df.columns:
                best_sec  = rotation_df.nlargest(1, "3M").iloc[0]
                worst_sec = rotation_df.nsmallest(1, "3M").iloc[0]
                col_b, col_w = st.columns(2)
                col_b.success(f"3M 최강세 섹터: **{best_sec['섹터']}** ({best_sec['3M']:.2f}%)")
                col_w.warning(f"3M 최약세 섹터: **{worst_sec['섹터']}** ({worst_sec['3M']:.2f}%)")

        if not sec_price_df.empty:
            st.divider()
            st.markdown("**섹터별 누적 수익률 추이**")
            sec_price_df["date"] = pd.to_datetime(sec_price_df["date"])

            all_sector_cum = {}
            for sec_name, sec_codes in SECTOR_MAP.items():
                avail = [c for c in sec_codes if c in sec_price_df["code"].values]
                if not avail:
                    continue
                pivot_s = (
                    sec_price_df[sec_price_df["code"].isin(avail)]
                    .pivot(index="date", columns="code", values="close")
                    .sort_index().ffill()
                )
                eq_ret = pivot_s.pct_change().dropna().mean(axis=1)
                all_sector_cum[sec_name] = (1 + eq_ret).cumprod()

            if all_sector_cum:
                cum_df = pd.DataFrame(all_sector_cum)
                fig_cum_sec = go.Figure()
                colors_sec = px.colors.qualitative.Plotly
                for i, col in enumerate(cum_df.columns):
                    fig_cum_sec.add_trace(go.Scatter(
                        x=cum_df.index, y=cum_df[col],
                        name=col,
                        line=dict(color=colors_sec[i % len(colors_sec)], width=1.5),
                    ))
                if not sec_bm_df.empty:
                    sec_bm_df["date"] = pd.to_datetime(sec_bm_df["date"])
                    bm_s = (
                        sec_bm_df.set_index("date")["close"]
                        .reindex(cum_df.index).ffill()
                    )
                    bm_cum_s = (1 + bm_s.pct_change().fillna(0)).cumprod()
                    fig_cum_sec.add_trace(go.Scatter(
                        x=bm_cum_s.index, y=bm_cum_s.values,
                        name="KOSPI",
                        line=dict(color="black", width=2, dash="dash"),
                    ))
                fig_cum_sec.update_layout(
                    title="섹터별 동일비중 누적 수익률 vs KOSPI",
                    height=420, margin=dict(t=50, b=0),
                )
                st.plotly_chart(fig_cum_sec, use_container_width=True)

    # ── 섹터 탭 3: 섹터별 AI 점수 ────────────────────────────────────────────
    with sector_tabs[2]:
        st.subheader("섹터별 AI 점수 분포")

        fig_box = px.box(
            sec_final_df,
            x="sector", y="final_ai_score",
            color="sector",
            points="all",
            title="섹터별 최종 AI 점수 분포 (박스플롯)",
            labels={"sector": "섹터", "final_ai_score": "최종 AI 점수"},
        )
        fig_box.update_layout(height=450, margin=dict(t=50, b=0), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        st.divider()
        st.markdown("**섹터별 팩터 평균 점수 히트맵**")
        factor_cols  = ["profitability_score", "valuation_score", "momentum_score",
                        "stability_score", "liquidity_score"]
        factor_labels = ["수익성", "밸류에이션", "모멘텀", "안정성", "유동성"]
        avail_f = [c for c in factor_cols if c in sec_final_df.columns]
        avail_l = [factor_labels[factor_cols.index(c)] for c in avail_f]

        if avail_f:
            sec_factor = sec_final_df.groupby("sector")[avail_f].mean()
            sec_factor.columns = avail_l
            fig_fhm = px.imshow(
                sec_factor,
                color_continuous_scale="RdYlGn",
                zmin=0, zmax=100,
                text_auto=".1f",
                title="섹터별 팩터 평균 점수",
                labels=dict(x="팩터", y="섹터", color="점수"),
            )
            fig_fhm.update_layout(height=380, margin=dict(t=50, b=0))
            fig_fhm.update_traces(textfont_size=10)
            st.plotly_chart(fig_fhm, use_container_width=True)

    # ── 섹터 탭 4: 섹터 종목 현황 ────────────────────────────────────────────
    with sector_tabs[3]:
        st.subheader("섹터별 종목 현황")

        sel_sec = st.selectbox(
            "섹터 선택",
            sorted(sec_final_df["sector"].unique()),
            key="sector_stock_sel",
        )
        sec_stocks = (
            sec_final_df[sec_final_df["sector"] == sel_sec]
            .sort_values("final_ai_rank")
        )
        show_sec_cols = [c for c in [
            "final_ai_rank", "code", "name", "final_ai_score",
            "quant_score", "momentum_score", "stability_score",
            "ml_upside_prob", "ret_3m", "ret_6m",
        ] if c in sec_stocks.columns]

        st.dataframe(
            sec_stocks[show_sec_cols].rename(columns={
                "final_ai_rank": "AI순위", "code": "종목코드", "name": "종목명",
                "final_ai_score": "AI점수", "quant_score": "퀀트점수",
                "momentum_score": "모멘텀", "stability_score": "안정성",
                "ml_upside_prob": "ML상승확률",
                "ret_3m": "3M수익률", "ret_6m": "6M수익률",
            }).style.background_gradient(subset=["AI점수"], cmap="Blues"),
            use_container_width=True, hide_index=True,
        )

        if not sec_stocks.empty and len(sec_stocks) >= 2:
            st.divider()
            st.markdown("**섹터 내 종목 팩터 비교**")
            f_cols = ["profitability_score", "valuation_score", "momentum_score",
                      "stability_score", "liquidity_score"]
            f_lbls = ["수익성", "밸류에이션", "모멘텀", "안정성", "유동성"]
            avail_fc = [c for c in f_cols if c in sec_stocks.columns]
            avail_fl = [f_lbls[f_cols.index(c)] for c in avail_fc]

            fig_sec_radar = go.Figure()
            colors_r = px.colors.qualitative.Plotly
            for i, (_, sr) in enumerate(sec_stocks.head(6).iterrows()):
                vals = [float(sr.get(c, 50)) for c in avail_fc]
                fig_sec_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=avail_fl + [avail_fl[0]],
                    name=sr.get("name", sr["code"]),
                    line=dict(color=colors_r[i % len(colors_r)]),
                    fill="toself",
                    fillcolor=(
                        f"rgba{tuple(list(px.colors.hex_to_rgb(colors_r[i % len(colors_r)])) + [0.1])}"
                    ),
                ))
            fig_sec_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title=f"{sel_sec} 섹터 종목 팩터 레이더 (최대 6개)",
                height=450, margin=dict(t=60, b=10),
            )
            st.plotly_chart(fig_sec_radar, use_container_width=True)
