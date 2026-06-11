"""
06_market_regime.py
KOSPI 기반 시장 국면 판단 (모멘텀·변동성·MDD·200일선) → SQLite 저장
"""
import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/quant_project.db"


def get_db():
    return sqlite3.connect(DB_PATH)


def detect_regime(bm_df):
    bm_df = bm_df.copy()
    bm_df["date"] = pd.to_datetime(bm_df["date"])
    bm_df = bm_df.set_index("date").sort_index()
    close = bm_df["close"]

    # 모멘텀 지표
    ret_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
    ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0
    ret_6m = (close.iloc[-1] / close.iloc[-126] - 1) * 100 if len(close) >= 126 else 0

    # 변동성 지표
    rets = close.pct_change().dropna()
    vol_20 = rets.tail(20).std() * np.sqrt(252) * 100
    vol_60 = rets.tail(60).std() * np.sqrt(252) * 100

    # MDD (최근 1년)
    recent = close.tail(252)
    roll_max = recent.cummax()
    mdd_1y = ((recent - roll_max) / roll_max).min() * 100

    # 200일선 대비 위치
    ma200 = close.tail(200).mean() if len(close) >= 200 else close.mean()
    above_ma200 = close.iloc[-1] > ma200

    # 국면 판단 점수 (0~100)
    momentum_signal = np.clip((ret_1m + ret_3m * 0.5 + ret_6m * 0.3) / 3, -20, 20)
    momentum_score = (momentum_signal + 20) / 40 * 100

    vol_signal = np.clip(vol_20, 5, 40)
    vol_score = (40 - vol_signal) / 35 * 100

    mdd_signal = np.clip(abs(mdd_1y), 0, 30)
    mdd_score = (30 - mdd_signal) / 30 * 100

    ma_score = 100 if above_ma200 else 0

    total_score = (
        momentum_score * 0.35
        + vol_score * 0.25
        + mdd_score * 0.25
        + ma_score * 0.15
    )

    # 국면 분류
    if total_score >= 65:
        regime = "강세장"
        strategy_rec = "공격형 모멘텀 전략 (모멘텀·수익성 팩터 비중 확대)"
        weight_adj = {"momentum": 1.3, "profitability": 1.2, "valuation": 1.0, "stability": 0.8, "liquidity": 1.0}
    elif total_score >= 45:
        regime = "중립장"
        strategy_rec = "균형형 전략 (팩터 동일 가중)"
        weight_adj = {"momentum": 1.0, "profitability": 1.0, "valuation": 1.1, "stability": 1.0, "liquidity": 1.0}
    elif total_score >= 30:
        regime = "약세장"
        strategy_rec = "방어형 전략 (안정성·유동성 팩터 비중 확대)"
        weight_adj = {"momentum": 0.7, "profitability": 0.9, "valuation": 1.1, "stability": 1.4, "liquidity": 1.2}
    else:
        regime = "하락장"
        strategy_rec = "현금 비중 확대 + 저변동성 방어주 위주"
        weight_adj = {"momentum": 0.5, "profitability": 0.8, "valuation": 1.0, "stability": 1.5, "liquidity": 1.3}

    regime_data = {
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "kospi_close": round(float(close.iloc[-1]), 2),
        "ret_1m": round(ret_1m, 2),
        "ret_3m": round(ret_3m, 2),
        "ret_6m": round(ret_6m, 2),
        "vol_20": round(vol_20, 2),
        "vol_60": round(vol_60, 2),
        "mdd_1y": round(mdd_1y, 2),
        "above_ma200": int(above_ma200),
        "ma200": round(float(ma200), 2),
        "momentum_score": round(momentum_score, 2),
        "vol_score": round(vol_score, 2),
        "mdd_score": round(mdd_score, 2),
        "ma_score": round(ma_score, 2),
        "total_score": round(total_score, 2),
        "regime": regime,
        "strategy_rec": strategy_rec,
        "w_momentum": weight_adj["momentum"],
        "w_profitability": weight_adj["profitability"],
        "w_valuation": weight_adj["valuation"],
        "w_stability": weight_adj["stability"],
        "w_liquidity": weight_adj["liquidity"],
    }

    return regime_data, weight_adj


def apply_regime_to_scores(score_df, weight_adj):
    df = score_df.copy()
    adjusted = (
        df["profitability_score"] * weight_adj["profitability"] * 0.2
        + df["valuation_score"] * weight_adj["valuation"] * 0.2
        + df["momentum_score"] * weight_adj["momentum"] * 0.25
        + df["stability_score"] * weight_adj["stability"] * 0.20
        + df["liquidity_score"] * weight_adj["liquidity"] * 0.15
    )
    # 재정규화
    mn, mx = adjusted.min(), adjusted.max()
    df["regime_adjusted_score"] = (adjusted - mn) / (mx - mn + 1e-9) * 100
    df["regime_rank"] = df["regime_adjusted_score"].rank(ascending=False).astype(int)
    return df


def main():
    print("=" * 60)
    print("06. 시장 국면 판단 시작")
    print("=" * 60)

    con = get_db()
    bm_df = pd.read_sql("SELECT * FROM benchmark_price", con)
    score_df = pd.read_sql("SELECT * FROM score_result_all", con)

    regime_data, weight_adj = detect_regime(bm_df)

    regime_df = pd.DataFrame([regime_data])
    regime_df.to_sql("market_regime", con, if_exists="replace", index=False)
    print(f"[완료] market_regime 저장")
    print(f"  KOSPI: {regime_data['kospi_close']:,.0f}pt")
    print(f"  국면: {regime_data['regime']} (종합점수: {regime_data['total_score']:.1f})")
    print(f"  추천 전략: {regime_data['strategy_rec']}")
    print(f"  200일선 대비: {'위' if regime_data['above_ma200'] else '아래'}")

    # 국면 반영 점수
    adjusted_df = apply_regime_to_scores(score_df, weight_adj)
    adjusted_df.to_sql("score_result_regime_adjusted", con, if_exists="replace", index=False)
    print(f"\n[완료] score_result_regime_adjusted 저장")
    print("\n국면 반영 상위 5개 종목:")
    top5 = adjusted_df.nsmallest(5, "regime_rank")[["regime_rank", "code", "name", "regime_adjusted_score"]]
    print(top5.to_string(index=False))

    con.close()
    print("\n[06] 시장 국면 판단 완료\n")


if __name__ == "__main__":
    main()
