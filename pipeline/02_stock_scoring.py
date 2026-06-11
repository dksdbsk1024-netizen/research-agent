"""
02_stock_scoring.py
수익성·밸류에이션·모멘텀·안정성·유동성 팩터 기반 종목 점수 계산 → SQLite 저장
실제 재무 데이터가 없으므로 가격 데이터에서 파생 가능한 팩터 위주로 계산
"""
import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/quant_project.db"


def get_db():
    return sqlite3.connect(DB_PATH)


def minmax_score(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return (series - mn) / (mx - mn) * 100


def compute_scores(price_df, master_df):
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values(["code", "date"])

    records = []
    for code in master_df["code"]:
        sub = price_df[price_df["code"] == code].copy()
        if len(sub) < 60:
            continue

        sub = sub.set_index("date").sort_index()
        close = sub["close"]
        volume = sub["volume"]

        latest_close = close.iloc[-1]

        # 모멘텀: 3M, 6M, 12M 수익률
        def ret(n):
            if len(close) > n:
                return (close.iloc[-1] / close.iloc[-n] - 1) * 100
            return 0.0

        ret_3m = ret(63)
        ret_6m = ret(126)
        ret_12m = ret(252)
        momentum = ret_3m * 0.4 + ret_6m * 0.3 + ret_12m * 0.3

        # 안정성: 20일/60일 변동성, 1년 MDD
        rets = close.pct_change().dropna()
        vol_20 = rets.tail(20).std() * np.sqrt(252) * 100
        vol_60 = rets.tail(60).std() * np.sqrt(252) * 100

        rolling_max = close.rolling(252, min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max * 100
        mdd_1y = drawdown.tail(252).min()

        stability = -(vol_20 * 0.4 + vol_60 * 0.3 + abs(mdd_1y) * 0.3)

        # 유동성: 60일 평균 거래대금
        amt_60 = (close * volume).tail(60).mean()

        # 수익성 proxy: 12M 수익률 / 변동성 (샤프류)
        profitability_proxy = ret_12m / (vol_60 + 1e-6)

        # 밸류에이션 proxy: 52주 저점 대비 현재가 위치 (낮을수록 저평가)
        high_52w = close.tail(252).max()
        low_52w = close.tail(252).min()
        val_position = (latest_close - low_52w) / (high_52w - low_52w + 1e-6)
        valuation_proxy = -val_position * 100  # 낮을수록 저평가 = 높은 점수

        records.append({
            "code": code,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "momentum_raw": momentum,
            "vol_20": vol_20,
            "vol_60": vol_60,
            "mdd_1y": mdd_1y,
            "stability_raw": stability,
            "amt_60": amt_60,
            "profitability_raw": profitability_proxy,
            "valuation_raw": valuation_proxy,
            "latest_close": latest_close,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.merge(master_df[["code", "name"]], on="code", how="left")

    # 각 팩터 0~100 정규화
    df["profitability_score"] = minmax_score(df["profitability_raw"])
    df["valuation_score"] = minmax_score(df["valuation_raw"])
    df["momentum_score"] = minmax_score(df["momentum_raw"])
    df["stability_score"] = minmax_score(df["stability_raw"])
    df["liquidity_score"] = minmax_score(df["amt_60"])

    # 기본 종합 점수 (동일 가중)
    df["quant_score"] = (
        df["profitability_score"] * 0.2
        + df["valuation_score"] * 0.2
        + df["momentum_score"] * 0.25
        + df["stability_score"] * 0.20
        + df["liquidity_score"] * 0.15
    )
    df["rank"] = df["quant_score"].rank(ascending=False).astype(int)

    return df.sort_values("rank").reset_index(drop=True)


def main():
    print("=" * 60)
    print("02. 종목 스코어링 시작")
    print("=" * 60)

    con = get_db()
    price_df = pd.read_sql("SELECT * FROM price_data", con)
    master_df = pd.read_sql("SELECT * FROM stock_master", con)

    scored = compute_scores(price_df, master_df)
    if scored.empty:
        print("[오류] 스코어링 결과 없음")
        con.close()
        return

    scored.to_sql("score_result_all", con, if_exists="replace", index=False)
    scored.to_sql("score_result", con, if_exists="replace", index=False)
    print(f"[완료] score_result_all: {len(scored)}개 종목 저장")
    print("\n상위 10개 종목:")
    cols = ["rank", "code", "name", "quant_score", "momentum_score", "stability_score"]
    print(scored[cols].head(10).to_string(index=False))

    con.close()
    print("\n[02] 종목 스코어링 완료\n")


if __name__ == "__main__":
    main()
