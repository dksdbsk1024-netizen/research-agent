"""
03_portfolio_backtest.py
기본 포트폴리오 전략(동일비중, 최소분산, 최대샤프) 백테스트 → SQLite 저장
"""
import sqlite3
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/quant_project.db"
TOP_N = 15
REBALANCE_MONTHS = 1


def get_db():
    return sqlite3.connect(DB_PATH)


def build_return_matrix(price_df, codes):
    price_df["date"] = pd.to_datetime(price_df["date"])
    pivot = price_df[price_df["code"].isin(codes)].pivot(index="date", columns="code", values="close")
    pivot = pivot.sort_index().ffill()
    rets = pivot.pct_change().dropna()
    return rets


def portfolio_metrics(weights, rets):
    port_ret = (rets * weights).sum(axis=1)
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)
    cum = (1 + port_ret).cumprod()
    roll_max = cum.cummax()
    mdd = ((cum - roll_max) / roll_max).min()
    return ann_ret, ann_vol, sharpe, mdd, port_ret


def equal_weight(n):
    return np.ones(n) / n


def min_variance(rets):
    n = rets.shape[1]
    cov = rets.cov().values * 252
    def objective(w):
        return w @ cov @ w
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.3)] * n
    result = minimize(objective, equal_weight(n), method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x if result.success else equal_weight(n)


def max_sharpe(rets):
    n = rets.shape[1]
    cov = rets.cov().values * 252
    mean_ret = rets.mean().values * 252
    def neg_sharpe(w):
        p_ret = w @ mean_ret
        p_vol = np.sqrt(w @ cov @ w)
        return -p_ret / (p_vol + 1e-9)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.3)] * n
    result = minimize(neg_sharpe, equal_weight(n), method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x if result.success else equal_weight(n)


def risk_parity(rets):
    n = rets.shape[1]
    cov = rets.cov().values * 252
    def risk_contrib_diff(w):
        port_vol = np.sqrt(w @ cov @ w)
        marginal = cov @ w
        contrib = w * marginal / (port_vol + 1e-9)
        target = port_vol / n
        return np.sum((contrib - target) ** 2)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.3)] * n
    result = minimize(risk_contrib_diff, equal_weight(n), method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x if result.success else equal_weight(n)


def run_backtest(rets, weight_fn, name):
    weights = weight_fn(rets) if callable(weight_fn) else weight_fn
    ann_ret, ann_vol, sharpe, mdd, port_ret = portfolio_metrics(weights, rets)
    return {
        "strategy": name,
        "ann_return": round(ann_ret * 100, 2),
        "ann_volatility": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "mdd": round(mdd * 100, 2),
        "weights": str(dict(zip(rets.columns, weights.round(4)))),
    }


def main():
    print("=" * 60)
    print("03. 포트폴리오 백테스트 시작")
    print("=" * 60)

    con = get_db()
    price_df = pd.read_sql("SELECT * FROM price_data", con)
    score_df = pd.read_sql("SELECT * FROM score_result_all", con)

    top_codes = score_df.nsmallest(TOP_N, "rank")["code"].tolist()
    rets = build_return_matrix(price_df, top_codes)

    if rets.empty or len(rets) < 60:
        print("[오류] 수익률 데이터 부족")
        con.close()
        return

    n = rets.shape[1]
    results = []

    print(f"TOP {TOP_N} 종목 기반 백테스트")
    results.append(run_backtest(rets, lambda r: equal_weight(r.shape[1]), "동일비중"))
    results.append(run_backtest(rets, min_variance, "최소분산"))
    results.append(run_backtest(rets, max_sharpe, "최대샤프"))
    results.append(run_backtest(rets, risk_parity, "Risk Parity"))

    result_df = pd.DataFrame(results)
    result_df.to_sql("portfolio_backtest_result", con, if_exists="replace", index=False)
    print("[완료] portfolio_backtest_result 저장")
    print(result_df[["strategy", "ann_return", "ann_volatility", "sharpe_ratio", "mdd"]].to_string(index=False))

    # 벤치마크 비교
    bm_df = pd.read_sql("SELECT * FROM benchmark_price", con)
    bm_df["date"] = pd.to_datetime(bm_df["date"])
    bm_df = bm_df.set_index("date").sort_index()
    bm_ret = bm_df["close"].pct_change().dropna()
    bm_common = bm_ret.reindex(rets.index).dropna()
    if len(bm_common) > 10:
        bm_ann = bm_common.mean() * 252 * 100
        bm_vol = bm_common.std() * np.sqrt(252) * 100
        bm_sharpe = bm_ann / (bm_vol + 1e-9)
        print(f"\nKOSPI 벤치마크: 연수익률={bm_ann:.2f}%, 변동성={bm_vol:.2f}%, Sharpe={bm_sharpe:.3f}")

    con.close()
    print("\n[03] 포트폴리오 백테스트 완료\n")


if __name__ == "__main__":
    main()
