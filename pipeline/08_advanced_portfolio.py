"""
08_advanced_portfolio.py
Risk Parity, AI 점수 가중, Efficient Frontier, 위험기여도 분석 → SQLite 저장
"""
import sqlite3
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/quant_project.db"
TOP_N = 15


def get_db():
    return sqlite3.connect(DB_PATH)


def build_return_matrix(price_df, codes):
    price_df["date"] = pd.to_datetime(price_df["date"])
    pivot = price_df[price_df["code"].isin(codes)].pivot(
        index="date", columns="code", values="close"
    )
    pivot = pivot.sort_index().ffill()
    return pivot.pct_change().dropna()


def portfolio_metrics(weights, rets):
    port_ret = (rets * weights).sum(axis=1)
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)
    cum = (1 + port_ret).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    cagr = (cum.iloc[-1]) ** (252 / len(port_ret)) - 1 if len(port_ret) > 0 else 0
    return {
        "ann_return": ann_ret * 100,
        "ann_vol": ann_vol * 100,
        "sharpe": sharpe,
        "mdd": mdd * 100,
        "cagr": cagr * 100,
    }


def risk_contribution(weights, cov):
    port_vol = np.sqrt(weights @ cov @ weights)
    marginal = cov @ weights
    contrib = weights * marginal / (port_vol + 1e-9)
    return contrib


def risk_parity_weights(rets):
    n = rets.shape[1]
    cov = rets.cov().values * 252
    w0 = np.ones(n) / n

    def obj(w):
        rc = risk_contribution(w, cov)
        target = np.sum(w @ cov @ w) / n
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.30)] * n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x if res.success else w0


def ai_score_weights(rets, ai_scores, codes):
    score_map = dict(zip(ai_scores["code"], ai_scores["final_ai_score"]))
    scores = np.array([score_map.get(c, 50) for c in codes])
    scores = np.clip(scores, 1, 100)
    weights = scores / scores.sum()
    # 상한 30% 제한
    while weights.max() > 0.30:
        weights = np.clip(weights, 0, 0.30)
        weights /= weights.sum()
    return weights


def min_variance_weights(rets):
    n = rets.shape[1]
    cov = rets.cov().values * 252
    w0 = np.ones(n) / n

    def obj(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.30)] * n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x if res.success else w0


def max_sharpe_weights(rets):
    n = rets.shape[1]
    cov = rets.cov().values * 252
    mean_ret = rets.mean().values * 252
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        ret = w @ mean_ret
        vol = np.sqrt(w @ cov @ w)
        return -ret / (vol + 1e-9)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.30)] * n
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x if res.success else w0


def main():
    print("=" * 60)
    print("08. 고도화 포트폴리오 분석 시작")
    print("=" * 60)

    con = get_db()
    price_df = pd.read_sql("SELECT * FROM price_data", con)
    final_df = pd.read_sql("SELECT * FROM score_result_final_ai_by_model_all", con)

    top_codes = final_df.nsmallest(TOP_N, "final_ai_rank")["code"].tolist()
    rets = build_return_matrix(price_df, top_codes)

    if rets.empty or len(rets) < 60:
        print("[오류] 수익률 데이터 부족")
        con.close()
        return

    actual_codes = list(rets.columns)
    n = len(actual_codes)

    strategies = {
        "동일비중": np.ones(n) / n,
        "최소분산": min_variance_weights(rets),
        "최대샤프": max_sharpe_weights(rets),
        "Risk Parity": risk_parity_weights(rets),
        "AI점수가중": ai_score_weights(rets, final_df, actual_codes),
    }

    perf_records = []
    weight_records = []
    risk_records = []

    cov_matrix = rets.cov().values * 252

    for strat_name, weights in strategies.items():
        weights = np.array(weights)
        if abs(weights.sum() - 1) > 0.01:
            weights /= weights.sum()

        metrics = portfolio_metrics(weights, rets)
        perf_records.append({"strategy": strat_name, **metrics})

        for code, w in zip(actual_codes, weights):
            name_row = final_df[final_df["code"] == code]
            name = name_row["name"].values[0] if not name_row.empty else code
            weight_records.append({"strategy": strat_name, "code": code, "name": name, "weight": round(w * 100, 2)})

        # 위험기여도
        rc = risk_contribution(weights, cov_matrix)
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        rc_pct = rc / (port_vol + 1e-9) * 100
        for code, w, rc_val, rc_pct_val in zip(actual_codes, weights, rc, rc_pct):
            name_row = final_df[final_df["code"] == code]
            name = name_row["name"].values[0] if not name_row.empty else code
            risk_records.append({
                "strategy": strat_name,
                "code": code,
                "name": name,
                "weight_pct": round(w * 100, 2),
                "risk_contrib": round(rc_val * 100, 4),
                "risk_contrib_pct": round(rc_pct_val, 2),
            })

        print(f"  [{strat_name}] 연수익률={metrics['ann_return']:.2f}%, Sharpe={metrics['sharpe']:.3f}, MDD={metrics['mdd']:.2f}%")

    pd.DataFrame(perf_records).to_sql("advanced_portfolio_performance", con, if_exists="replace", index=False)
    pd.DataFrame(weight_records).to_sql("advanced_portfolio_weights", con, if_exists="replace", index=False)
    pd.DataFrame(risk_records).to_sql("portfolio_risk_contribution", con, if_exists="replace", index=False)

    print("\n[완료] advanced_portfolio_performance, weights, risk_contribution 저장")
    con.close()
    print("\n[08] 고도화 포트폴리오 완료\n")


if __name__ == "__main__":
    main()
