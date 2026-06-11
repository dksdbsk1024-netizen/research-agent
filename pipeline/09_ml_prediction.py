"""
09_ml_prediction.py
ML 모델 앙상블로 KOSPI 초과수익 상승확률 예측 → SQLite 저장

모델 선택 근거:
  LogisticRegression  - 선형 기준 모델. 과적합 위험 낮고 해석 용이. 가격/모멘텀 선형 패턴 포착
  RandomForest        - 배깅 앙상블. 비선형 패턴 포착, 특성 중요도 제공, 분산 감소
  GradientBoosting    - 부스팅 앙상블. 순차적 오류 보정, 복잡한 비선형 패턴 포착
  ExtraTrees          - 극도 무작위화 트리. RF 대비 빠르고 노이즈에 강건, 주가 노이즈 처리에 유리

최종 AI 점수는 4개 모델 상승확률 평균을 사용합니다.
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/quant_project.db"
FORWARD_DAYS = 20  # 예측 기간 (거래일)
MIN_SAMPLES = 100


def get_db():
    return sqlite3.connect(DB_PATH)


def build_features(price_df, bm_df, code):
    price_df["date"] = pd.to_datetime(price_df["date"])
    bm_df["date"] = pd.to_datetime(bm_df["date"])

    sub = price_df[price_df["code"] == code].set_index("date").sort_index()
    bm = bm_df.set_index("date").sort_index()

    close = sub["close"]
    volume = sub["volume"]
    bm_close = bm["close"]

    common_idx = close.index.intersection(bm_close.index)
    if len(common_idx) < MIN_SAMPLES + FORWARD_DAYS:
        return None, None

    close = close.reindex(common_idx)
    bm_close = bm_close.reindex(common_idx)
    volume = volume.reindex(common_idx).fillna(0)

    rets = close.pct_change()
    bm_rets = bm_close.pct_change()
    excess = rets - bm_rets

    # 피처 생성
    features = pd.DataFrame(index=common_idx)
    for n in [5, 10, 20, 60]:
        features[f"ret_{n}d"] = close.pct_change(n)
        features[f"vol_{n}d"] = rets.rolling(n).std()

    features["bm_ret_20d"] = bm_close.pct_change(20)
    features["excess_ret_20d"] = excess.rolling(20).sum()
    features["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)

    ema5 = close.ewm(span=5).mean()
    ema20 = close.ewm(span=20).mean()
    features["ema_cross"] = (ema5 - ema20) / (ema20 + 1e-9)
    features["price_vs_ma60"] = (close / close.rolling(60).mean() - 1)

    # 타겟: 이후 FORWARD_DAYS 동안 KOSPI 초과수익 양수면 1
    future_excess = excess.shift(-FORWARD_DAYS).rolling(FORWARD_DAYS).sum()
    target = (future_excess > 0).astype(int)

    df = features.join(target.rename("target")).dropna()
    df = df.iloc[:-FORWARD_DAYS]  # 미래 데이터 누수 방지

    if len(df) < MIN_SAMPLES:
        return None, None

    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def train_and_predict(X, y, code):
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.transform(X)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=0.1),
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=50, max_depth=4, random_state=42),
    }

    results = []
    for model_name, model in models.items():
        try:
            if model_name == "LogisticRegression":
                model.fit(X_train_s, y_train)
                pred = model.predict(X_test_s)
                prob = model.predict_proba(X_test_s)[:, 1]
                latest_prob = model.predict_proba(X_all_s[-1:])[:, 1][0]
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)[:, 1]
                latest_prob = model.predict_proba(X.iloc[-1:])[:, 1][0]

            acc = accuracy_score(y_test, pred)
            try:
                auc = roc_auc_score(y_test, prob)
            except Exception:
                auc = 0.5

            results.append({
                "code": code,
                "model": model_name,
                "accuracy": round(acc, 4),
                "roc_auc": round(auc, 4),
                "upside_prob": round(latest_prob, 4),
                "train_size": len(X_train),
                "test_size": len(X_test),
            })
        except Exception as e:
            pass

    return results if results else None


def compute_final_ai_score(enhanced_df, ml_results_df):
    # 모델별 상승확률 평균
    avg_prob = ml_results_df.groupby("code")["upside_prob"].mean().reset_index()
    avg_prob.columns = ["code", "ml_upside_prob"]

    # ML 위험 점수 (낮은 상승확률 = 높은 위험)
    avg_prob["ml_risk_score"] = 1 - avg_prob["ml_upside_prob"]

    merged = enhanced_df.merge(avg_prob, on="code", how="left")
    merged["ml_upside_prob"] = merged["ml_upside_prob"].fillna(0.5)
    merged["ml_risk_score"] = merged["ml_risk_score"].fillna(0.5)

    # ML 점수 정규화 (0~100)
    mn, mx = merged["ml_upside_prob"].min(), merged["ml_upside_prob"].max()
    merged["ml_score_norm"] = (merged["ml_upside_prob"] - mn) / (mx - mn + 1e-9) * 100

    risk_mn, risk_mx = merged["ml_risk_score"].min(), merged["ml_risk_score"].max()
    merged["risk_defense_score"] = 100 - (merged["ml_risk_score"] - risk_mn) / (risk_mx - risk_mn + 1e-9) * 100

    # 최종 AI 점수: 리서치 강화 70% + ML 20% + 위험방어 10%
    merged["final_ai_score"] = (
        merged["research_enhanced_score"] * 0.70
        + merged["ml_score_norm"] * 0.20
        + merged["risk_defense_score"] * 0.10
    )
    merged["final_ai_rank"] = merged["final_ai_score"].rank(ascending=False).astype(int)

    return merged


def main():
    print("=" * 60)
    print("09. ML 예측 점수 생성 시작 (4개 모델 앙상블)")
    print("=" * 60)

    con = get_db()
    price_df = pd.read_sql("SELECT * FROM price_data", con)
    bm_df = pd.read_sql("SELECT * FROM benchmark_price", con)
    enhanced_df = pd.read_sql("SELECT * FROM score_result_research_enhanced_all", con)

    all_ml_results = []
    codes = enhanced_df["code"].tolist()

    for code in codes:
        X, y = build_features(price_df, bm_df, code)
        if X is None:
            print(f"  [{code}] 데이터 부족, 건너뜀")
            continue
        results = train_and_predict(X, y, code)
        if results:
            all_ml_results.extend(results)
            prob_mean = np.mean([r["upside_prob"] for r in results])
            print(f"  [{code}] 모델 {len(results)}개 완료, 평균 상승확률: {prob_mean:.3f}")

    if not all_ml_results:
        print("[경고] ML 결과 없음. 퀀트 점수 기반 heuristic 사용")
        heuristic = []
        for _, row in enhanced_df.iterrows():
            heuristic.append({
                "code": row["code"],
                "model": "Heuristic",
                "accuracy": 0.5,
                "roc_auc": 0.5,
                "upside_prob": row["quant_score"] / 100,
                "train_size": 0,
                "test_size": 0,
            })
        all_ml_results = heuristic

    ml_df = pd.DataFrame(all_ml_results)
    ml_df.to_sql("ml_prediction_score_by_model_all", con, if_exists="replace", index=False)
    print(f"\n[완료] ml_prediction_score_by_model_all: {len(ml_df)}행 저장")

    # 최종 AI 점수 계산
    final_df = compute_final_ai_score(enhanced_df, ml_df)
    final_df.to_sql("score_result_final_ai_by_model_all", con, if_exists="replace", index=False)
    print(f"[완료] score_result_final_ai_by_model_all 저장")

    print("\n최종 AI 점수 TOP 10:")
    top10 = final_df.nsmallest(10, "final_ai_rank")[
        ["final_ai_rank", "code", "name", "quant_score", "research_enhanced_score", "ml_upside_prob", "final_ai_score"]
    ]
    print(top10.to_string(index=False))

    con.close()
    print("\n[09] ML 예측 완료\n")


if __name__ == "__main__":
    main()
