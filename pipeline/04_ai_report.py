"""
04_ai_report.py
Evidence JSON 기반 자동 리포트 생성 → SQLite 저장
API Key 있으면 OpenAI 생성형 리포트, 없으면 템플릿 리포트
"""
import sqlite3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/quant_project.db"
DEFAULT_TOP_N = 5


def get_db():
    return sqlite3.connect(DB_PATH)


def build_evidence(final_df, regime_df, portfolio_df, consensus_df, top_n=None):
    """
    top_n: 종목 리포트에 포함할 종목 수. None이면 DEFAULT_TOP_N(5) 사용.
    """
    n = top_n if top_n is not None else DEFAULT_TOP_N
    regime = regime_df.iloc[0].to_dict() if not regime_df.empty else {}
    top_stocks = final_df.nsmallest(n, "final_ai_rank") if not final_df.empty else pd.DataFrame()

    evidence = {
        "report_date": datetime.today().strftime("%Y-%m-%d"),
        "analyzed_top_n": n,
        "market_regime": {
            "kospi_close": regime.get("kospi_close", "N/A"),
            "regime": regime.get("regime", "N/A"),
            "total_score": regime.get("total_score", "N/A"),
            "ret_1m": regime.get("ret_1m", "N/A"),
            "ret_3m": regime.get("ret_3m", "N/A"),
            "vol_20": regime.get("vol_20", "N/A"),
            "mdd_1y": regime.get("mdd_1y", "N/A"),
            "above_ma200": bool(regime.get("above_ma200", 0)),
            "strategy_rec": regime.get("strategy_rec", "N/A"),
        },
        "top_stocks": [],
        "portfolio_summary": [],
        "note": "본 리포트는 교육 및 분석 목적의 참고자료이며 투자 권유가 아닙니다.",
        "data_source": "NAVER 금융 + DART OpenAPI + FnGuide 컨센서스 기반",
    }

    for _, row in top_stocks.iterrows():
        cons_row = consensus_df[consensus_df["code"] == row["code"]] if not consensus_df.empty else pd.DataFrame()
        stock_ev = {
            "rank": int(row.get("final_ai_rank", 0)),
            "code": row["code"],
            "name": row.get("name", ""),
            "final_ai_score": round(float(row.get("final_ai_score", 0)), 2),
            "quant_score": round(float(row.get("quant_score", 0)), 2),
            "research_score": round(float(row.get("research_score", 50)), 2),
            "ml_upside_prob": round(float(row.get("ml_upside_prob", 0.5)), 4),
            "ret_3m": round(float(row.get("ret_3m", 0)), 2),
            "ret_6m": round(float(row.get("ret_6m", 0)), 2),
            "momentum_score": round(float(row.get("momentum_score", 0)), 2),
            "stability_score": round(float(row.get("stability_score", 0)), 2),
        }
        if not cons_row.empty:
            stock_ev["opinion"] = cons_row.iloc[0].get("opinion", "N/A")
            stock_ev["upside_ratio"] = round(float(cons_row.iloc[0].get("upside_ratio", 0)), 2)
            stock_ev["data_type"] = cons_row.iloc[0].get("data_type", "proxy")
        else:
            stock_ev["opinion"] = "N/A"
            stock_ev["upside_ratio"] = 0.0
            stock_ev["data_type"] = "N/A"
        evidence["top_stocks"].append(stock_ev)

    if not portfolio_df.empty:
        for _, row in portfolio_df.iterrows():
            evidence["portfolio_summary"].append({
                "strategy": row.get("strategy", ""),
                "ann_return": round(float(row.get("ann_return", 0)), 2),
                "sharpe": round(float(row.get("sharpe", 0)), 3),
                "mdd": round(float(row.get("mdd", 0)), 2),
            })

    return evidence


def generate_template_report(evidence):
    regime = evidence["market_regime"]
    top = evidence["top_stocks"]
    portfolio = evidence["portfolio_summary"]
    date = evidence["report_date"]

    lines = [
        f"# AI 주식 리서치 리포트",
        f"작성일: {date}",
        "",
        "---",
        "",
        "## 1. 시장 국면 브리프",
        "",
        f"**현재 KOSPI**: {regime.get('kospi_close', 'N/A'):,}pt",
        f"**국면 판단**: {regime.get('regime', 'N/A')} (종합점수: {regime.get('total_score', 'N/A'):.1f}/100)",
        f"**최근 1개월 수익률**: {regime.get('ret_1m', 'N/A'):.2f}%",
        f"**최근 3개월 수익률**: {regime.get('ret_3m', 'N/A'):.2f}%",
        f"**20일 변동성**: {regime.get('vol_20', 'N/A'):.2f}%",
        f"**1년 MDD**: {regime.get('mdd_1y', 'N/A'):.2f}%",
        f"**200일선 대비**: {'위 (상승 추세)' if regime.get('above_ma200') else '아래 (주의 필요)'}",
        "",
        f"**추천 전략**: {regime.get('strategy_rec', 'N/A')}",
        "",
        "---",
        "",
        "## 2. AI 점수 상위 종목",
        "",
    ]

    for s in top:
        lines += [
            f"### {s['rank']}위. {s['name']} ({s['code']})",
            f"- 최종 AI 점수: **{s['final_ai_score']:.1f}점**",
            f"- 퀀트 점수: {s['quant_score']:.1f} | 리서치 점수: {s['research_score']:.1f} | ML 상승확률: {s.get('ml_upside_prob', 0.5)*100:.1f}%",
            f"- 3M 수익률: {s['ret_3m']:.2f}% | 6M 수익률: {s['ret_6m']:.2f}%",
            f"- 모멘텀 점수: {s['momentum_score']:.1f} | 안정성 점수: {s['stability_score']:.1f}",
            f"- 투자의견(proxy): {s.get('opinion', 'N/A')} | 목표가 상승여력: {s.get('upside_ratio', 0):.1f}%",
            f"- *데이터: {s.get('data_type', 'proxy')} 기반*",
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. 포트폴리오 전략 비교",
        "",
        "| 전략 | 연수익률 | Sharpe | MDD |",
        "|------|---------|--------|-----|",
    ]
    for p in portfolio:
        lines.append(f"| {p['strategy']} | {p['ann_return']:.2f}% | {p['sharpe']:.3f} | {p['mdd']:.2f}% |")

    lines += [
        "",
        "---",
        "",
        f"> {evidence.get('note', '')}",
        f"> 데이터 출처: {evidence.get('data_source', '')}",
    ]

    return "\n".join(lines)


def generate_openai_report(evidence, api_key):
    try:
        import requests as _req

        system_msg = (
            "당신은 10년 경력의 한국 주식 시장 전문 애널리스트입니다. "
            "퀀트 모델, 시장 국면 분석, 포트폴리오 전략에 능통하며, "
            "수치 데이터를 단순 나열하는 것이 아니라 그 의미를 해석하고 "
            "투자자에게 실질적인 시사점을 제공하는 리서치 리포트를 작성합니다. "
            "리포트는 구체적인 수치를 인용하되, 각 수치가 의미하는 바와 "
            "다른 지표와의 관계를 설명해야 합니다. "
            "본 리포트는 교육 및 분석 목적의 참고자료이며 투자 권유가 아닙니다."
        )

        user_msg = (
            "아래 Evidence 데이터를 바탕으로 전문적인 한국 주식 리서치 리포트를 작성해주세요.\n\n"
            "## 작성 지침\n"
            "- 각 섹션은 수치를 인용하고, 그 수치가 시사하는 바를 2~4문장으로 해석하세요.\n"
            "- 종목 간 상대적 비교(강점·약점)를 포함하세요.\n"
            "- 현재 시장 국면이 각 종목·전략에 미치는 영향을 설명하세요.\n"
            "- 주목해야 할 리스크 요인을 구체적으로 서술하세요.\n"
            "- Evidence에 없는 기업 내부 정보나 뉴스는 추가하지 마세요.\n"
            "- 실제 컨센서스(data_type=real)와 proxy 데이터를 구분해 신뢰도를 명시하세요.\n\n"
            "## 리포트 구성 (마크다운 형식)\n"
            "1. **Executive Summary** — 시장 국면과 핵심 투자 판단 3~5문장\n"
            "2. **시장 국면 분석** — 국면 판단 근거(모멘텀·변동성·MDD·200일선) 해석 및 전략적 시사점\n"
            "3. **AI 추천 종목 심층 분석** — 상위 종목별로: 점수 해석, 퀀트·리서치·ML 관점 종합, "
            "컨센서스와의 정합성, 주목 포인트 및 리스크\n"
            "4. **포트폴리오 전략 평가** — 각 전략의 성과 해석, 현 국면에서의 적합성, 추천 전략과 이유\n"
            "5. **핵심 리스크 및 유의사항** — 모델 한계, 데이터 신뢰도, 시장 리스크 요인\n\n"
            "## Evidence\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2)
        )

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            "max_tokens": 4000,
            "temperature": 0.6,
        }
        resp = _req.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[OpenAI 오류: {e}]\n\n" + generate_template_report(evidence)


def main():
    print("=" * 60)
    print("04. AI 리포트 생성 시작")
    print("=" * 60)

    con = get_db()

    def safe_read(table):
        try:
            return pd.read_sql(f"SELECT * FROM {table}", con)
        except Exception:
            return pd.DataFrame()

    final_df = safe_read("score_result_final_ai_by_model_all")
    regime_df = safe_read("market_regime")
    portfolio_df = safe_read("advanced_portfolio_performance")
    consensus_df = safe_read("research_consensus_factor_all")

    evidence = build_evidence(final_df, regime_df, portfolio_df, consensus_df)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        print("OpenAI API 키 감지 → 생성형 리포트 사용")
        report_text = generate_openai_report(evidence, api_key)
        report_type = "openai"
    else:
        print("API 키 없음 → 템플릿 리포트 사용")
        report_text = generate_template_report(evidence)
        report_type = "template"

    report_df = pd.DataFrame([{
        "report_date": evidence["report_date"],
        "report_type": report_type,
        "evidence_json": json.dumps(evidence, ensure_ascii=False),
        "report_text": report_text,
    }])
    report_df.to_sql("ai_report_result_final_enhanced", con, if_exists="replace", index=False)
    print(f"[완료] ai_report_result_final_enhanced 저장 (유형: {report_type})")

    con.close()
    print("\n[04] AI 리포트 생성 완료\n")


if __name__ == "__main__":
    main()
