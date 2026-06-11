"""
run_pipeline.py
전체 분석 파이프라인을 순서대로 실행합니다.
"""
import subprocess
import sys
import os
import time

SCRIPTS = [
    ("pipeline/01_data_collection.py", "데이터 수집 (가격 + KOSPI 실제)"),
    ("pipeline/02_stock_scoring.py", "종목 스코어링"),
    ("pipeline/01b_fundamental_data.py", "실제 재무 데이터 수집 (PER/PBR/ROE)"),
    ("pipeline/03_portfolio_backtest.py", "포트폴리오 백테스트"),
    ("pipeline/06_market_regime.py", "시장 국면 판단"),
    ("pipeline/07_report_consensus.py", "리포트/컨센서스 팩터"),
    ("pipeline/09_ml_prediction.py", "ML 예측"),
    ("pipeline/08_advanced_portfolio.py", "고도화 포트폴리오"),
    ("pipeline/04_ai_report.py", "AI 리포트 생성"),
]


def run_script(path, name):
    print(f"\n{'='*60}")
    print(f"▶ {name} 실행 중...")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(
        [sys.executable, path],
        capture_output=False,
    )
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"✓ {name} 완료 ({elapsed:.1f}초)")
        return True
    else:
        print(f"✗ {name} 실패 (종료코드: {result.returncode})")
        return False


def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("  AI 주식 리서치 에이전트 - 전체 파이프라인 실행")
    print("=" * 60)

    success_count = 0
    for script_path, name in SCRIPTS:
        ok = run_script(script_path, name)
        if ok:
            success_count += 1

    print(f"\n{'='*60}")
    print(f"파이프라인 완료: {success_count}/{len(SCRIPTS)} 성공")
    print(f"{'='*60}")

    if success_count == len(SCRIPTS):
        print("\n✓ 모든 단계 완료. 아래 명령어로 대시보드를 실행하세요:")
        print("\n  streamlit run app.py --server.port 8502\n")
    else:
        print("\n일부 단계가 실패했습니다. 로그를 확인하세요.")


if __name__ == "__main__":
    main()
