"""
scheduler.py
매일 장 마감 후 파이프라인을 자동으로 실행합니다.

실행 방법:
    python scheduler.py            # 포그라운드 실행
    nohup python scheduler.py &    # 백그라운드 실행 (로그: data/scheduler.log)

동작:
    - 평일(월~금) 16:10 KST에 run_pipeline.py 전체 실행
    - 실행 로그는 data/scheduler.log에 저장
    - 첫 실행 시 즉시 파이프라인을 한 번 실행 (--run-now 옵션)

의존성:
    pip install schedule
"""
import os
import sys
import time
import logging
import subprocess
import schedule
from datetime import datetime

LOG_PATH = os.path.join("data", "scheduler.log")
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("=== 파이프라인 자동 실행 시작 ===")
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "run_pipeline.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            logger.info(f"파이프라인 완료 ({elapsed:.0f}초)")
            push_db_to_github()
        else:
            logger.error(f"파이프라인 실패 (종료코드: {result.returncode}, {elapsed:.0f}초)")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    logger.error(f"  stderr: {line}")
    except Exception as e:
        logger.error(f"파이프라인 실행 오류: {e}")


def push_db_to_github():
    logger.info("=== GitHub 자동 푸시 시작 ===")
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        cmds = [
            ["git", "add", "data/quant_project.db"],
            ["git", "commit", "-m", f"데이터 자동 갱신: {today}"],
            ["git", "push"],
        ]
        for cmd in cmds:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                # commit은 변경사항 없으면 실패해도 정상
                if "nothing to commit" in r.stdout + r.stderr:
                    logger.info("변경사항 없음 — 푸시 건너뜀")
                    return
                logger.error(f"git 오류 ({' '.join(cmd)}): {r.stderr.strip()}")
                return
        logger.info(f"GitHub 푸시 완료 → Streamlit Cloud 자동 재배포 시작")
    except Exception as e:
        logger.error(f"GitHub 푸시 오류: {e}")


def is_weekday():
    return datetime.now().weekday() < 5  # 0=월 ... 4=금


def scheduled_job():
    if is_weekday():
        run_pipeline()
    else:
        logger.info("주말 — 파이프라인 건너뜀")


def main():
    run_now = "--run-now" in sys.argv

    logger.info("스케줄러 시작 (평일 16:10 KST 실행)")

    if run_now:
        logger.info("--run-now 옵션: 즉시 파이프라인 실행")
        run_pipeline()

    schedule.every().day.at("16:10").do(scheduled_job)

    logger.info("다음 실행 예정: " + str(schedule.next_run()))

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
