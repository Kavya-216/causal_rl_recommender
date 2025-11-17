from pathlib import Path
import logging
from src.preprocessing.mind_preprocess import main as preprocess_main


logger = logging.getLogger("auto_clean")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def run_auto_clean(force: bool = False):
    try:
        preprocess_main(force=force)
        logger.info("Auto-clean completed successfully")
    except Exception as e:
        logger.error(f"Auto-clean failed: {e}")
        raise


if __name__ == "__main__":
    run_auto_clean()
