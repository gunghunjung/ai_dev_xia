"""PROJECT WASTELAND entry point."""
import logging
import os
import sys
import traceback

from config import SAVE_DIR, GAME_TITLE, GAME_VERSION


def _setup_logging():
    os.makedirs(SAVE_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(os.path.join(SAVE_DIR, "game.log"), encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )


def main():
    _setup_logging()
    logging.info("%s %s starting", GAME_TITLE, GAME_VERSION)
    try:
        from game.app import App
        App().run()
    except Exception:
        crash = os.path.join(SAVE_DIR, "crash.log")
        with open(crash, "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        logging.exception("Fatal error — details written to %s", crash)
        raise


if __name__ == "__main__":
    main()
