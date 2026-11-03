"""Full scripted playthrough of the first-playable checklist through the real App."""
import os
import subprocess
import sys


def test_scripted_playthrough(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONIOENCODING="utf-8",
               WASTELAND_NO_PAD="1")
    r = subprocess.run([sys.executable, os.path.join(root, "tests", "playtest_bot.py"), str(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
