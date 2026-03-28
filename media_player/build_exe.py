"""
EXE 빌드 스크립트
실행: python build_exe.py

PyInstaller가 없으면 자동 설치 후 빌드합니다.
VLC가 설치된 경우 libvlc.dll / libvlccore.dll / plugins/ 를 자동으로 번들에 포함합니다.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path


# ── VLC 설치 경로 자동 탐지 ───────────────────────────────────────────────
def find_vlc_dir() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\VideoLAN\VLC"),
        Path(r"C:\Program Files (x86)\VideoLAN\VLC"),
    ]
    # 환경변수에서도 탐색
    vlc_env = os.environ.get("VLC_PATH")
    if vlc_env:
        candidates.insert(0, Path(vlc_env))

    for p in candidates:
        if p.exists() and (p / "libvlc.dll").exists():
            return p
    return None


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[빌드] PyInstaller 설치 중…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def build() -> None:
    ensure_pyinstaller()

    base_dir = Path(__file__).parent
    vlc_dir  = find_vlc_dir()

    if vlc_dir is None:
        print("[경고] VLC 설치 경로를 찾지 못했습니다.")
        print("       C:\\Program Files\\VideoLAN\\VLC 에 VLC를 설치하거나")
        print("       VLC_PATH 환경변수를 설정해 주세요.")
        print("       libvlc.dll 없이 빌드를 계속합니다 (실행 시 오류 가능).")

    # ── PyInstaller 인자 ───────────────────────────────────────────────────
    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",                      # 폴더형 배포 (DLL 포함 가능)
        "--windowed",                    # 콘솔 창 숨김
        "--name", "PyPlayer",
        "--distpath", str(base_dir / "dist"),
        "--workpath", str(base_dir / "build"),
        "--specpath", str(base_dir),
    ]

    # VLC DLL + 플러그인 번들
    if vlc_dir:
        print(f"[빌드] VLC 경로 감지: {vlc_dir}")
        for dll in ["libvlc.dll", "libvlccore.dll", "axvlc.dll"]:
            dll_path = vlc_dir / dll
            if dll_path.exists():
                args += ["--add-binary", f"{dll_path};."]

        plugins_dir = vlc_dir / "plugins"
        if plugins_dir.exists():
            args += ["--add-data", f"{plugins_dir};plugins"]

    # 메인 스크립트
    args.append(str(base_dir / "main.py"))

    print("[빌드] PyInstaller 실행 중…")
    print("       (처음 실행 시 2~5분 소요될 수 있습니다)\n")
    result = subprocess.run(args, cwd=str(base_dir))

    if result.returncode == 0:
        exe_path = base_dir / "dist" / "PyPlayer" / "PyPlayer.exe"
        print(f"\n✅  빌드 성공!")
        print(f"    실행 파일: {exe_path}")
        if exe_path.exists():
            print(f"    파일 크기: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("\n❌  빌드 실패. 위 오류 메시지를 확인해 주세요.")


if __name__ == "__main__":
    build()
