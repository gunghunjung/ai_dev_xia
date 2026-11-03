# PyInstaller spec for PROJECT WASTELAND (Windows, one-folder build).
# Build:  .venv\Scripts\pyinstaller build.spec --noconfirm
# Output: dist\ProjectWasteland\ProjectWasteland.exe  (saves are written next to the exe)
import os

ROOT = os.path.abspath(".")

a = Analysis(
    ["main.py"],
    pathex=[ROOT],
    datas=[("data", "data"), ("assets", "assets")],
    hiddenimports=["pygame._sdl2.controller"],
    excludes=["tkinter", "pytest", "PIL.ImageQt"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ProjectWasteland",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ProjectWasteland", upx=False)
