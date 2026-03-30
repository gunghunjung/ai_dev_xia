# -*- mode: python ; coding: utf-8 -*-
import os, sys

ANACONDA = r'C:\Users\HS1\anaconda3'

a = Analysis(
    ['retreat_planner_tk.py'],
    pathex=[],
    binaries=[
        (os.path.join(ANACONDA, 'DLLs', '_tkinter.pyd'),        '.'),
        (os.path.join(ANACONDA, 'Library', 'bin', 'tcl86t.dll'), '.'),
        (os.path.join(ANACONDA, 'Library', 'bin', 'tk86t.dll'),  '.'),
    ],
    datas=[
        (os.path.join(ANACONDA, 'Library', 'lib', 'tcl8.6'), 'tcl8.6'),
        (os.path.join(ANACONDA, 'Library', 'lib', 'tk8.6'),  'tk8.6'),
        (os.path.join(ANACONDA, 'Lib', 'tkinter'),           'tkinter'),
    ],
    hiddenimports=['tkinter', 'tkinter.ttk', 'tkinter.font',
                   'tkinter.filedialog', 'tkinter.messagebox', '_tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5','numpy','pandas','scipy','matplotlib','PIL',
              'cv2','sklearn','torch','tensorflow','IPython',
              'PySide2','PySide6','PyQt6','cryptography',
              'pyarrow','h5py','numba','sympy'],
    noarchive=False,
    optimize=2,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='야유회기획도우미',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
