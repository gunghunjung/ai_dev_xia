# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec – AshfallProtocol CLIENT (game window)
# Build: pyinstaller AshfallProtocol_Client.spec --clean

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # pygame internals
        'pygame',
        'pygame.font',
        'pygame.mixer',
        'pygame.math',
        'pygame._sdl2',
        # network package (subpackage, needs explicit listing)
        'network',
        'network.protocol',
        'network.server',
        'network.client',
        'network.discovery',
        # game modules (imported lazily inside functions)
        'game',
        'player',
        'enemy',
        'weapon',
        'bullet',
        'effects',
        'ui',
        'config',
        'asset_loader',
        'menu',
        'lobby',
        'mp_game',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # keep the bundle lean – server GUI not needed in client exe
        'server_gui',
        'server_standalone',
        'tkinter',
        'unittest',
        'test',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AshfallProtocol',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # no black cmd window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
