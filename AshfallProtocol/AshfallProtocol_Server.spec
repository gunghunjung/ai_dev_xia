# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec – AshfallProtocol SERVER (graphical server manager)
# Build: pyinstaller AshfallProtocol_Server.spec --clean

a = Analysis(
    ['server_gui.py'],
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
        # network package
        'network',
        'network.protocol',
        'network.server',
        'network.client',
        'network.discovery',
        # game logic (imported by GameServer)
        'game',
        'player',
        'enemy',
        'weapon',
        'bullet',
        'effects',
        'ui',
        'config',
        'asset_loader',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # client-only modules not needed in server exe
        'menu',
        'lobby',
        'mp_game',
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
    name='AshfallProtocol_Server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI window – no cmd console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
