# -*- mode: python ; coding: utf-8 -*-
# Portable one-folder build. From project root:
#   pyinstaller --clean build_portable.spec
#
# Output: dist/MyAppPortable/MyApp.exe (+ _internal)
# Copy the whole dist/MyAppPortable folder to USB; runtime data lives next to the exe.

import os

block_cipher = None

ROOT = os.path.abspath(os.path.dirname(SPEC))

a = Analysis(
    [os.path.join(ROOT, "launcher.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (os.path.join(ROOT, "config", "default.yaml"), "config"),
        (os.path.join(ROOT, "assets"), "assets"),
    ],
    hiddenimports=[
        "src.bootstrap",
        "src.main",
        "src.app_paths",
        "src.session_runner",
        "analyze_session",
        "calibrate",
        "keyboard",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MyApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MyAppPortable",
)
