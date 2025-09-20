# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('mirix/prompts/personas', 'mirix/prompts/personas'),
        ('mirix/prompts/system', 'mirix/prompts/system'),
        ('mirix/functions/function_sets', 'mirix/functions/function_sets'),
        ('mirix/configs', 'mirix/configs'),
    ],
    hiddenimports=[
        'mirix.functions.function_sets.base',
        'mirix.functions.function_sets.memory_tools',
        'mirix.functions.function_sets.extras',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
