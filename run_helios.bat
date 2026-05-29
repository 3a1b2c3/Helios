@echo off
:: Windows launcher for PKU-YuanGroup/Helios -- one-shot setup + model download + example run.
:: Mirrors the FastVideo run_matrixgame3.bat / run_cosmos.bat pattern.
::
:: Usage:
::   run_helios.bat                          DEFAULT: download + run (skip setup)
::   run_helios.bat --setup                  also build the venv + install deps
::   run_helios.bat --skip-run               download only, don't run inference
::   run_helios.bat --setup                  full flow (setup + download + run)
::   run_helios.bat --mode i2v               image-to-video (uses example/wave.jpg)
::   run_helios.bat --skip-download          skip download (assume cached)
::   run_helios.bat --low-vram               enable group offloading (default; needed on single 5090 / 32 GB)
::   run_helios.bat --high-vram              disable group offloading (only on >32 GB VRAM)
::
:: Only the distilled variant is supported here (it's the only one we care about
:: for 5090 inference, and base/mid are ~138 GB each — not worth caching).
::
:: Env overrides:
::   HELIOS_VENV=C:\path\to\.venv            default: %~dp0.venv
::   HF_HOME=C:\path\to\hf-cache             default: %USERPROFILE%\.cache\huggingface

:: --- log capture wrapper: on first invocation, re-exec ourselves with full
:: stdout/stderr redirected to logs\helios_<timestamp>.log so every phase
:: (setup install lines, pre-fetch download, inference) lands in one file.
:: Tail in another terminal:  Get-Content -Wait "<path printed below>"
if not defined HELIOS_LOG_REENTRY (
    setlocal EnableExtensions EnableDelayedExpansion
    if not exist "%~dp0logs" mkdir "%~dp0logs"
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value ^| findstr "="') do set "LDT=%%I"
    set "HELIOS_LOG=%~dp0logs\helios_!LDT:~0,8!_!LDT:~8,6!.log"
    echo Helios run logging to: !HELIOS_LOG!
    echo Tail in another terminal:  Get-Content -Wait "!HELIOS_LOG!"
    echo.
    set "HELIOS_LOG_REENTRY=1"
    call "%~f0" %* > "!HELIOS_LOG!" 2>&1
    set "EXIT=!ERRORLEVEL!"
    echo.
    echo --- last 40 log lines ---
    powershell -NoProfile -Command "Get-Content -LiteralPath '!HELIOS_LOG!' -Tail 40"
    echo.
    echo Full log: !HELIOS_LOG!
    endlocal & exit /b %EXIT%
)

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

:: --- arg parse ---
:: Default: download + run. --setup opts the setup phase back in; --skip-run
:: opts the run phase out. Legacy --run flag kept as a no-op so existing
:: command lines don't break.
set "MODE=i2v"
set "SKIP_SETUP=1"
set "SKIP_DOWNLOAD=0"
set "SKIP_RUN=0"
:: Default to --low-vram: 5090 (32 GB) OOMs on Helios-14B at default settings.
:: Pass --high-vram to opt out on hardware with more headroom.
set "LOW_VRAM=1"
:parse
if "%~1"=="" goto args_done
if /I "%~1"=="--mode"          ( set "MODE=%~2" & shift & shift & goto parse )
if /I "%~1"=="--setup"         ( set "SKIP_SETUP=0" & shift & goto parse )
if /I "%~1"=="--run"           ( set "SKIP_RUN=0" & shift & goto parse )
if /I "%~1"=="--skip-setup"    ( set "SKIP_SETUP=1" & shift & goto parse )
if /I "%~1"=="--skip-download" ( set "SKIP_DOWNLOAD=1" & shift & goto parse )
if /I "%~1"=="--skip-run"      ( set "SKIP_RUN=1" & shift & goto parse )
if /I "%~1"=="--low-vram"      ( set "LOW_VRAM=1" & shift & goto parse )
if /I "%~1"=="--high-vram"     ( set "LOW_VRAM=0" & shift & goto parse )
if /I "%~1"=="--help"          goto :help
if /I "%~1"=="-h"              goto :help
echo ERROR: unknown arg %~1
exit /b 2
:args_done

:: Distilled variant only — best efficiency on 5090, ~80 GB after filtering
:: out the redundant transformer_ode mirror dir.
set "VARIANT=distilled"
set "HF_REPO=BestWishYsh/Helios-Distilled"

:: --- paths ---
if not defined HELIOS_VENV set "HELIOS_VENV=%~dp0.venv"
set "VENV_PY=!HELIOS_VENV!\Scripts\python.exe"
set "ENTRY=%~dp0infer_helios.py"

:: uv on Windows typically installs to %USERPROFILE%\.local\bin which is on
:: PowerShell's PATH but not cmd's. Prepend it so `where uv` finds it from cmd.
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv.exe not on PATH. Install uv first: https://docs.astral.sh/uv/
    echo Tried: %%USERPROFILE%%\.local\bin
    exit /b 2
)
for /f "delims=" %%U in ('where uv') do set "UV_EXE=%%U" & goto :uv_found
:uv_found

:: --- Strip ambient venv state so spawned interpreter doesn't graft another
::     venv's stdlib path (the SRE / _sre mismatch). ---
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

echo ============================================================
echo Helios real-time long video generation
echo ============================================================
echo   venv     : !HELIOS_VENV!
echo   mode     : %MODE%
echo   variant  : distilled  ^(repo: !HF_REPO!^)
echo   low-vram : %LOW_VRAM%  ^(group offload on if 1^)
echo   skip     : setup=%SKIP_SETUP%  download=%SKIP_DOWNLOAD%  run=%SKIP_RUN%
echo ============================================================
echo.

:: ============================================================
:: 1/3  setup venv + deps
:: ============================================================
if "%SKIP_SETUP%"=="1" goto :phase_download

if not exist "!VENV_PY!" (
    echo --- creating venv at !HELIOS_VENV! ^(Python 3.11^) ---
    "!UV_EXE!" venv "!HELIOS_VENV!" --python 3.11
    if errorlevel 1 ( echo ERROR: uv venv failed & exit /b 1 )
)

echo --- installing torch 2.10.0 + cu128 ---
"!UV_EXE!" pip install --python "!VENV_PY!" torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 ( echo ERROR: torch install failed & exit /b 1 )

:: triton-windows BEFORE the requirements pass so triton is already satisfied
:: when uv reads `triton==3.6.0` from requirements.txt (we filter that line
:: out below, but installing triton-windows first also blocks any transitive
:: triton install from trying the Linux PyPI build). Pinning to a recent
:: triton-windows tracking torch 2.10 -- let uv pick the matching wheel.
echo --- installing triton-windows ^(replaces Linux triton on Windows^) ---
"!UV_EXE!" pip install --python "!VENV_PY!" triton-windows
if errorlevel 1 ( echo WARN: triton-windows install failed -- continuing, torch.compile may not work )

:: sageattention is universal py3-none-any on PyPI; kernels JIT-compile through
:: Triton for sm_120 at first call. Diffusers exposes it via backend name "_sage".
:: Cheap to install (no compile) and gives ~FA2-class perf without the wheel
:: pinning headaches of flash-attn.
echo --- installing sageattention ^(Triton-JIT attention; needs triton-windows^) ---
"!UV_EXE!" pip install --python "!VENV_PY!" sageattention
if errorlevel 1 ( echo WARN: sageattention install failed -- continuing, attention falls back to SDPA )

:: Build a Windows-safe requirements file by filtering out Linux-only lines
:: and the triton pin (which has no Windows wheel). findstr /V drops any
:: line starting with one of the listed prefixes.
set "REQS_WIN=%TEMP%\helios_reqs_win.txt"
findstr /V /B /R /C:"triton" /C:"deepspeed" /C:"mpi4py" /C:"tf_keras" /C:"tensorflow" /C:"# " "%~dp0requirements.txt" > "!REQS_WIN!"
echo --- installing Helios requirements ^(filtered for Windows^) ---
echo     filtered out: triton, deepspeed, mpi4py, tf_keras, tensorflow
echo     file: !REQS_WIN!
"!UV_EXE!" pip install --python "!VENV_PY!" -r "!REQS_WIN!"
if errorlevel 1 (
    echo WARN: requirements install had errors -- retrying with --no-deps and selective backfill
    "!UV_EXE!" pip install --python "!VENV_PY!" -r "!REQS_WIN!" --no-deps
    "!UV_EXE!" pip install --python "!VENV_PY!" kernels==0.13.0 transformers==5.3.0 sentence-transformers==5.2.3 accelerate==1.12.0 peft==0.18.1 "huggingface-hub>=1.4.1" zstandard==0.25.0 wandb==0.23.0 "numpy<2.0.0" opencv-python gradio moviepy imageio-ffmpeg ftfy Jinja2 einops packaging ninja omegaconf loguru
    if errorlevel 1 ( echo ERROR: deps install failed even after fallback & exit /b 1 )
)

echo --- installing diffusers from main ^(Helios needs ContextParallelConfig^) ---
"!UV_EXE!" pip install --python "!VENV_PY!" "git+https://github.com/huggingface/diffusers.git"
if errorlevel 1 ( echo ERROR: diffusers install failed & exit /b 1 )

:: flash-attn (Windows prebuild from mjun0812). Helios uses SDPA otherwise, so
:: this is a perf win, not a correctness requirement. Skipped unless the user
:: sets HELIOS_FLASH_ATTN_WHEEL=<url-or-path> — flash-attn is torch-minor-version
:: locked (C++ ABI), so we don't ship a default URL; pick a wheel that matches
:: your venv's torch (currently 2.10) + cu128 + cp311 + win.
::
:: Verified: torch2.8 wheels FAIL on torch2.10 with `DLL load failed while
:: importing flash_attn_2_cuda: The specified procedure could not be found.`
::
:: Wheels list: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
:: See reference_windows_wheels.md.
if defined HELIOS_FLASH_ATTN_WHEEL (
    echo --- installing flash-attn ^(Windows prebuild, sm_120^) ---
    echo     wheel: !HELIOS_FLASH_ATTN_WHEEL!
    "!UV_EXE!" pip install --python "!VENV_PY!" "!HELIOS_FLASH_ATTN_WHEEL!"
    if not errorlevel 1 (
        :: Verify it imports — pip install succeeds even when the CUDA DLL has
        :: ABI mismatches with the runtime torch. Roll back on import failure.
        "!VENV_PY!" -c "import flash_attn" 2>nul
        if errorlevel 1 (
            echo WARN: flash-attn installed but failed to import ^(C++ ABI mismatch^).
            echo       Rolling back so Helios falls back cleanly to PyTorch SDPA.
            "!UV_EXE!" pip uninstall --python "!VENV_PY!" flash-attn >nul 2>nul
        ) else (
            echo flash-attn imports OK.
        )
    ) else (
        echo WARN: flash-attn install failed -- Helios will fall back to PyTorch SDPA.
    )
) else (
    echo --- flash-attn install skipped ^(set HELIOS_FLASH_ATTN_WHEEL to enable^) ---
    echo     Helios will use PyTorch SDPA. For a perf boost, point at a wheel built
    echo     against this venv's torch ^(check `pip show torch` for the version^):
    echo       https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
)

echo --- removing leftover wandb/triton cache ---
if exist "%USERPROFILE%\.triton\cache" rmdir /s /q "%USERPROFILE%\.triton\cache"

echo Setup complete: !HELIOS_VENV!
echo.

:: ============================================================
:: 2/3  download model snapshot
:: ============================================================
:phase_download
if "%SKIP_DOWNLOAD%"=="1" goto :phase_run

echo --- pre-fetching HF snapshot: !HF_REPO! ---
echo     skipping transformer_init/ ^(fine-tune mirror, not used at inference^)
echo     including transformer_ode/ ^(needed by --is_enable_stage2^)
echo     subset ~131 GB ^(full snapshot ~138 GB; saves ~7 GB by skipping transformer_init^)
"!VENV_PY!" -c "import os; os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER','0'); from huggingface_hub import snapshot_download; p = snapshot_download('!HF_REPO!', max_workers=1, ignore_patterns=['transformer_init/*']); print('snapshot at:', p)"
if errorlevel 1 ( echo ERROR: HF download failed & exit /b 1 )
echo.

:: ============================================================
:: 3/3  run example
:: ============================================================
:phase_run
if "%SKIP_RUN%"=="1" (
    echo Run skipped ^(--skip-run^).
    exit /b 0
)

:: --- Windows env tweaks (same as run_matrixgame3.bat / run_cosmos.bat) ---
if not defined GLOO_SOCKET_IFNAME set "GLOO_SOCKET_IFNAME=Wi-Fi"
set "HF_DEACTIVATE_ASYNC_LOAD=1"
set "HF_HUB_ENABLE_HF_TRANSFER=0"
set "USE_LIBUV=0"
set "TORCH_TCPSTORE_USE_LIBUV=0"
set "PYTHONIOENCODING=utf-8"
set "CUDA_VISIBLE_DEVICES=0"

:: Mode-specific args. Defaults match scripts/inference/helios-*_*.sh.
:: 384x640 / 99 frames / 24fps -> ~4s clip, fits 32 GB 5090 with distilled
:: at bf16. Bump --num_frames for longer clips (must remain divisible by 9).
::
:: --is_enable_stage2 is REQUIRED for the Helios-Distilled checkpoint: it's a
:: pyramid/multi-stage model. Without the flag the pipeline takes the single-
:: stage branch (pipeline_helios_diffusers.py:1254) and crashes with
::   KeyError: None  at scheduling_helios_diffusers.py:227
::     stage_timesteps = self.timesteps_per_stage[stage_index]
:: because stage_index defaults to None on that call. Override stage2 off via
:: HELIOS_NO_STAGE2=1 if you ever point this at a single-stage checkpoint.
set "STAGE2_ARG=--is_enable_stage2"
if defined HELIOS_NO_STAGE2 if not "%HELIOS_NO_STAGE2%"=="0" set "STAGE2_ARG="
set "COMMON_ARGS=--base_model_path !HF_REPO! --transformer_path !HF_REPO! --weight_dtype bf16 --height 384 --width 640 --num_frames 99 --fps 24 --guidance_scale 5.0 --seed 42 --output_folder %~dp0output_helios !STAGE2_ARG!"

:: Prompts and per-mode extra args. The PROMPT variable holds the text WITHOUT
:: enclosing quotes — quoting happens at the python.exe invocation line below.
:: Putting `\"…\"` inside `set "MODE_ARGS=…"` doesn't survive: CMD strips the
:: outer quotes, leaving literal `\"…\"` that MSVC's argv parser then mangles
:: (the prompt's first word gets eaten by --prompt and the rest become unknown
:: positionals, e.g. `unrecognized arguments: vibrant tropical fish …`).
set "PROMPT="
set "MODE_ARGS="
if /I "%MODE%"=="t2v" (
    set "MODE_ARGS=--sample_type t2v"
    set "PROMPT=A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. Bright blue and yellow scales, dynamic motion, close-up shot."
)
if /I "%MODE%"=="i2v" (
    set "MODE_ARGS=--sample_type i2v --image_path %~dp0example\wave.jpg --image_noise_sigma_min 0.111 --image_noise_sigma_max 0.135"
    set "PROMPT=A towering emerald wave surges forward, its crest curling with raw power. Sunlight glints off the translucent water. Dynamic motion, cinematic shot."
)
if /I "%MODE%"=="v2v" (
    set "MODE_ARGS=--sample_type v2v --video_path %~dp0example\car.mp4"
    set "PROMPT=A red sports car driving down a winding mountain road at dusk. Cinematic, dynamic motion."
)
if not defined MODE_ARGS (
    echo ERROR: --mode must be t2v^|i2v^|v2v ^(got %MODE%^)
    exit /b 2
)

set "LOW_VRAM_ARGS="
if "%LOW_VRAM%"=="1" set "LOW_VRAM_ARGS=--enable_low_vram_mode --group_offloading_type leaf_level --num_blocks_per_group 4"

echo --- running infer_helios.py ---
echo   python   : !VENV_PY!
echo   entry    : !ENTRY!
echo   output   : %~dp0output_helios\
echo.

:: Python's -X utf8 guarantees UTF-8 stdout/stderr so the bat-level log is decodable.
"!VENV_PY!" -X utf8 "!ENTRY!" %COMMON_ARGS% %MODE_ARGS% --prompt "!PROMPT!" %LOW_VRAM_ARGS%
set "EXIT_CODE=!ERRORLEVEL!"

if not !EXIT_CODE!==0 (
    echo.
    echo ERROR: infer_helios exited with code !EXIT_CODE!
    echo Common causes:
    echo   - OOM on 5090: retry with --low-vram
    echo   - Missing CUDA toolkit: ensure CUDA 12.8 runtime is on PATH
    echo   - flash-attn / triton missing on Windows: --enable_compile drops a warning, output still ok
    exit /b !EXIT_CODE!
)

echo.
echo --- done ---
echo videos at: %~dp0output_helios\
exit /b 0

:help
echo Usage:
echo   run_helios.bat                          DEFAULT: download + run ^(distilled^)
echo   run_helios.bat --setup                  also build venv + install deps
echo   run_helios.bat --skip-run               download only, don't run inference
echo   run_helios.bat --setup                  full flow ^(setup + download + run^)
echo   run_helios.bat --mode i2v               image-to-video
echo   run_helios.bat --mode v2v               video-to-video
echo   run_helios.bat --skip-download          skip download ^(assume cached^)
echo   run_helios.bat --low-vram               group offload ^(default^)
echo   run_helios.bat --high-vram              disable group offload ^(needs ^>32 GB VRAM^)
echo.
echo Note: only the distilled variant is supported. base/mid would each cost
echo another ~138 GB ^(or ~80 GB filtered^) — not worth it for this disk.
echo.
echo Model:
echo   BestWishYsh/Helios-Distilled  ~80 GB after filter ^(DMD-distilled, 5090-friendly^)
echo   ^(transformer_ode mirror dir is skipped to save ~60 GB^)
echo.
echo Output: %~dp0output_helios\
exit /b 0
