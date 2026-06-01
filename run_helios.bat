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
    REM wmic was deprecated in Windows 11 23H2 and removed in many recent
    REM installs -- when it returns nothing the LDT var stays empty and the
    REM log filename comes out as literal "helios_~0,8LDT:~8,6.log".
    REM PowerShell's Get-Date is available on every supported Windows.
    for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "_TS=%%T"
    if not defined _TS set "_TS=unknown"
    set "HELIOS_LOG=%~dp0logs\helios_!_TS!.log"
    echo Helios run logging to: !HELIOS_LOG!
    echo Tail in another terminal:  Get-Content -Wait "!HELIOS_LOG!"
    echo.
    set "HELIOS_LOG_REENTRY=1"
    REM Tee stdout+stderr to console AND log file. The 2>&1 happens at cmd level
    REM BEFORE the pipe to PowerShell, so PS 5.1 doesn't wrap each stderr write
    REM as a NativeCommandError ErrorRecord (the red-text noise you'd otherwise
    REM see at startup). Trade-off: cmd's pipe doesn't propagate the inner bat's
    REM exit code through to ERRORLEVEL -- it reflects PowerShell's exit
    REM instead -- so the actual python error code surfaces only inside the log.
    call "%~f0" %* 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -Command "$input | Tee-Object -FilePath '!HELIOS_LOG!'"
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
:: Render dimensions. Defaults match Helios's distilled-checkpoint training-time
:: small preset (fast iteration); override via --height N --width N for higher
:: quality (e.g. 720 x 1280 matches the training resolution and noticeably
:: improves output sharpness, at the cost of more VRAM and longer steps).
set "HELIOS_HEIGHT=384"
set "HELIOS_WIDTH=640"
:: num_frames must be a multiple of 33 -- Helios processes video in 33-frame
:: chunks (latent chunk size 9 * vae_scale_factor_temporal 4 + 1 = 33). The
:: README's adjusted-frame table only lists 33*N values; passing 99 yields a
:: ~4-second clip at 24 fps. See validation block below for the constraint.
set "HELIOS_NUM_FRAMES=99"
:: Group offload resident-window size (number of transformer blocks kept on GPU
:: simultaneously). 4 is the upstream default and works for ~24 GB peak at
:: 384x640. Drop to 2 or 1 if VRAM is tight (e.g. 720p+ on 5090) -- each step
:: gets slower due to extra block swapping, but peak resident VRAM drops
:: proportionally. Only matters when --low-vram is on (the default).
set "HELIOS_NUM_BLOCKS_PER_GROUP=4"
:: Variant picks the HF repo and -- importantly -- the guidance_scale, since
:: the step-wise distilled model ignores guidance (bakes it in via DMD) while
:: Mid and Base actually use it. Each variant ~131 GB on disk (separate cache).
:: Default is base (highest-quality non-step-wise model). For fast iteration,
:: pass --variant distilled (~5 min/video vs ~50-100 min for base).
set "HELIOS_VARIANT=base"
:: Random seed by default so each run varies; cmd's %%RANDOM%% returns 0-32767
:: which is plenty for visual variety. Override with --seed N for reproducible
:: runs. The chosen seed is echoed in the startup banner so you can replay it.
set "HELIOS_SEED=!RANDOM!"
:: i2v image-noise sigma controls how much the model deviates from the input
:: image. Lower = stick close to input (less creative drift); higher = more
:: drastic transformation. Helios defaults (0.111 / 0.135) are a moderate
:: range; try 0.05-0.08 for tighter adherence or 0.15-0.25 for stronger
:: motion/transformation. Only applies in --mode i2v.
set "HELIOS_IMAGE_SIGMA_MIN=0.111"
set "HELIOS_IMAGE_SIGMA_MAX=0.135"
:: torch.compile fusion of transformer/text_encoder/vae forward passes. Takes
:: ~5 min to JIT-compile the first step, then subsequent steps run ~10-20%%
:: faster with marginally sharper output. Off by default since it adds startup
:: cost; enable with --enable-compile when iterating with the same setup.
set "HELIOS_ENABLE_COMPILE=0"
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
if /I "%~1"=="--height"        ( set "HELIOS_HEIGHT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--width"         ( set "HELIOS_WIDTH=%~2" & shift & shift & goto parse )
if /I "%~1"=="--num_frames"    ( set "HELIOS_NUM_FRAMES=%~2" & shift & shift & goto parse )
if /I "%~1"=="--num-frames"    ( set "HELIOS_NUM_FRAMES=%~2" & shift & shift & goto parse )
if /I "%~1"=="--num_blocks_per_group" ( set "HELIOS_NUM_BLOCKS_PER_GROUP=%~2" & shift & shift & goto parse )
if /I "%~1"=="--num-blocks-per-group" ( set "HELIOS_NUM_BLOCKS_PER_GROUP=%~2" & shift & shift & goto parse )
if /I "%~1"=="--variant"       ( set "HELIOS_VARIANT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--seed"          ( set "HELIOS_SEED=%~2" & shift & shift & goto parse )
if /I "%~1"=="--image_noise_sigma_min" ( set "HELIOS_IMAGE_SIGMA_MIN=%~2" & shift & shift & goto parse )
if /I "%~1"=="--image-noise-sigma-min" ( set "HELIOS_IMAGE_SIGMA_MIN=%~2" & shift & shift & goto parse )
if /I "%~1"=="--image_noise_sigma_max" ( set "HELIOS_IMAGE_SIGMA_MAX=%~2" & shift & shift & goto parse )
if /I "%~1"=="--image-noise-sigma-max" ( set "HELIOS_IMAGE_SIGMA_MAX=%~2" & shift & shift & goto parse )
if /I "%~1"=="--enable_compile" ( set "HELIOS_ENABLE_COMPILE=1" & shift & goto parse )
if /I "%~1"=="--enable-compile" ( set "HELIOS_ENABLE_COMPILE=1" & shift & goto parse )
if /I "%~1"=="--help"          goto :help
if /I "%~1"=="-h"              goto :help
echo ERROR: unknown arg %~1
exit /b 2
:args_done

REM Helios's official check (pipeline_helios_diffusers.py:314) only enforces
REM `height % 16 == 0 and width % 16 == 0`. But empirically the pipeline has
REM additional internal alignment expectations beyond that -- runs at H=720
REM (45*16) crash at the end-of-loop `torch.cat` in pipeline_helios_diffusers.py:1328,
REM and runs at H=736 (46*16) crash mid-denoise in stage2_sample's
REM convert_flow_pred_to_x0 at scheduling_helios_diffusers.py:846. Both pass /16
REM but fail because some part of the history-window / stage-2 chunk math wants
REM larger power-of-2 alignment. Enforce /64 which covers all observed cases.
REM Known-safe heights/widths: 384, 448, 512, 576, 640, 704, 768, 832, 896.
REM Saves ~45 minutes of wasted denoising on bad values.
set /a "_H_MOD = !HELIOS_HEIGHT! %% 64"
set /a "_W_MOD = !HELIOS_WIDTH! %% 64"
if not "!_H_MOD!"=="0" (
    set /a "_H_LO = !HELIOS_HEIGHT! - !_H_MOD!"
    set /a "_H_HI = !_H_LO! + 64"
    echo ERROR: --height !HELIOS_HEIGHT! must be divisible by 64. Try !_H_LO! or !_H_HI!.
    exit /b 2
)
if not "!_W_MOD!"=="0" (
    set /a "_W_LO = !HELIOS_WIDTH! - !_W_MOD!"
    set /a "_W_HI = !_W_LO! + 64"
    echo ERROR: --width !HELIOS_WIDTH! must be divisible by 64. Try !_W_LO! or !_W_HI!.
    exit /b 2
)

REM Helios processes video in 33-frame chunks (latent chunk size 9 * VAE temporal
REM scale 4 + 1 = 33; documented in docs/README.md's "Example frame counts"
REM table where all "Adjusted Frames" values are 33*N). Passing non-33*N values
REM lets the pipeline auto-round but produces surprising output lengths; fail
REM fast and hint at neighboring valid counts so the user picks deliberately.
set /a "_F_MOD = !HELIOS_NUM_FRAMES! %% 33"
if not "!_F_MOD!"=="0" (
    set /a "_F_LO = !HELIOS_NUM_FRAMES! - !_F_MOD!"
    set /a "_F_HI = !_F_LO! + 33"
    echo ERROR: --num_frames !HELIOS_NUM_FRAMES! must be a multiple of 33 [Helios chunk size].
    echo Try !_F_LO! or !_F_HI!.
    echo Common values: 33, 66, 99 [default ~4s], 132 [~5.5s], 264 [~11s], 726 [~30s].
    exit /b 2
)

:: Variant -> HF repo + guidance_scale + stage2-flag mapping.
::   distilled (default): ~3 effective denoise steps, no guidance, pyramid stage2 ON, ~5 min/video
::   mid:                ~30 steps, guidance=5.0, single-stage, ~50-100 min/video, ~20% quality gain
::   base:               ~30 steps, guidance=5.0, single-stage, ~50-100 min/video, ~50-70% quality gain
:: Each variant lives in a separate HF repo (~131 GB after filtering transformer_init).
:: Step-wise distilled ignores --guidance_scale; Mid/Base actually use it.
:: Stage2 mapping: Distilled's scheduler config has stages>1 (pyramid distillation)
:: and REQUIRES --is_enable_stage2 to avoid KeyError: None at scheduling_helios_diffusers.py:227.
:: Mid/Base have stages=1 (full single-stage denoising) and CRASH WITH --is_enable_stage2
:: at pipeline_helios_diffusers.py:705 (KeyError: 1 when accessing ori_start_sigmas[1]
:: because the scheduler only populated key 0). Override either default via HELIOS_NO_STAGE2.
set "VARIANT=!HELIOS_VARIANT!"
if /I "!VARIANT!"=="distilled" (
    set "HF_REPO=BestWishYsh/Helios-Distilled"
    set "HELIOS_GUIDANCE_SCALE=1.0"
    set "_VARIANT_STAGE2_DEFAULT=1"
) else if /I "!VARIANT!"=="mid" (
    set "HF_REPO=BestWishYsh/Helios-Mid"
    set "HELIOS_GUIDANCE_SCALE=5.0"
    set "_VARIANT_STAGE2_DEFAULT=0"
) else if /I "!VARIANT!"=="base" (
    set "HF_REPO=BestWishYsh/Helios-Base"
    set "HELIOS_GUIDANCE_SCALE=5.0"
    set "_VARIANT_STAGE2_DEFAULT=0"
) else (
    echo ERROR: --variant must be distilled^|mid^|base ^(got !VARIANT!^)
    exit /b 2
)

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
echo   variant  : !VARIANT!  ^(repo: !HF_REPO!, guidance_scale: !HELIOS_GUIDANCE_SCALE!^)
echo   seed     : !HELIOS_SEED!  ^(override with --seed N to reproduce^)
if "!_VARIANT_STAGE2_DEFAULT!"=="1" (
    echo   stage2   : ON  ^(--is_enable_stage2; required for distilled pyramid^)
) else (
    echo   stage2   : OFF ^(single-stage; correct for Mid/Base^)
)
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

:: --- CUDA / GPU sanity probe -------------------------------------------------
:: torch+cu128 wheels ship their own CUDA runtime DLLs so the toolkit doesn't
:: need to be on PATH at inference time, but a functioning NVIDIA driver is
:: required. Probe via nvidia-smi up front so OOMs / driver errors surface
:: here with the actual GPU state instead of inside a 40-line python trace.
echo --- GPU / CUDA probe ---
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo WARN: nvidia-smi not on PATH -- CUDA driver may be missing; inference will fail.
) else (
    REM driver_version isn't accepted as a --query-gpu field on every driver
    REM build (some report it only via the top-level header); query it as a
    REM separate gpu-summary line instead.
    for /f "delims=" %%G in ('nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2^>nul') do echo   GPU: %%G
    for /f "delims=" %%D in ('nvidia-smi --query --display=COMPUTE 2^>nul ^| findstr /C:"Driver Version"') do echo   %%D
)
set "_TK_CU128=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if exist "!_TK_CU128!" (
    echo   CUDA 12.8 toolkit: found at !_TK_CU128!
) else (
    echo   CUDA 12.8 toolkit: not installed ^(optional; torch+cu128 wheels include the runtime^)
)
echo   CUDA_PATH    : %CUDA_PATH%
echo   CUDA_HOME    : %CUDA_HOME%
REM Warn if CUDA_PATH points at a non-12.8 toolkit -- torch+cu128 wheels will
REM still work via their bundled runtime, but custom CUDA-extension builds (e.g.
REM sageattention's Triton JIT, lingbot's gsplat) link against whatever CUDA_PATH
REM resolves to, and a mismatch produces obscure "no kernel image" runtime errors.
if defined CUDA_PATH (
    echo %CUDA_PATH% | findstr /C:"v12.8" >nul
    if errorlevel 1 (
        echo   WARN: CUDA_PATH does not point at v12.8 -- custom-built kernels may mismatch torch's bundled runtime
    )
)
where nvcc >nul 2>nul
if errorlevel 1 (
    echo   nvcc         : not on PATH ^(OK for inference; needed only for CUDA-extension builds^)
) else (
    for /f "tokens=*" %%V in ('nvcc --version 2^>nul ^| findstr /C:"release"') do echo   nvcc         : %%V
)
echo.

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
:: Variant-specific default is set above (_VARIANT_STAGE2_DEFAULT): 1 for Distilled
:: (needs --is_enable_stage2), 0 for Mid/Base (crash if --is_enable_stage2 passed).
set "STAGE2_ARG="
if "!_VARIANT_STAGE2_DEFAULT!"=="1" set "STAGE2_ARG=--is_enable_stage2"
if defined HELIOS_NO_STAGE2 if not "%HELIOS_NO_STAGE2%"=="0" set "STAGE2_ARG="
if defined HELIOS_FORCE_STAGE2 if not "%HELIOS_FORCE_STAGE2%"=="0" set "STAGE2_ARG=--is_enable_stage2"
REM Guidance scale comes from the variant mapping above: 1.0 for distilled
REM (which bakes guidance in via DMD and ignores the runtime arg) vs 5.0 for
REM Mid/Base (which actually use it and otherwise produce blurry, low-CFG output).
REM Output folder defaults to %~dp0output_helios but can be overridden via
REM HELIOS_OUTPUT_FOLDER env (used by drive_helios_i2v.bat to redirect single-
REM example runs into a dedicated dir). The output folder is created by Python.
if not defined HELIOS_OUTPUT_FOLDER set "HELIOS_OUTPUT_FOLDER=%~dp0output_helios"

set "COMMON_ARGS=--base_model_path !HF_REPO! --transformer_path !HF_REPO! --weight_dtype bf16 --height !HELIOS_HEIGHT! --width !HELIOS_WIDTH! --num_frames !HELIOS_NUM_FRAMES! --fps 24 --guidance_scale !HELIOS_GUIDANCE_SCALE! --seed !HELIOS_SEED! --output_folder !HELIOS_OUTPUT_FOLDER! !STAGE2_ARG!"

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
    REM HELIOS_IMAGE_PATH overrides the default wave.jpg conditioning image
    REM (used by drive_helios_i2v.bat to feed a per-run extracted first-frame PNG).
    if not defined HELIOS_IMAGE_PATH set "HELIOS_IMAGE_PATH=%~dp0example\wave.jpg"
    set "MODE_ARGS=--sample_type i2v --image_path !HELIOS_IMAGE_PATH! --image_noise_sigma_min !HELIOS_IMAGE_SIGMA_MIN! --image_noise_sigma_max !HELIOS_IMAGE_SIGMA_MAX!"
    set "PROMPT=A towering emerald wave surges forward, its crest curling with raw power. Sunlight glints off the translucent water. Dynamic motion, cinematic shot."
)
if /I "%MODE%"=="v2v" (
    set "MODE_ARGS=--sample_type v2v --video_path %~dp0example\racer\racer.mp4"
    set "PROMPT=A fast-paced RC car race through a suburban street on a sunny day, with the miniature vehicle zipping past houses, driveways, and mailboxes, accompanied by the hum of its motor and the cheerful sounds of children playing in the background."
)
if not defined MODE_ARGS (
    echo ERROR: --mode must be t2v^|i2v^|v2v ^(got %MODE%^)
    exit /b 2
)

REM Optional prompt override (used by drive_helios_i2v.bat for single-example
REM runs that need a custom prompt instead of the per-mode default).
if defined HELIOS_PROMPT_OVERRIDE if not "!HELIOS_PROMPT_OVERRIDE!"=="" set "PROMPT=!HELIOS_PROMPT_OVERRIDE!"

set "LOW_VRAM_ARGS="
if "%LOW_VRAM%"=="1" set "LOW_VRAM_ARGS=--enable_low_vram_mode --group_offloading_type leaf_level --num_blocks_per_group !HELIOS_NUM_BLOCKS_PER_GROUP!"

set "COMPILE_ARGS="
if "!HELIOS_ENABLE_COMPILE!"=="1" set "COMPILE_ARGS=--enable_compile"

echo --- running infer_helios.py ---
echo   python   : !VENV_PY!
echo   entry    : !ENTRY!
echo   output   : %~dp0output_helios\
echo.

:: Capture inference start time as ticks (UTC, 100ns units). Computing wall
:: clock from cmd's %TIME% is messy because of midnight rollovers; use
:: PowerShell's DateTime.UtcNow.Ticks for a monotonic high-res integer instead.
for /f %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "_T_START=%%T"

:: Background RAM sampler: polls every 2s for the largest python.exe working
:: set and writes the running peak (bytes + GB) to a temp file. Identified by
:: its command line containing the unique metrics-file path so we can find +
:: kill it cleanly after the python invocation finishes (no PID juggling).
set "_METRICS_FILE=%TEMP%\helios_metrics_%RANDOM%.txt"
echo 0 0 > "%_METRICS_FILE%"
start /b "" powershell -NoProfile -WindowStyle Hidden -Command "$peak=0; while ($true) { try { $m=(Get-Process python -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Maximum).Maximum; if ($m -and $m -gt $peak) { $peak=$m; '{0} {1:N2}' -f $peak, ($peak/1GB) | Out-File -LiteralPath '%_METRICS_FILE%' -Force -Encoding ascii } } catch {}; Start-Sleep -Seconds 2 }"

:: Also snapshot GPU VRAM peak (sample every 2s in parallel)
set "_GPU_FILE=%TEMP%\helios_gpu_peak_%RANDOM%.txt"
echo 0 > "%_GPU_FILE%"
start /b "" powershell -NoProfile -WindowStyle Hidden -Command "$peak=0; while ($true) { try { $m=(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | ForEach-Object { [int]$_.Trim() } | Measure-Object -Maximum).Maximum; if ($m -and $m -gt $peak) { $peak=$m; $peak | Out-File -LiteralPath '%_GPU_FILE%' -Force -Encoding ascii } } catch {}; Start-Sleep -Seconds 2 }"

echo --- inference start: %TIME% ---

:: Python's -X utf8 guarantees UTF-8 stdout/stderr so the bat-level log is decodable.
"!VENV_PY!" -X utf8 "!ENTRY!" %COMMON_ARGS% %MODE_ARGS% --prompt "!PROMPT!" %LOW_VRAM_ARGS% %COMPILE_ARGS%
set "EXIT_CODE=!ERRORLEVEL!"

REM Stop the background samplers (find by metrics-file path in their cmdline)
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'powershell.exe' -and ($_.CommandLine -match 'helios_metrics_' -or $_.CommandLine -match 'helios_gpu_peak_') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

REM Compute elapsed wall clock. 1 tick = 100ns -> divide by 10000000 to get
REM seconds. PowerShell handles the arithmetic + formatting in a single call.
for /f %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "_T_END=%%T"
for /f %%E in ('powershell -NoProfile -Command "$d=([TimeSpan]::FromTicks(!_T_END! - !_T_START!)); '{0:00}:{1:00}:{2:00}.{3:000}' -f $d.Hours,$d.Minutes,$d.Seconds,$d.Milliseconds"') do set "_T_ELAPSED=%%E"
for /f %%S in ('powershell -NoProfile -Command "[Math]::Round(([TimeSpan]::FromTicks(!_T_END! - !_T_START!)).TotalSeconds, 1)"') do set "_T_ELAPSED_SEC=%%S"

REM Read peak metrics from temp files (each file format: "bytes GB" or just MiB)
set "_RAM_PEAK_GB=?"
for /f "tokens=2" %%R in ('type "!_METRICS_FILE!" 2^>nul') do set "_RAM_PEAK_GB=%%R"
set "_GPU_PEAK_MIB=?"
for /f %%G in ('type "!_GPU_FILE!" 2^>nul') do set "_GPU_PEAK_MIB=%%G"
del /q "!_METRICS_FILE!" "!_GPU_FILE!" >nul 2>nul

REM Compute generation speed: real fps (frames produced per wall-clock second)
REM and s/frame. Helios writes one mp4 per run, so videos == 1 and total
REM frames == HELIOS_NUM_FRAMES (already validated /33 above).
for /f %%F in ('powershell -NoProfile -Command "if (!_T_ELAPSED_SEC! -gt 0) { [Math]::Round(!HELIOS_NUM_FRAMES! / !_T_ELAPSED_SEC!, 3) } else { 0 }"') do set "_REAL_FPS=%%F"
for /f %%P in ('powershell -NoProfile -Command "if (!HELIOS_NUM_FRAMES! -gt 0) { [Math]::Round(!_T_ELAPSED_SEC! / !HELIOS_NUM_FRAMES!, 2) } else { 0 }"') do set "_SEC_PER_FRAME=%%P"

echo --- inference end:   %TIME%  (elapsed: !_T_ELAPSED!) ---
echo --- peak python RAM: !_RAM_PEAK_GB! GB ---
echo --- peak GPU VRAM:   !_GPU_PEAK_MIB! MiB ---
echo --- generation speed: !_REAL_FPS! frames/sec wall-clock  ^|  !_SEC_PER_FRAME! sec/frame  ^|  !HELIOS_NUM_FRAMES! frames in !_T_ELAPSED_SEC! sec ---

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
echo   run_helios.bat --variant base           use Helios-Base ^(higher quality, ~131 GB extra download, ~50-100 min/video^)
echo   run_helios.bat --variant mid            use Helios-Mid ^(moderate-quality intermediate^)
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
