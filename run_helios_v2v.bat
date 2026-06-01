@echo off
:: Single-inference v2v wrapper with speed + RAM instrumentation.
::
:: Delegates to run_helios.bat with --mode v2v forced. Uses the racer.mp4
:: source video + RC-car prompt configured in run_helios.bat. After inference
:: the bat prints:
::
::     --- inference end:   HH:MM:SS.ss  (elapsed: HH:MM:SS.mmm) ---
::     --- peak python RAM: X.XX GB ---
::     --- peak GPU VRAM:   X MiB ---
::     --- generation speed: X.XXX frames/sec wall-clock  |  X.XX sec/frame  |  N frames in T sec ---
::
:: All other args forwarded:
::   run_helios_v2v.bat                            defaults (Base, 384x640, 99 frames, racer.mp4)
::   run_helios_v2v.bat --height 768 --width 1280  higher resolution
::   run_helios_v2v.bat --variant distilled        switch to fast distilled
::   run_helios_v2v.bat --seed 7                   reproducible seed
::   run_helios_v2v.bat --num_frames 132           longer clip
::   run_helios_v2v.bat --enable_compile           torch.compile fusion

call "%~dp0run_helios.bat" --mode v2v %*
exit /b %ERRORLEVEL%
