@echo off
:: Thin wrapper that forwards to run_helios.bat --setup --skip-download.
:: Use this to build the venv + install Helios deps without touching the
:: model download or running inference.
::
:: Forwards all extra args to run_helios.bat (e.g. --low-vram has no effect
:: here, but other flags pass through cleanly).
::
:: Usage:
::   setup.bat                        venv + deps install only (distilled, default)
::   setup.bat --variant base         set up for Helios-Base (higher quality, slower)
::   setup.bat --variant mid          set up for Helios-Mid
::   setup.bat --low-vram             same, plus pass --low-vram through
::                                    (won't change setup itself but lands in
::                                    the env so a follow-up run picks it up)
::
:: Note: the venv install is the same regardless of variant -- variant only
:: changes the HF repo. After `setup.bat --variant base` finishes, do
:: `run_helios.bat --variant base` to pull the ~131 GB Base snapshot.

call "%~dp0run_helios.bat" --setup --skip-download %*
exit /b %ERRORLEVEL%
