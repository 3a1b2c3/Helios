@echo off
:: Thin wrapper that forwards to run_helios.bat --setup --skip-download.
:: Use this to build the venv + install Helios deps without touching the
:: model download or running inference.
::
:: Forwards all extra args to run_helios.bat (e.g. --low-vram has no effect
:: here, but other flags pass through cleanly).
::
:: Usage:
::   setup.bat                        venv + deps install only
::   setup.bat --low-vram             same, plus pass --low-vram through
::                                    (won't change setup itself but lands in
::                                    the env so a follow-up run picks it up)

call "%~dp0run_helios.bat" --setup --skip-download %*
exit /b %ERRORLEVEL%
