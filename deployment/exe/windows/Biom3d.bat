@echo off
REM Quit at first error
setlocal enabledelayedexpansion

REM Charging env from env.bat
call  bin\env.bat

REM Check if it is first launch
if "%FIRST_LAUNCH%"=="1" (
    echo First launch detected, initializng the virtual environment
    call "%~dp0bin\Scripts\conda-unpack.exe"
    echo Virtual environment initialized
    echo Checking for installed version of CUDA and installing appropriate PyTorch
    REM Avoid dependency conflict, to remove when not needed anymore
    "%~dp0bin\python.exe" "%~dp0bin\auto_update.py"
    echo Done
    (
	echo @echo off
        echo set FIRST_LAUNCH=0
    )>%~dp0bin\env.bat
    
)
echo Starting Biom3d...
REM Launch Biom3d GUI
"%~dp0bin\python.exe" -m biom3d.gui
