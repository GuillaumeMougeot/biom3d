@echo off
set ENV_NAME=installer
for /f "delims=" %%i in ('where conda.exe') do (
    set "CONDA_PATH=%%i"
)
set DIR=Biom3d
set ARCH=x86_64
set ARCHIVE_NAME=%DIR%_Windows_%ARCH%
if not [%1]==[] set ARCH=%1

:: Checking if venv exist
call "%CONDA_PATH%" env list | findstr /i "%ENV_NAME%" >nul
if %errorlevel%==0 (
    echo Environment "%ENV_NAME%" already exists.
) else (
    echo Create environment "%ENV_NAME%"...
    call "%CONDA_PATH%" create -y -n %ENV_NAME% python=3.11 tk -y
)

call conda activate %ENV_NAME%
call conda install conda-pack -y
:: Avoid pip/conda conflict
call conda install pip=23.1 -y
:: Omero dependencies
call conda install zeroc-ice=3.6.5 -y
call pip install pillow future portalocker pywin32 requests "urllib3<2"
:: Forced to do --no-deps because it would try to reinstall zeroc-ice by compiling it
call pip install omero-py --no-deps
call pip install ezomero --no-deps
call pip install ../../../
call pip cache purge

:: Pack
if exist %DIR% (
    echo Folder %DIR% already exist, deleting...
    rmdir /s /q %DIR%
)
mkdir %DIR%
call conda pack --format=no-archive -o %DIR%\bin
call conda deactivate
(
	echo @echo off
    echo set FIRST_LAUNCH=1
)>%DIR%\bin\env.bat 
copy Biom3d.bat %DIR%/Biom3d.bat
copy "%~dp0..\auto_update.py" %DIR%\bin\auto_update.py
copy logo.ico %DIR%\Biom3d.ico
:: Doesn't work due to antivirus lock
:: powershell -Command "Compress-Archive -Path '%DIR%' -DestinationPath '%DIR%.zip' -Force"
:: Need 7z
7z a -tzip %ARCHIVE_NAME%.zip %DIR%\