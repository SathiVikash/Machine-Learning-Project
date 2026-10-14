@echo off
setlocal
echo ===================================================
echo Launching Urban Heat Risk Administrative Dashboard
echo ===================================================

set SCRIPTS_DIR=%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts

if exist "%SCRIPTS_DIR%\streamlit.exe" (
    "%SCRIPTS_DIR%\streamlit.exe" run app\dashboard.py
    goto end
)

python -m streamlit run app\dashboard.py 2>nul
if %ERRORLEVEL% equ 0 goto end

py -m streamlit run app\dashboard.py 2>nul
if %ERRORLEVEL% equ 0 goto end

echo [ERROR] Could not find streamlit. Ensure Python 3.10 and streamlit are installed.
pause

:end
endlocal
