# Launcher script for the Streamlit dashboard in PowerShell
$scriptsPath = "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts\streamlit.exe"

if (Test-Path $scriptsPath) {
    & $scriptsPath run app/dashboard.py
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m streamlit run app/dashboard.py
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    py -m streamlit run app/dashboard.py
} else {
    Write-Error "Streamlit executable not found. Please check your Python Scripts path."
}
