# Starts the API (and the built dashboard, if frontend/dist exists) on http://127.0.0.1:8000
# Streamlit keeps its own port (8501) and is not affected.
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
& .\venv\Scripts\python.exe -m ecommerce.backend.main
