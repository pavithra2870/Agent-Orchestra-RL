# PowerShell script to run the Streamlit app
# Usage: .\run_app.ps1

Write-Host " Starting Agent Orchestra RL Frontend..." -ForegroundColor Cyan
Write-Host ""

# Check if streamlit is installed
$streamlitCheck = pip list | Select-String "streamlit"
if (-not $streamlitCheck) {
    Write-Host " Streamlit not found. Installing dependencies..." -ForegroundColor Yellow
    pip install -r ../requirements.txt
    Write-Host ""
}

# Run the app
Write-Host "Launching Streamlit app..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

streamlit run app.py
