# Dev server with hot reload on http://localhost:5173 (proxies /api to the backend on port 8000)
Set-Location (Resolve-Path "$PSScriptRoot\..\frontend")
if (-not (Test-Path node_modules)) { npm install }
npm run dev
