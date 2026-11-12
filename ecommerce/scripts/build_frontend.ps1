Set-Location (Resolve-Path "$PSScriptRoot\..\frontend")
if (-not (Test-Path node_modules)) { npm install }
npm run build
