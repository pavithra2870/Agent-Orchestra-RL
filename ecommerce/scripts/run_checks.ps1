# Everything that must stay green: the e-commerce test suite, the base-project suite, and the Streamlit app.
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = Join-Path $root "venv\Scripts\python.exe"

Set-Location $root
& $py -m pytest ecommerce/tests -q

# the base project is run from inside its own folder
Set-Location (Join-Path $root "base")
& $py -m pytest tests -q
& $py -c "from streamlit.testing.v1 import AppTest; at = AppTest.from_file('app.py', default_timeout=120); at.run(); print('Streamlit app exceptions:', list(at.exception))"
