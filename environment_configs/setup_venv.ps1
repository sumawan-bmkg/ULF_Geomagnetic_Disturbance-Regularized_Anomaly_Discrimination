# Setup Virtual Environment for EQ-Precursor-Model-Zoo
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\environment_configs\requirements.txt
Write-Host "Environment setup complete!" -ForegroundColor Green
