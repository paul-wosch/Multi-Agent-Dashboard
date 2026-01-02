# PowerShell quick-start (Windows)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$REPO_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $REPO_ROOT

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Error 'python not found; install Python 3.11+ and retry'
  exit 1
}

python -m venv .venv

# Activate venv for this PowerShell session
. .\.venv\Scripts\Activate.ps1

pip install -e .

if (Test-Path .env.template -and -not (Test-Path .env)) {
  Copy-Item .env.template .env
  Write-Host "Copied .env.template → .env (edit .env to add secrets)"
} elseif (-not (Test-Path .env)) {
  'OPENAI_API_KEY=your_key_here' | Out-File -Encoding utf8 .env
  Write-Host "Created .env (edit to add real secrets)"
}

Write-Host "Starting Streamlit app..."
streamlit run src/multi_agent_dashboard/ui/app.py