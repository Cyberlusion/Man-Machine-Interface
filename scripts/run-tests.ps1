# Run Cyberlusion test suite

Write-Host "Running Cyberlusion Tests..." -ForegroundColor Green

# Python tests
Write-Host "`nRunning Python tests..." -ForegroundColor Yellow
python -m pytest tests/ -v --cov=src --cov-report=html

# Rust tests
Write-Host "`nRunning Rust tests..." -ForegroundColor Yellow
cargo test --all

# Security scan
Write-Host "`nRunning security scan..." -ForegroundColor Yellow
bandit -r src/ -f json -o security_report.json

Write-Host "`nAll tests completed!" -ForegroundColor Green