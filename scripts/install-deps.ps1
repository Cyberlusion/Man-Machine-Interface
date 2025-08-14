# Install dependencies for Cyberlusion on Windows

Write-Host "Installing Cyberlusion dependencies..." -ForegroundColor Green

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "Please run this script as Administrator" -ForegroundColor Red
    exit 1
}

# Install Chocolatey if not present
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Install tools via Chocolatey
Write-Host "Installing tools via Chocolatey..." -ForegroundColor Yellow
choco install -y git python rust visualstudio2022-workload-nativedesktop cmake ninja

# Install Python packages
Write-Host "Installing Python packages..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install poetry
poetry install

# Install Rust components
Write-Host "Setting up Rust..." -ForegroundColor Yellow
rustup default stable
rustup target add thumbv7em-none-eabihf

Write-Host "Dependencies installed successfully!" -ForegroundColor Green