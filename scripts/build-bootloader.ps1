# Build secure bootloader for Cyberlusion

Write-Host "Building Secure Bootloader..." -ForegroundColor Green

# Check for ARM toolchain
if (!(Get-Command arm-none-eabi-gcc -ErrorAction SilentlyContinue)) {
    Write-Host "ARM toolchain not found. Please install GNU Arm Embedded Toolchain" -ForegroundColor Red
    Write-Host "Download from: https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm" -ForegroundColor Yellow
    exit 1
}

# Create build directory
if (!(Test-Path "build\bootloader")) {
    New-Item -ItemType Directory -Path "build\bootloader" -Force | Out-Null
}

Set-Location "build\bootloader"

# Run CMake
cmake -G "Ninja" -DCMAKE_TOOLCHAIN_FILE="..\..\cmake\arm-toolchain.cmake" "..\..\src\layer1_firmware\bootloader"

# Build
ninja

Write-Host "Bootloader built successfully!" -ForegroundColor Green
Write-Host "Output: build\bootloader\secure_bootloader.elf" -ForegroundColor Cyan