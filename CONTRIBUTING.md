# Contributing to Cyberlusion

## Security First
All contributions must prioritize security and user sovereignty.

## Code Standards
- MISRA C compliance for embedded code
- Formal verification for critical paths
- 100% test coverage for security components
- Signed commits required

## Windows Development Setup

### Required Tools:
1. Visual Studio 2022 with C++ workload
2. Python 3.11+ (via python.org or Microsoft Store)
3. Rust toolchain (rustup.rs)
4. Git for Windows with GPG

### Setting up GPG on Windows:
```powershell
# Install GPG4Win
winget install GnuPG.GPG4Win

# Generate key
gpg --full-generate-key

# Configure Git
git config --global user.signingkey YOUR_KEY_ID
git config --global commit.gpgsign true
```

## Review Process
1. Security review
2. Code review
3. Formal verification check
4. Clinical safety assessment

## Reporting Vulnerabilities
Please report security vulnerabilities to nublexer@hotmail.com
Use our PGP key for sensitive communications.