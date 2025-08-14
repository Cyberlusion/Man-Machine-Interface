# Cyberlusion Master Makefile
# GNU Make 4.0+ required

.PHONY: all clean test verify audit deploy

# Compiler and tool configuration
CC := arm-none-eabi-gcc
CXX := arm-none-eabi-g++
RUST := rustc
COQ := coqc
SPIN := spin
AFL := afl-fuzz

# Security flags
SECURITY_FLAGS := -fstack-protector-strong \
                  -fPIE -pie \
                  -D_FORTIFY_SOURCE=2 \
                  -Wformat -Wformat-security \
                  -fno-strict-overflow \
                  -fno-delete-null-pointer-checks

# Targets
all: bootloader firmware neural_stack security_engine

bootloader:
	@echo "Building secure bootloader..."
	$(MAKE) -C src/layer1_firmware/bootloader

firmware:
	@echo "Building RTOS and firmware..."
	$(MAKE) -C src/layer1_firmware/rtos

neural_stack:
	@echo "Building neural interface stack..."
	$(MAKE) -C src/layer2_neural

security_engine:
	@echo "Building behavioral security engine..."
	$(MAKE) -C src/layer3_security

# Testing targets
test: unit_test integration_test security_test

unit_test:
	@echo "Running unit tests..."
	@pytest tests/unit/

integration_test:
	@echo "Running integration tests..."
	@pytest tests/integration/

security_test:
	@echo "Running security tests..."
	@python3 tools/analysis/security_scan.py

# Formal verification
verify:
	@echo "Running formal verification..."
	$(MAKE) -C verification/formal

# Security audit
audit:
	@echo "Running security audit..."
	@cargo audit
	@safety check
	@bandit -r src/

# Fuzzing
fuzz:
	@echo "Starting fuzzing campaign..."
	$(AFL) -i tests/fuzzing/input -o tests/fuzzing/output ./build/test/fuzz_target

# Documentation
docs:
	@echo "Building documentation..."
	@doxygen Doxyfile
	@mkdocs build

# Deployment
deploy: verify audit
	@echo "Preparing deployment package..."
	@bash tools/deployment/package.sh

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/
	@find . -name "*.o" -delete
	@find . -name "*.pyc" -delete