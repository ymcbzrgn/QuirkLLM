#!/bin/bash
# QuirkLLM Test Runner
# Çalıştırmak için: chmod +x run_tests.sh && ./run_tests.sh

set -e

echo "=== QuirkLLM Test Suite ==="
echo ""

# Python3 versiyonunu kontrol et
echo "Python version:"
python3 --version
echo ""

# pytest modülünü kontrol et
if ! python3 -c "import pytest" 2>/dev/null; then
    echo "❌ pytest not found!"
    echo "Installing pytest and pytest-cov..."
    python3 -m pip install --user pytest pytest-cov
    echo ""
fi

# Testleri çalıştır
echo "Running tests with coverage..."
python3 -m pytest \
    --cov=quirkllm \
    --cov-report=term-missing \
    --cov-report=html \
    -v \
    2>&1

echo ""
echo "=== Coverage report saved to: htmlcov/index.html ==="
echo "Open it with: open htmlcov/index.html"
