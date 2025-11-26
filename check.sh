#!/usr/bin/env bash
# 🧪 QuirkLLM Quality Control Script
# Run this before committing code

set -e  # Exit on error

echo "🔍 QuirkLLM Quality Checks Starting..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Export Poetry path
export PATH="/Users/yamacbezirgan/.local/bin:$PATH"

# Change to project directory
cd "$(dirname "$0")"

echo "📋 Step 1: Code Formatting (Black)"
if poetry run black --check quirkllm/ tests/; then
    echo -e "${GREEN}✅ Formatting: PASS${NC}"
else
    echo -e "${RED}❌ Formatting: FAIL${NC}"
    echo -e "${YELLOW}💡 Run: poetry run black quirkllm/ tests/${NC}"
    exit 1
fi
echo ""

echo "🔎 Step 2: Linting (Ruff)"
if poetry run ruff check quirkllm/ tests/; then
    echo -e "${GREEN}✅ Linting: PASS${NC}"
else
    echo -e "${RED}❌ Linting: FAIL${NC}"
    echo -e "${YELLOW}💡 Run: poetry run ruff check --fix quirkllm/ tests/${NC}"
    exit 1
fi
echo ""

echo "🔬 Step 3: Type Checking (Mypy)"
if poetry run mypy quirkllm/ --ignore-missing-imports; then
    echo -e "${GREEN}✅ Type Checking: PASS${NC}"
else
    echo -e "${RED}❌ Type Checking: FAIL${NC}"
    exit 1
fi
echo ""

echo "🧪 Step 4: Unit Tests (Pytest)"
if poetry run pytest tests/ -v --cov=quirkllm --cov-report=term-missing; then
    echo -e "${GREEN}✅ Tests: PASS${NC}"
else
    echo -e "${RED}❌ Tests: FAIL${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}✨ All Quality Checks Passed! ✨${NC}"
echo ""
echo "📊 Coverage Report: htmlcov/index.html"
