#!/bin/bash
set -e

echo "=========================================="
echo "LLM Serving Framework Installation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    print_error "Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    exit 1
fi
print_status "Python version: $PYTHON_VERSION"

# Check for CUDA
echo -e "\nChecking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    print_status "CUDA Version: $CUDA_VERSION"
    HAS_CUDA=true
else
    print_warning "CUDA not found - will run in CPU mode (slower)"
    HAS_CUDA=false
fi

# Create virtual environment
echo -e "\nCreating virtual environment..."
python3 -m venv venv
source venv/bin/activate
print_status "Virtual environment created"

# Upgrade pip
echo -e "\nUpgrading pip..."
pip install --upgrade pip setuptools wheel
print_status "Pip upgraded"

# Install PyTorch
echo -e "\nInstalling PyTorch..."
if [ "$HAS_CUDA" = true ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    print_status "PyTorch installed with CUDA support"
else
    pip install torch torchvision torchaudio
    print_status "PyTorch installed (CPU only)"
fi

# Install core dependencies
echo -e "\nInstalling core dependencies..."
pip install -r requirements-core.txt
print_status "Core dependencies installed"

# Try to install vLLM (optional)
if [ "$HAS_CUDA" = true ]; then
    echo -e "\nAttempting to install vLLM..."
    if pip install vllm 2>/dev/null; then
        print_status "vLLM installed successfully"
    else
        print_warning "vLLM installation failed - will use fallback mode"
        print_warning "For vLLM support, manually install: pip install vllm"
    fi
else
    print_warning "Skipping vLLM (requires CUDA)"
fi

# Install development dependencies
echo -e "\nInstalling development dependencies..."
pip install pytest pytest-asyncio black flake8 mypy
print_status "Development dependencies installed"

# Create necessary directories
echo -e "\nCreating project directories..."
mkdir -p logs models config/grafana/dashboards config/grafana/datasources
print_status "Directories created"

# Copy environment template
if [ ! -f .env ]; then
    cp .env.example .env
    print_status "Environment file created (.env)"
    print_warning "Please edit .env with your configuration"
else
    print_warning ".env file already exists"
fi

# Download a test model (optional)
read -p "Download test model (GPT-2, ~500MB)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading GPT-2..."
    python3 << EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Downloading GPT-2 model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
print("Model downloaded successfully!")
EOF
    print_status "Test model downloaded"
fi

# Run tests
echo -e "\n=========================================="
echo "Installation complete!"
echo "=========================================="
print_status "Virtual environment: $(pwd)/venv"
print_status "Activate with: source venv/bin/activate"

echo -e "\nNext steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Edit .env file with your settings"
echo "3. Run tests: pytest tests/"
echo "4. Start server: python -m uvicorn src.api.server:app --reload"
echo "5. Or use Docker: docker-compose up -d"

echo -e "\nTo test installation:"
echo "  python scripts/test_installation.py"
