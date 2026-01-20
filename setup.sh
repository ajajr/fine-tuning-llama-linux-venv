#!/bin/bash

set -e

echo "Setting up Python 3.12 virtual environment for Ubuntu 24.04..."

# Update package manager
sudo apt-get update

# Install Python 3.12 and required tools (all available in Ubuntu 24.04)
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev git curl

# Create virtual environment without pip (avoid ensurepip issues)
python3.12 -m venv --without-pip --clear venv

# Fix permissions for Windows-mounted directories
# chmod -R u+rwx venv 2>/dev/null || true

# Activate and install pip manually
source venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
pip install --upgrade pip setuptools

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining requirements
pip install -r requirements.txt

echo "Setup complete! Activate with: source venv/bin/activate"

echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo -e "${GREEN}Virtual environment is ready at: ./venv${NC}"
echo -e "${GREEN}To activate the environment, run: ${NC}${BLUE}source venv/bin/activate${NC}"
