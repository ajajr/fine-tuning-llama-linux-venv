# LLaMA 3.2 Fine-tuning Project

This project provides a setup for fine-tuning Meta's LLaMA 3.2 1B model using a Python virtual environment on Ubuntu 24.04.

## Prerequisites

- **Ubuntu 24.04 LTS** (or WSL 2 with Ubuntu 24.04)
- Python 3.12 (included with Ubuntu 24.04)
- Git
- curl
- CUDA-capable GPU (optional but recommended for faster training)

## Note for WSL Users

If using WSL 2, it's recommended to work in your WSL home directory rather than `/mnt/c/` (Windows drive) for better file permissions:

```bash
# Copy project to WSL home
cp -r /mnt/f/testing/docker\ test ~
cd ~/docker\ test

# Run setup
./setup.sh
```

## Installation

### Quick Setup

Run the automated setup script to install all dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:
1. Update system packages
2. Install Python 3.12
3. Create a Python virtual environment
4. Install pip and all required packages
5. Install PyTorch with CUDA support

### Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment with Python 3.12
python3.12 -m venv --without-pip venv

# Activate the environment
source venv/bin/activate

# Install pip
curl https://bootstrap.pypa.io/get-pip.py | python

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

# Activate the environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r requirements.txt
```

## Activation

To activate the virtual environment after setup:

```bash
source venv/bin/activate
```

Or use the convenience script:

```bash
chmod +x activate_env.sh
./activate_env.sh
```

## Project Structure

```
.
├── setup.sh                    # Automated setup script
├── activate_env.sh            # Virtual environment activation script
├── requirements.txt           # Python package dependencies
├── finetune_llama.py          # Main fine-tuning script
├── models/                    # Pre-downloaded LLaMA model directory
├── data/                      # Training data directory
├── output/                    # Output directory for trained models
└── README.md                  # This file
```

## Usage

### Running the Fine-tuning Script

With the virtual environment activated:

```bash
python finetune_llama.py
```

### Configuration

Edit `finetune_llama.py` to customize:
- Model parameters
- Training hyperparameters
- Data loading settings
- Output paths

## Dependencies

Key packages installed:
- **torch/torchvision/torchaudio**: Deep learning framework with GPU support
- **transformers**: Hugging Face transformer models library
- **datasets**: Dataset loading and processing
- **peft**: Parameter-Efficient Fine-Tuning methods
- **accelerate**: Distributed training support
- **tokenizers**: Fast tokenization library
- **wandb**: Experiment tracking
- **jupyter/jupyterlab**: Interactive notebooks

See `requirements.txt` for the complete list.

## Performance Notes

- CUDA GPU recommended for faster training
- CPU-only training is supported but significantly slower
- Adjust batch size and gradient accumulation based on available VRAM

## Deactivation

To deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### Virtual Environment Creation Issues
If `python3.12 -m venv` fails:
```bash
sudo apt-get install -y python3.12-venv
```

Then try again:
```bash
rm -rf venv
python3.12 -m venv --without-pip venv
source venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
pip install -r requirements.txt
```

### CUDA Issues
If CUDA support is not detected, verify your GPU drivers and reinstall PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### Permission Denied on Scripts
Make scripts executable:
```bash
chmod +x setup.sh activate_env.sh
```

### Virtual Environment Not Found
Recreate the environment:
```bash.11 -m venv venv
# or if Python 3.11 is not available
python3.10
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Resources

- [LLaMA Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)

## License

Refer to the model's LICENSE file in the models directory for usage terms.
