#!/bin/bash
# =====================================================
# Project Setup Script
# Initializes Git, DVC, and project structure
# =====================================================

set -e

echo "=========================================="
echo "  YOLO Pipeline Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration - MODIFY THESE
DVC_REMOTE_PATH="/mnt/dvc-storage"  # Local path for DVC remote storage
GITHUB_REPO=""                       # Leave empty to skip GitHub setup

# ----------------------------------------
# 1. Check Prerequisites
# ----------------------------------------
echo -e "\n${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}Git is not installed. Installing...${NC}"
    sudo apt update && sudo apt install -y git
fi
echo "  ✓ Git: $(git --version)"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 is not installed!${NC}"
    exit 1
fi
echo "  ✓ Python: $(python3 --version)"

# Check/Install pip packages
echo -e "\n${YELLOW}[2/6] Installing Python dependencies...${NC}"
pip install --quiet dvc ultralytics albumentations sahi wandb pyyaml tqdm pillow

# Verify installations
python3 -c "import dvc; print(f'  ✓ DVC: {dvc.__version__}')"
python3 -c "import ultralytics; print(f'  ✓ Ultralytics: {ultralytics.__version__}')"
python3 -c "import wandb; print(f'  ✓ W&B: {wandb.__version__}')"

# ----------------------------------------
# 2. Initialize Git Repository
# ----------------------------------------
echo -e "\n${YELLOW}[3/6] Initializing Git repository...${NC}"

if [ ! -d ".git" ]; then
    git init
    echo "  ✓ Git repository initialized"
else
    echo "  ✓ Git repository already exists"
fi

# Configure Git (if not already configured)
if [ -z "$(git config user.name)" ]; then
    echo -e "${YELLOW}  Git user.name not set. Please configure:${NC}"
    read -p "  Enter your name: " git_name
    git config user.name "$git_name"
fi

if [ -z "$(git config user.email)" ]; then
    echo -e "${YELLOW}  Git user.email not set. Please configure:${NC}"
    read -p "  Enter your email: " git_email
    git config user.email "$git_email"
fi

# ----------------------------------------
# 3. Initialize DVC
# ----------------------------------------
echo -e "\n${YELLOW}[4/6] Initializing DVC...${NC}"

if [ ! -d ".dvc" ]; then
    dvc init
    echo "  ✓ DVC initialized"
else
    echo "  ✓ DVC already initialized"
fi

# ----------------------------------------
# 4. Configure DVC Remote (Local Storage)
# ----------------------------------------
echo -e "\n${YELLOW}[5/6] Configuring DVC remote storage...${NC}"

# Create remote directory if it doesn't exist
if [ ! -d "$DVC_REMOTE_PATH" ]; then
    echo "  Creating DVC remote directory: $DVC_REMOTE_PATH"
    sudo mkdir -p "$DVC_REMOTE_PATH"
    sudo chown $(whoami):$(whoami) "$DVC_REMOTE_PATH"
fi

# Configure DVC remote
dvc remote add -d local "$DVC_REMOTE_PATH" 2>/dev/null || \
dvc remote modify local url "$DVC_REMOTE_PATH"
echo "  ✓ DVC remote configured: $DVC_REMOTE_PATH"

# ----------------------------------------
# 5. Create .gitignore
# ----------------------------------------
echo -e "\n${YELLOW}[6/6] Creating .gitignore...${NC}"

cat > .gitignore << 'EOF'
# DVC
/data
/models/*.pt
!/models/.gitkeep

# Training outputs
/runs
/outputs

# Python
__pycache__/
*.py[cod]
*$py.class
.Python
*.so
.eggs/
*.egg-info/
*.egg

# Virtual environment
venv/
env/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Weights & Biases
wandb/

# Logs
*.log
EOF

echo "  ✓ .gitignore created"

# ----------------------------------------
# 6. Track data with DVC
# ----------------------------------------
echo -e "\n${GREEN}=========================================="
echo "  Setup Complete!"
echo "==========================================${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Copy your data to the 'data/' directory"
echo "   Expected structure:"
echo "     data/"
echo "       train/images/"
echo "       train/labels/"
echo "       val/images/"
echo "       val/labels/"
echo "       test/images/  (optional)"
echo "       test/labels/  (optional)"
echo ""
echo "2. Track your data with DVC:"
echo "   ${GREEN}dvc add data${NC}"
echo ""
echo "3. Login to Weights & Biases:"
echo "   ${GREEN}wandb login${NC}"
echo ""
echo "4. Run the pipeline:"
echo "   ${GREEN}dvc repro${NC}"
echo ""
echo "5. After training, review results and push if satisfied:"
echo "   ${GREEN}./scripts/review_and_push.sh${NC}"
echo ""

# Initial commit
if [ -z "$(git log --oneline 2>/dev/null | head -1)" ]; then
    echo -e "${YELLOW}Creating initial commit...${NC}"
    git add .
    git add .dvc .dvcignore 2>/dev/null || true
    git commit -m "Initial project setup with DVC pipeline"
    echo "  ✓ Initial commit created"
fi

echo -e "\n${GREEN}Setup complete!${NC}"
