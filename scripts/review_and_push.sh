#!/bin/bash
# =====================================================
# Review and Push Script
# Manual gate before pushing versioned model and data
# =====================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "  Model Review & Push"
echo "=========================================="

# ----------------------------------------
# 1. Display Training Results
# ----------------------------------------
echo -e "\n${BLUE}[1/5] Training Metrics${NC}"
echo "----------------------------------------"

if [ -f "outputs/train_metrics.json" ]; then
    echo "Training metrics:"
    python3 -c "
import json
with open('outputs/train_metrics.json') as f:
    m = json.load(f)
    for k, v in m.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')
"
else
    echo -e "${RED}  No training metrics found!${NC}"
fi

# ----------------------------------------
# 2. Display Evaluation Results
# ----------------------------------------
echo -e "\n${BLUE}[2/5] Evaluation Metrics${NC}"
echo "----------------------------------------"

if [ -f "outputs/eval_metrics.json" ]; then
    echo "Validation metrics:"
    python3 -c "
import json
with open('outputs/eval_metrics.json') as f:
    m = json.load(f)
    print(f\"  mAP50:     {m.get('mAP50', 'N/A'):.4f}\" if isinstance(m.get('mAP50'), float) else f\"  mAP50: {m.get('mAP50', 'N/A')}\")
    print(f\"  mAP50-95:  {m.get('mAP50-95', 'N/A'):.4f}\" if isinstance(m.get('mAP50-95'), float) else f\"  mAP50-95: {m.get('mAP50-95', 'N/A')}\")
    print(f\"  Precision: {m.get('precision', 'N/A'):.4f}\" if isinstance(m.get('precision'), float) else f\"  Precision: {m.get('precision', 'N/A')}\")
    print(f\"  Recall:    {m.get('recall', 'N/A'):.4f}\" if isinstance(m.get('recall'), float) else f\"  Recall: {m.get('recall', 'N/A')}\")
"
else
    echo -e "${YELLOW}  No evaluation metrics found${NC}"
fi

# ----------------------------------------
# 3. Display Test Results (if available)
# ----------------------------------------
echo -e "\n${BLUE}[3/5] Test Metrics${NC}"
echo "----------------------------------------"

if [ -f "outputs/test_metrics.json" ]; then
    echo "Test metrics:"
    python3 -c "
import json
with open('outputs/test_metrics.json') as f:
    m = json.load(f)
    print(f\"  Total images: {m.get('total_images', 'N/A')}\")
    print(f\"  Total predictions: {m.get('total_predictions', 'N/A')}\")
    if m.get('has_labels'):
        print(f\"  Precision: {m.get('precision', 'N/A'):.4f}\" if isinstance(m.get('precision'), float) else '')
        print(f\"  Recall:    {m.get('recall', 'N/A'):.4f}\" if isinstance(m.get('recall'), float) else '')
        print(f\"  F1 Score:  {m.get('f1_score', 'N/A'):.4f}\" if isinstance(m.get('f1_score'), float) else '')
    else:
        print('  (No labels available for metrics)')
"
else
    echo -e "${YELLOW}  No test metrics found (test stage may not have run)${NC}"
fi

# ----------------------------------------
# 4. Show Model Info
# ----------------------------------------
echo -e "\n${BLUE}[4/5] Model Info${NC}"
echo "----------------------------------------"

BEST_MODEL="runs/train/exp/weights/best.pt"
if [ -f "$BEST_MODEL" ]; then
    echo "  Best model: $BEST_MODEL"
    echo "  Size: $(du -h $BEST_MODEL | cut -f1)"
    echo "  Modified: $(stat -c %y $BEST_MODEL 2>/dev/null || stat -f %Sm $BEST_MODEL)"
else
    echo -e "${RED}  Best model not found!${NC}"
    exit 1
fi

# ----------------------------------------
# 5. Compare with Previous (if exists)
# ----------------------------------------
echo -e "\n${BLUE}[5/5] Comparison with Previous Version${NC}"
echo "----------------------------------------"

if dvc metrics diff 2>/dev/null | grep -q .; then
    echo "Changes from previous version:"
    dvc metrics diff
else
    echo "  No previous version to compare"
fi

# ----------------------------------------
# User Decision
# ----------------------------------------
echo ""
echo "=========================================="
echo -e "${YELLOW}Review the metrics above.${NC}"
echo ""
echo "Options:"
echo "  ${GREEN}y${NC} - Push model and data to DVC remote + Git"
echo "  ${YELLOW}t${NC} - Tag this version and push"
echo "  ${RED}n${NC} - Cancel (do not push)"
echo ""
read -p "Push this model version? [y/t/n]: " choice

case $choice in
    y|Y)
        echo -e "\n${GREEN}Pushing to DVC remote...${NC}"
        dvc push
        
        echo -e "${GREEN}Committing to Git...${NC}"
        git add dvc.lock outputs/*.json params.yaml 2>/dev/null || true
        git add *.dvc .dvc/config 2>/dev/null || true
        
        # Get commit message
        read -p "Enter commit message (or press Enter for default): " commit_msg
        if [ -z "$commit_msg" ]; then
            commit_msg="Update model - $(date +%Y-%m-%d)"
        fi
        
        git commit -m "$commit_msg"
        
        echo -e "\n${GREEN}✓ Changes committed locally${NC}"
        echo ""
        read -p "Push to GitHub? [y/n]: " push_git
        if [ "$push_git" = "y" ] || [ "$push_git" = "Y" ]; then
            git push
            echo -e "${GREEN}✓ Pushed to GitHub${NC}"
        fi
        ;;
        
    t|T)
        echo -e "\n${GREEN}Tagging and pushing...${NC}"
        
        # Get tag name
        read -p "Enter version tag (e.g., v1.0.0): " tag_name
        read -p "Enter tag message: " tag_msg
        
        dvc push
        
        git add dvc.lock outputs/*.json params.yaml 2>/dev/null || true
        git add *.dvc .dvc/config 2>/dev/null || true
        git commit -m "Release $tag_name: $tag_msg"
        git tag -a "$tag_name" -m "$tag_msg"
        
        echo ""
        read -p "Push to GitHub (with tags)? [y/n]: " push_git
        if [ "$push_git" = "y" ] || [ "$push_git" = "Y" ]; then
            git push
            git push --tags
            echo -e "${GREEN}✓ Pushed to GitHub with tag $tag_name${NC}"
        fi
        ;;
        
    *)
        echo -e "\n${YELLOW}Push cancelled.${NC}"
        echo "Your trained model is still available locally."
        echo "Run this script again when ready to push."
        ;;
esac

echo ""
echo "Done!"
