#!/bin/bash
# =====================================================
# Quick Run Script
# Convenient shortcuts for common operations
# =====================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

show_help() {
    echo "Usage: ./scripts/run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  full        Run full pipeline (prepare → train → evaluate → test)"
    echo "  train       Run only training (skips prepare if data unchanged)"
    echo "  eval        Run only evaluation (requires trained model)"
    echo "  test        Run only test inference"
    echo "  prepare     Run only data preparation"
    echo "  status      Show pipeline status"
    echo "  metrics     Display all metrics"
    echo "  push        Review and push (interactive)"
    echo "  clean       Clean outputs and runs"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run.sh full          # Run everything"
    echo "  ./scripts/run.sh train         # Train only"
    echo "  ./scripts/run.sh metrics       # Show metrics"
}

case "$1" in
    full)
        echo -e "${GREEN}Running full pipeline...${NC}"
        dvc repro
        ;;
        
    train)
        echo -e "${GREEN}Running training stage...${NC}"
        dvc repro train
        ;;
        
    eval|evaluate)
        echo -e "${GREEN}Running evaluation...${NC}"
        dvc repro evaluate
        ;;
        
    test)
        echo -e "${GREEN}Running test inference...${NC}"
        dvc repro test
        ;;
        
    prepare)
        echo -e "${GREEN}Running data preparation...${NC}"
        dvc repro prepare
        ;;
        
    status)
        echo -e "${GREEN}Pipeline Status:${NC}"
        dvc status
        echo ""
        echo -e "${GREEN}DVC Remote Status:${NC}"
        dvc status -r local
        ;;
        
    metrics)
        echo -e "${GREEN}Current Metrics:${NC}"
        echo ""
        dvc metrics show
        ;;
        
    push)
        ./scripts/review_and_push.sh
        ;;
        
    clean)
        echo -e "${YELLOW}This will delete:${NC}"
        echo "  - outputs/"
        echo "  - runs/"
        echo ""
        read -p "Are you sure? [y/n]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            rm -rf outputs/ runs/
            echo -e "${GREEN}Cleaned!${NC}"
        else
            echo "Cancelled."
        fi
        ;;
        
    help|--help|-h|"")
        show_help
        ;;
        
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
