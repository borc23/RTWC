#!/bin/bash
# =====================================================
# Quick Run Script
# Convenient shortcuts for common operations
# =====================================================

set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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
    echo "  ./scripts/run.sh full                       # Run everything"
    echo "  ./scripts/run.sh train                      # Train only"
    echo "  ./scripts/run.sh version_data 1.3.2         # Version data"
    echo "  ./scripts/run.sh metrics                    # Show metrics"
}

get_data_versions() {
    local dvc_file=${1:-data.dvc}
    git log --oneline --format="%h %s" "$dvc_file" | grep -i "Data version v"
}

checkout_data_version() {
    local commit_hash="$1"
    local dvc_file="$2"

    echo -e "${GREEN}Checking out commit $commit_hash....${NC}"
    git checkout "$commit_hash" -- "$dvc_file"
    dvc checkout "$dvc_file"
    echo -e "${GREEN}Data version $commit_hash restored.${NC}"
}

get_data_version() {
    local dvc_file=${1:-data.dvc}

    if [[ ! -f "$dvc_file" ]]; then
        echo -e "${RED}Error: $dvc_file not found.${NC}"
        return 1
    fi

    local versions=()
    while IFS= read -r line; do
        versions+=("$line")
    done < <(get_data_versions "$dvc_file")

    if [[ ${#versions[@]} -eq 0 ]]; then
        echo -e "${RED}No versions found for $dvc_file${NC}"
        return 1
    fi

    echo -e "${GREEN}Available versions of $dvc_file:${NC}"
    PS3="Select row number: "
    select version in "${versions[@]}"; do
        if [[ -n "$version" ]]; then
            local commit_hash=$(echo "$version" | awk '{print $1}')
            checkout_data_version "$commit_hash" "$dvc_file"
            break
        else
            echo -e "${YELLOW}Invalid selection. Try again.${NC}"
        fi
    done
}

case "$1" in
    full)
        echo -e "${GREEN}Running full pipeline...${NC}"
        dvc repro
        ;;
        
    train)
        echo -e "${GREEN}Running training stage...${NC}"
        if [ -n "$2" ]; then
            dvc repro train --vars wandb_project="$2"
        else
            dvc repro train
        fi
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

    version_data)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Usage: ./scripts/run.sh version_data <version>${NC}"
            echo "Example: ./scripts/run.sh version_data 1.0.3"
            exit 1
        fi

        VERSION="$2"
        echo -e "${GREEN}Setting data version to v${VERSION}...${NC}"

        STASH_NEEDED=false
        if ! git diff --quiet || ! git diff --cached --quiet; then
            echo -e "${YELLOW}Unstaged changes detected. Stashing...${NC}"
            git stash push -u -m "Auto-stash before data version v${VERSION}"
            STASH_NEEDED=true
        fi

        echo -e "${YELLOW}Syncing with remote...${NC}"
        git fetch origin
        if ! git diff --quiet HEAD origin/master; then
            echo -e "${YELLOW}Remote has changes. Pulling...${NC}"
            git pull --rebase origin master || {
                echo -e "${RED}Failed to sync with remote. Please resolve conflicts.${NC}"
                if [ "$STASH_NEEDED" = true ]; then
                    echo -e "${YELLOW}Restoring stashed changes...${NC}"
                    git stash pop
                fi
                exit 1
            }
        fi

        if [ "$STASH_NEEDED" = true ]; then
            echo -e "${YELLOW}Restoring stashed changes...${NC}"
            git stash pop || {
                echo -e "${RED}Warning: Could not restore stashed changes. Check 'git stash list'${NC}"
            }
        fi

        dvc add ./data/

        git add data.dvc .gitignore
        git commit -m "Data version v${VERSION}" || {
            echo -e "${YELLOW}No changes to commit or commit failed${NC}"
        }

        echo -e "${GREEN}Pushing data to DVC remote...${NC}"
        dvc push || {
            echo -e "${RED}DVC push failed!${NC}"
            exit 1
        }

        echo -e "${GREEN}Pushing to Git remote...${NC}"
        git push || {
            echo -e "${RED}Git push failed!${NC}"
            exit 1
        }

        echo -e "${GREEN}Data version v${VERSION} successfully created and pushed!${NC}"
        ;;

    get_data_version)
        get_data_version "${2:-data.dvc}"
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
        echo "  - ./outputs/"
        echo "  - ./runs/"
        echo ""
        read -p "Are you sure? [y/n]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            rm -rf ./outputs/ ./runs/
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
