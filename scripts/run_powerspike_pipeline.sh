#!/bin/bash
# Powerspike-Aware Gank Priority Training Pipeline
# Run this script to execute the full training pipeline end-to-end

set -e  # Exit on error

echo "========================================================================"
echo "POWERSPIKE-AWARE JUNGLE TRAINING PIPELINE"
echo "========================================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: Must run from project root directory${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Data Collection${NC}"
echo "----------------------------------------"
echo "Collecting 2000+ jungle matches from Challenger/Grandmaster leaderboards"
echo "This will take 2-4 hours due to Riot API rate limits..."
echo ""
read -p "Press Enter to start data collection (or Ctrl+C to skip)..."

python scripts/collect_from_leaderboard.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data collection complete!${NC}"
else
    echo -e "${RED}❌ Data collection failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 2: Powerspike Data Collection${NC}"
echo "----------------------------------------"
echo "Extracting champion powerspike snapshots from matches..."
echo ""

python scripts/collect_powerspike_data.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Powerspike data collection complete!${NC}"
else
    echo -e "${RED}❌ Powerspike data collection failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 3: Train Gank Priority Model${NC}"
echo "----------------------------------------"
echo "Training Random Forest to predict lane gank priorities..."
echo ""

python scripts/train_gank_priority_model.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Gank priority model trained!${NC}"
else
    echo -e "${RED}❌ Gank priority model training failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 4: Process Training Data${NC}"
echo "----------------------------------------"
echo "Creating enhanced state vectors with powerspike features..."
echo ""

python src/training_data.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training data processed!${NC}"
else
    echo -e "${RED}❌ Training data processing failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 5: Train Behavior Cloning Model${NC}"
echo "----------------------------------------"
echo "Training BC model with powerspike-aware state representation..."
echo ""

python scripts/train_behavior_cloning.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ BC model trained!${NC}"
else
    echo -e "${RED}❌ BC model training failed${NC}"
    exit 1
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}✅ PIPELINE COMPLETE!${NC}"
echo "========================================================================"
echo ""
echo "Models created:"
echo "  • models/gank_priority_rf.pkl (Random Forest gank predictor)"
echo "  • models/jungle_net_best.pt (BC model with powerspike features)"
echo ""
echo "Next steps:"
echo "  1. Review model performance in training logs"
echo "  2. Test gank priority predictions:"
echo "     python src/gank_priority.py"
echo "  3. Train RL agent with reward shaping:"
echo "     python scripts/train_rl_agent.py"
echo ""
echo "See POWERSPIKE_IMPLEMENTATION_SUMMARY.md for details."
echo "See docs/powerspike_gank_priority_guide.md for usage guide."
echo ""
