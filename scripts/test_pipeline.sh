#!/bin/bash
# Test the powerspike-aware pipeline with a small dataset
# This runs end-to-end testing before the full 2000+ match collection

set -e  # Exit on error

echo "========================================================================"
echo "POWERSPIKE PIPELINE - SMALL TEST RUN"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Collect ~50 jungle matches (10-15 min)"
echo "  2. Collect powerspike data for those matches (5 min)"
echo "  3. Train gank priority model (2 min)"
echo "  4. Process training data (1 min)"
echo "  5. Train BC model (5 min)"
echo "  6. Show accuracy metrics"
echo ""
echo "Total time: ~25-30 minutes"
echo ""
read -p "Press Enter to start test run (or Ctrl+C to cancel)..."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

cd "$(dirname "$0")/.."

echo ""
echo -e "${YELLOW}[1/6] Collecting Test Data (50 matches)${NC}"
echo "========================================================================"
python scripts/collect_test_data.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Data collection failed${NC}"
    exit 1
fi

# Check if data was collected
if [ ! -f "data/processed/test_jungle_data.json" ]; then
    echo -e "${RED}❌ Test jungle data not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Test data collected${NC}"

echo ""
echo -e "${YELLOW}[2/6] Collecting Powerspike Data${NC}"
echo "========================================================================"

# Create a test version of powerspike collection that uses test_jungle_data.json
python -c "
import sys
from pathlib import Path
sys.path.append('scripts')

# Import and modify the powerspike collector
exec(open('scripts/collect_powerspike_data.py').read().replace(
    'challenger_jungle_data.json',
    'test_jungle_data.json'
).replace(
    'powerspike_match_data.json',
    'test_powerspike_data.json'
))
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Powerspike collection failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Powerspike data collected${NC}"

echo ""
echo -e "${YELLOW}[3/6] Training Gank Priority Model${NC}"
echo "========================================================================"

# Train on test data
python -c "
import sys
from pathlib import Path

# Modify the training script to use test data
code = open('scripts/train_gank_priority_model.py').read()
code = code.replace('challenger_jungle_data.json', 'test_jungle_data.json')
code = code.replace('powerspike_match_data.json', 'test_powerspike_data.json')
code = code.replace('gank_priority_rf.pkl', 'test_gank_priority_rf.pkl')

exec(code)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Gank priority model training failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Gank priority model trained${NC}"

echo ""
echo -e "${YELLOW}[4/6] Processing Training Data${NC}"
echo "========================================================================"

# Process test data
python -c "
import sys
from pathlib import Path
sys.path.append('src')

code = open('src/training_data.py').read()
code = code.replace('challenger_jungle_data.json', 'test_jungle_data.json')
code = code.replace('states.npy', 'test_states.npy')
code = code.replace('actions.npy', 'test_actions.npy')
code = code.replace('game_indices.npy', 'test_game_indices.npy')

exec(code)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Data processing failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Training data processed${NC}"

echo ""
echo -e "${YELLOW}[5/6] Training BC Model (10 epochs for quick test)${NC}"
echo "========================================================================"

# Train BC model with fewer epochs for testing
python -c "
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import sys

sys.path.append('src')
from jungle_model import JungleNet
from training_data import ALL_CAMPS

# Load test data
data_dir = Path('data/processed')
states = np.load(data_dir / 'test_states.npy')
actions = np.load(data_dir / 'test_actions.npy')

print(f'Loaded {len(states)} training examples')
print(f'State dimension: {states.shape[1]}')
print(f'Action space: {len(ALL_CAMPS)}')

# Simple train/val split
n_train = int(0.8 * len(states))
indices = np.random.permutation(len(states))

train_states = states[indices[:n_train]]
train_actions = actions[indices[:n_train]]
val_states = states[indices[n_train:]]
val_actions = actions[indices[n_train:]]

print(f'Train: {len(train_states)}, Val: {len(val_states)}')

# Create dataloaders
train_dataset = TensorDataset(
    torch.from_numpy(train_states).float(),
    torch.from_numpy(train_actions).long()
)
val_dataset = TensorDataset(
    torch.from_numpy(val_states).float(),
    torch.from_numpy(val_actions).long()
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = JungleNet(state_dim=states.shape[1], action_dim=len(ALL_CAMPS))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Quick training (10 epochs)
print('\nTraining for 10 epochs...')
best_val_acc = 0.0

for epoch in range(10):
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for states_batch, actions_batch in train_loader:
        states_batch = states_batch.to(device)
        actions_batch = actions_batch.to(device)

        logits = model(states_batch)
        loss = criterion(logits, actions_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        train_correct += (predictions == actions_batch).sum().item()
        train_total += actions_batch.size(0)

    train_acc = train_correct / train_total

    # Validate
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for states_batch, actions_batch in val_loader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            logits = model(states_batch)
            loss = criterion(logits, actions_batch)

            val_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            val_correct += (predictions == actions_batch).sum().item()
            val_total += actions_batch.size(0)

    val_acc = val_correct / val_total

    print(f'Epoch {epoch+1:2d}/10 | Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.3f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_path = Path('models/test_jungle_net.pt')
        model_path.parent.mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, model_path)

print(f'\nBest validation accuracy: {best_val_acc:.3f}')
print(f'Model saved to: models/test_jungle_net.pt')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ BC model training failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ BC model trained${NC}"

echo ""
echo -e "${YELLOW}[6/6] Testing Gank Priority Predictor${NC}"
echo "========================================================================"

python -c "
from pathlib import Path
import sys
sys.path.append('src')

try:
    from gank_priority import GankPriorityPredictor

    model_path = Path('models/test_gank_priority_rf.pkl')
    if model_path.exists():
        predictor = GankPriorityPredictor(model_path)

        print('\nTesting gank priority predictions...\n')

        # Test scenario
        top_p, mid_p, bot_p = predictor.predict_lane_priorities(
            jungler_level=4,
            jungler_gold=1000,
            jungler_spike_score=0.6,
            ally_top_spike=0.3,
            ally_mid_spike=0.6,
            ally_bot_spike=0.8,
            enemy_top_spike=0.7,
            enemy_mid_spike=0.6,
            enemy_bot_spike=0.4,
            game_time_seconds=240
        )

        print('Early game scenario (4 min):')
        print(f'  Ally bot (Draven-like): 0.8 spike vs Enemy bot: 0.4 spike')
        print(f'  Ally top (Kayle-like): 0.3 spike vs Enemy top: 0.7 spike')
        print('')
        print(f'Predicted gank priorities:')
        print(f'  Top: {top_p:.3f} ({top_p*100:.1f}%)')
        print(f'  Mid: {mid_p:.3f} ({mid_p*100:.1f}%)')
        print(f'  Bot: {bot_p:.3f} ({bot_p*100:.1f}%)')
        print(f'\n  → Should prioritize BOT (strong early champion)')

    else:
        print('⚠ Gank priority model not found (this is okay for testing)')
except Exception as e:
    print(f'⚠ Could not test gank priority: {e}')
"

echo ""
echo "========================================================================"
echo -e "${GREEN}✅ TEST PIPELINE COMPLETE!${NC}"
echo "========================================================================"
echo ""
echo "Results saved:"
echo "  • data/processed/test_jungle_data.json"
echo "  • data/processed/test_powerspike_data.json"
echo "  • models/test_gank_priority_rf.pkl"
echo "  • models/test_jungle_net.pt"
echo ""
echo "Next steps:"
echo "  1. Review the accuracy numbers above"
echo "  2. If accuracy looks reasonable (>40% on small test set is OK),"
echo "     run the full collection:"
echo "     ./scripts/run_powerspike_pipeline.sh"
echo ""
