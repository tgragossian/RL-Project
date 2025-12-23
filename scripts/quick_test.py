#!/usr/bin/env python3
"""
Quick test of the powerspike pipeline with minimal data.

This script tests the entire pipeline end-to-end with just 50 matches.
Much faster than the full 2000+ match collection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_data_collection():
    """Step 1: Collect test data (50 matches)."""
    print("="*70)
    print("STEP 1: Collecting 50 test matches from NA")
    print("="*70)

    from collect_test_data import main as collect_test
    collect_test()

    # Verify data was collected
    test_data_path = Path(__file__).parent.parent / "data" / "processed" / "test_jungle_data.json"
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not created at {test_data_path}")

    print("\n✅ Test data collection successful!\n")


def test_model_accuracy():
    """Quick test to see if we can train on the existing data."""
    print("="*70)
    print("QUICK ACCURACY TEST (using existing data if available)")
    print("="*70)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    from jungle_model import JungleNet
    from training_data import ALL_CAMPS

    # Check for existing processed data
    data_dir = Path(__file__).parent.parent / "data" / "processed"

    # Try test data first, fall back to main data
    if (data_dir / "test_states.npy").exists():
        states_file = "test_states.npy"
        actions_file = "test_actions.npy"
        print("Using test dataset")
    elif (data_dir / "states.npy").exists():
        states_file = "states.npy"
        actions_file = "actions.npy"
        print("Using main dataset")
    else:
        print("⚠ No processed data found. Run data collection first:")
        print("  python scripts/collect_test_data.py")
        print("  python src/training_data.py")
        return

    states = np.load(data_dir / states_file)
    actions = np.load(data_dir / actions_file)

    print(f"\nDataset Info:")
    print(f"  Examples: {len(states)}")
    print(f"  State dimension: {states.shape[1]}")
    print(f"  Action space: {len(ALL_CAMPS)} actions")

    if len(states) < 50:
        print(f"⚠ Very small dataset ({len(states)} examples)")
        print("  Results may not be meaningful")
        print("  Collect more data for better accuracy estimates")

    # Train/val split
    n_train = max(int(0.8 * len(states)), 1)
    indices = np.random.permutation(len(states))

    train_states = states[indices[:n_train]]
    train_actions = actions[indices[:n_train]]
    val_states = states[indices[n_train:]]
    val_actions = actions[indices[n_train:]]

    print(f"\nSplit: Train={len(train_states)}, Val={len(val_states)}")

    # Create dataloaders
    batch_size = min(16, len(train_states))
    train_dataset = TensorDataset(
        torch.from_numpy(train_states).float(),
        torch.from_numpy(train_actions).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_states).float(),
        torch.from_numpy(val_actions).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = JungleNet(state_dim=states.shape[1], action_dim=len(ALL_CAMPS))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Quick training (5 epochs for fast test)
    print(f"\n{'='*70}")
    print("Training for 5 epochs (quick test)...")
    print(f"{'='*70}\n")

    for epoch in range(5):
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

        train_acc = train_correct / train_total if train_total > 0 else 0

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

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1}/5 | "
              f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss/len(val_loader) if len(val_loader) > 0 else 0:.4f} Acc: {val_acc:.3f}")

    print(f"\n{'='*70}")
    print("ACCURACY ASSESSMENT")
    print(f"{'='*70}")
    print(f"Final validation accuracy: {val_acc:.1%}")
    print(f"Random baseline: {1/len(ALL_CAMPS):.1%} (1/{len(ALL_CAMPS)})")

    if val_acc > 0.4:
        print("\n✅ GOOD: Accuracy > 40% - Model is learning meaningful patterns")
    elif val_acc > 0.2:
        print("\n⚠ OK: Accuracy 20-40% - Model is learning but needs more data")
    else:
        print("\n⚠ LOW: Accuracy < 20% - Need more/better data")

    print(f"\nFor reference:")
    print(f"  • 40-50%: Good for test dataset")
    print(f"  • 60-70%: Expected with full dataset (2000+ matches)")
    print(f"  • 90%+ top-3: Expected with full dataset")


def main():
    print("\n" + "="*70)
    print("POWERSPIKE PIPELINE - QUICK TEST")
    print("="*70)
    print("\nThis script will:")
    print("  1. Check if data exists")
    print("  2. Run a quick training test to estimate accuracy")
    print("  3. Show you if the pipeline is working")
    print("\n" + "="*70 + "\n")

    try:
        # Just test model accuracy with whatever data we have
        test_model_accuracy()

        print(f"\n{'='*70}")
        print("✅ QUICK TEST COMPLETE")
        print(f"{'='*70}")
        print("\nNext steps:")
        print("  • If accuracy looks reasonable, collect full dataset:")
        print("    python scripts/collect_from_leaderboard.py")
        print("  • Or test with 50 matches first:")
        print("    python scripts/collect_test_data.py")

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
