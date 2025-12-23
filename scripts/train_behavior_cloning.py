"""
Train jungle decision model via behavior cloning (supervised learning).

This learns to imitate high-elo jungler decisions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent / "src"))
from jungle_model import JungleNet
from training_data import ALL_CAMPS


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for states, actions in dataloader:
        states = states.to(device)
        actions = actions.to(device)

        # Forward pass
        logits = model(states)
        loss = criterion(logits, actions)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == actions).sum().item()
        total += actions.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, compute_topk=False):
    """Evaluate model with optional top-k accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    correct_top2 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)

            logits = model(states)
            loss = criterion(logits, actions)

            total_loss += loss.item()

            # Top-1 accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == actions).sum().item()

            if compute_topk:
                # Top-2 accuracy
                top2_preds = torch.topk(logits, k=2, dim=1).indices
                correct_top2 += (top2_preds == actions.unsqueeze(1)).any(dim=1).sum().item()

                # Top-3 accuracy
                top3_preds = torch.topk(logits, k=3, dim=1).indices
                correct_top3 += (top3_preds == actions.unsqueeze(1)).any(dim=1).sum().item()

            total += actions.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    if compute_topk:
        top2_acc = correct_top2 / total
        top3_acc = correct_top3 / total
        return avg_loss, accuracy, top2_acc, top3_acc

    return avg_loss, accuracy


def main():
    print("="*70)
    print("JUNGLE BEHAVIOR CLONING TRAINING")
    print("="*70)

    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    states = np.load(data_dir / "states.npy")
    actions = np.load(data_dir / "actions.npy")

    # Load game indices to enable game-based splitting
    game_indices_file = data_dir / "game_indices.npy"
    if game_indices_file.exists():
        game_indices = np.load(game_indices_file)
        print(f"\nDataset:")
        print(f"  Examples: {len(states)}")
        print(f"  State dim: {states.shape[1]}")
        print(f"  Actions: {len(np.unique(actions))}")
        print(f"  Games: {len(np.unique(game_indices))}")

        # Train/val split by GAMES (not individual states)
        unique_games = np.unique(game_indices)
        n_train_games = int(0.8 * len(unique_games))

        # Shuffle games
        shuffled_games = np.random.permutation(unique_games)
        train_games = shuffled_games[:n_train_games]
        val_games = shuffled_games[n_train_games:]

        # Get state indices for each split
        train_mask = np.isin(game_indices, train_games)
        val_mask = np.isin(game_indices, val_games)

        train_states = states[train_mask]
        train_actions = actions[train_mask]
        val_states = states[val_mask]
        val_actions = actions[val_mask]

        print(f"\nSplit (by whole games):")
        print(f"  Train: {len(train_states)} states from {len(train_games)} games")
        print(f"  Val: {len(val_states)} states from {len(val_games)} games")
    else:
        # Fallback to old random split if game_indices.npy doesn't exist
        print(f"\nWARNING: game_indices.npy not found, using random split")
        print(f"  This will leak game data between train/val!")
        print(f"\nDataset:")
        print(f"  Examples: {len(states)}")
        print(f"  State dim: {states.shape[1]}")
        print(f"  Actions: {len(np.unique(actions))}")

        n_train = int(0.8 * len(states))
        indices = np.random.permutation(len(states))

        train_states = states[indices[:n_train]]
        train_actions = actions[indices[:n_train]]
        val_states = states[indices[n_train:]]
        val_actions = actions[indices[n_train:]]

        print(f"\nSplit:")
        print(f"  Train: {len(train_states)}")
        print(f"  Val: {len(val_states)}")

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
    print(f"\nDevice: {device}")

    model = JungleNet(state_dim=states.shape[1], action_dim=len(ALL_CAMPS))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")

    n_epochs = 50
    best_val_acc = 0.0

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = Path(__file__).parent.parent / "models" / "jungle_net_best.pt"
            model_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_path)

    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {model_path}")

    # Final evaluation with top-k accuracy
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")

    # Load best model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate with top-k metrics
    val_loss, val_acc, top2_acc, top3_acc = evaluate(
        model, val_loader, criterion, device, compute_topk=True
    )

    print(f"\nFinal Accuracy Metrics:")
    print(f"  Top-1: {val_acc*100:.1f}%")
    print(f"  Top-2: {top2_acc*100:.1f}%")
    print(f"  Top-3: {top3_acc*100:.1f}%")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
