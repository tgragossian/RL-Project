"""
Train behavior cloning model with K-Fold Cross-Validation and architecture search.

This script:
1. Performs 5-fold cross-validation
2. Tests multiple network architectures
3. Finds the best hyperparameters
4. Trains final model on all data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
from pathlib import Path
import sys
import json
from sklearn.model_selection import KFold
from itertools import product

sys.path.append(str(Path(__file__).parent.parent / "src"))
from training_data import ALL_CAMPS


class ConfigurableJungleNet(nn.Module):
    """
    Flexible neural network with configurable architecture.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list, dropout: float = 0.2):
        super().__init__()

        layers = []
        input_dim = state_dim

        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

    def predict(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(state)
            action = torch.argmax(logits, dim=1)
            return action.item()


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for states, actions in dataloader:
        states = states.to(device)
        actions = actions.to(device)

        logits = model(states)
        loss = criterion(logits, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == actions).sum().item()
        total += actions.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)

            logits = model(states)
            loss = criterion(logits, actions)

            total_loss += loss.item()

            # Top-1
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == actions).sum().item()

            # Top-3
            top3_preds = torch.topk(logits, k=3, dim=1).indices
            correct_top3 += (top3_preds == actions.unsqueeze(1)).any(dim=1).sum().item()

            total += actions.size(0)

    acc = correct / total
    top3_acc = correct_top3 / total
    return total_loss / len(dataloader), acc, top3_acc


def cross_validate_config(
    states, actions, config, n_splits=5, n_epochs=30, device='cpu'
):
    """
    Perform K-fold cross-validation for a given architecture config.

    Returns: (mean_val_acc, std_val_acc, mean_top3_acc)
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accs = []
    fold_top3_accs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(states)):
        # Create datasets
        train_dataset = TensorDataset(
            torch.from_numpy(states[train_idx]).float(),
            torch.from_numpy(actions[train_idx]).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(states[val_idx]).float(),
            torch.from_numpy(actions[val_idx]).long()
        )

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Create model
        model = ConfigurableJungleNet(
            state_dim=states.shape[1],
            action_dim=len(ALL_CAMPS),
            hidden_layers=config['hidden_layers'],
            dropout=config['dropout']
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        # Train
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_top3 = evaluate(model, val_loader, criterion, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_top3 = val_top3
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break  # Early stopping

        fold_accs.append(best_val_acc)
        fold_top3_accs.append(best_top3)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    mean_top3 = np.mean(fold_top3_accs)

    return mean_acc, std_acc, mean_top3


def main():
    print("="*70)
    print("ARCHITECTURE SEARCH WITH CROSS-VALIDATION")
    print("="*70)

    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    states = np.load(data_dir / "states.npy")
    actions = np.load(data_dir / "actions.npy")

    print(f"\nDataset:")
    print(f"  Examples: {len(states)}")
    print(f"  State dim: {states.shape[1]}")
    print(f"  Actions: {len(np.unique(actions))}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Define search space
    print(f"\n{'='*70}")
    print("DEFINING SEARCH SPACE")
    print(f"{'='*70}")

    architectures = [
        [128, 128],              # Current (2 layers)
        [256, 128],              # Wider first layer
        [128, 128, 64],          # Current (3 layers)
        [256, 128, 64],          # Wider
        [128, 64, 32],           # Narrower
        [256, 256, 128],         # Much wider
        [512, 256, 128],         # Very wide
        [128, 128, 128, 64],     # Deeper (4 layers)
    ]

    learning_rates = [0.001, 0.0003]
    dropouts = [0.2, 0.3]
    batch_sizes = [16, 32]

    configs = []
    for arch, lr, dropout, batch_size in product(architectures, learning_rates, dropouts, batch_sizes):
        configs.append({
            'hidden_layers': arch,
            'lr': lr,
            'dropout': dropout,
            'batch_size': batch_size,
        })

    print(f"Total configurations to test: {len(configs)}")
    print(f"Using 5-fold cross-validation")
    print(f"This will take ~{len(configs) * 5 * 2} minutes...")

    # Test each configuration
    print(f"\n{'='*70}")
    print("RUNNING CROSS-VALIDATION")
    print(f"{'='*70}\n")

    results = []

    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Testing: {config['hidden_layers']} "
              f"lr={config['lr']:.4f} dropout={config['dropout']} batch={config['batch_size']}")

        mean_acc, std_acc, mean_top3 = cross_validate_config(
            states, actions, config, n_splits=5, n_epochs=30, device=device
        )

        results.append({
            'config': config,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_top3': mean_top3,
        })

        print(f"  → Val Acc: {mean_acc:.4f} ± {std_acc:.4f} | Top-3: {mean_top3:.4f}\n")

    # Find best configuration
    results.sort(key=lambda x: x['mean_acc'], reverse=True)

    print(f"\n{'='*70}")
    print("TOP 5 CONFIGURATIONS")
    print(f"{'='*70}\n")

    for i, result in enumerate(results[:5]):
        config = result['config']
        print(f"{i+1}. Architecture: {config['hidden_layers']}")
        print(f"   LR: {config['lr']}, Dropout: {config['dropout']}, Batch: {config['batch_size']}")
        print(f"   Val Acc: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
        print(f"   Top-3: {result['mean_top3']:.4f}")
        print()

    # Train final model with best config
    best_config = results[0]['config']

    print(f"{'='*70}")
    print("TRAINING FINAL MODEL WITH BEST CONFIG")
    print(f"{'='*70}\n")

    print(f"Best configuration:")
    print(f"  Architecture: {best_config['hidden_layers']}")
    print(f"  Learning rate: {best_config['lr']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  Batch size: {best_config['batch_size']}")

    # Train on full dataset
    full_dataset = TensorDataset(
        torch.from_numpy(states).float(),
        torch.from_numpy(actions).long()
    )

    # 80/20 split for final training
    n_train = int(0.8 * len(states))
    indices = np.random.permutation(len(states))

    train_dataset = Subset(full_dataset, indices[:n_train])
    val_dataset = Subset(full_dataset, indices[n_train:])

    train_loader = DataLoader(train_dataset, batch_size=best_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_config['batch_size'], shuffle=False)

    # Create final model
    final_model = ConfigurableJungleNet(
        state_dim=states.shape[1],
        action_dim=len(ALL_CAMPS),
        hidden_layers=best_config['hidden_layers'],
        dropout=best_config['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=best_config['lr'])

    # Train for 50 epochs
    print(f"\nTraining for 50 epochs...")
    best_val_acc = 0.0
    model_path = Path(__file__).parent.parent / "models" / "jungle_net_best_cv.pt"
    model_path.parent.mkdir(exist_ok=True)

    for epoch in range(50):
        train_loss, train_acc = train_epoch(final_model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_top3 = evaluate(final_model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:3d}/50 | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} Top-3: {val_top3:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': final_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': best_config,
            }, model_path)

    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {model_path}")

    # Save results
    results_file = Path(__file__).parent.parent / "models" / "cv_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': [
                {
                    'config': r['config'],
                    'mean_acc': float(r['mean_acc']),
                    'std_acc': float(r['std_acc']),
                    'mean_top3': float(r['mean_top3']),
                }
                for r in results
            ],
            'best_config': best_config,
            'best_val_acc': float(best_val_acc),
        }, f, indent=2)

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
