"""
PyTorch model for jungle decision-making via behavior cloning.

Simple feedforward network that learns to predict which camp to clear next.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JungleNet(nn.Module):
    """
    Neural network for predicting jungle decisions.

    Input: State vector (71 dims)
    Output: Logits over 20 actions (17 camps + 3 ganks)
    """

    def __init__(self, state_dim: int = 71, action_dim: int = 20, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, state):
        """
        Args:
            state: (batch_size, state_dim) tensor

        Returns:
            logits: (batch_size, action_dim) raw scores
        """
        return self.network(state)

    def predict(self, state):
        """
        Predict best action given state.

        Args:
            state: (state_dim,) numpy array or tensor

        Returns:
            action: int (0-16)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(state)
            action = torch.argmax(logits, dim=1)
            return action.item()

    def get_action_probs(self, state):
        """
        Get probability distribution over actions.

        Args:
            state: (state_dim,) numpy array or tensor

        Returns:
            probs: (action_dim,) probability distribution
        """
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(state)
            probs = F.softmax(logits, dim=1)
            return probs.squeeze().numpy()


if __name__ == "__main__":
    # Test model
    print("Testing JungleNet...")

    model = JungleNet()
    print(f"\nModel architecture:")
    print(model)

    # Test forward pass
    batch_size = 4
    state = torch.randn(batch_size, 71)
    logits = model(state)

    print(f"\nInput shape: {state.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output logits: {logits[0][:5]}...")  # First 5 logits

    # Test prediction
    import numpy as np
    test_state = np.random.randn(71).astype(np.float32)
    action = model.predict(test_state)
    probs = model.get_action_probs(test_state)

    print(f"\nPredicted action: {action}")
    print(f"Action probabilities (top 5):")
    top_actions = np.argsort(probs)[-5:][::-1]
    for a in top_actions:
        print(f"  Action {a}: {probs[a]:.3f}")

    print("\n✓ Model test passed!")
