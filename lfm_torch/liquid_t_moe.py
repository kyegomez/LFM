import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger

# from zeta import MixtureOfExperts, FeedForward

# Logging Configuration
logger.add(
    "liquid_transformer.log",
    format="{time} {level} {message}",
    level="DEBUG",
)


class LiquidCell(nn.Module):
    """
    Liquid Neural Network Cell with enhanced production-readiness.

    This liquid cell dynamically updates its hidden state with input features and
    continuously adapts the internal state over time using non-linear updates.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        dropout (float): Dropout rate for regularization.
        layer_norm (bool): Whether to apply layer normalization to stabilize updates.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        layer_norm: bool = True,
    ):
        super(LiquidCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        # Linear layers for input-to-hidden and hidden-to-hidden connections
        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_h = nn.Linear(hidden_size, hidden_size)

        # Optionally add layer normalization
        self.layer_norm = (
            nn.LayerNorm(hidden_size) if layer_norm else None
        )

        # Stable non-linear activation (can switch to ReLU or GELU)
        self.activation = nn.Tanh()

        logger.info(
            f"Initialized LiquidCell with input_size={input_size}, hidden_size={hidden_size}, dropout={dropout}"
        )

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        Forward pass of the LiquidCell.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).
            h (Tensor): Hidden state tensor of shape (batch_size, hidden_size).

        Returns:
            Tensor: Updated hidden state of shape (batch_size, hidden_size).
        """
        logger.debug(
            f"Input shape: {x.shape}, Hidden state shape: {h.shape}"
        )

        # Update hidden state with dynamic input and previous hidden state
        new_h = self.activation(self.w_in(x) + self.w_h(h))

        # Optionally apply layer normalization
        if self.layer_norm:
            new_h = self.layer_norm(new_h)

        # Apply dropout for regularization
        new_h = self.dropout(new_h)

        logger.debug(f"Updated hidden state shape: {new_h.shape}")
        return new_h

    def initialize_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> Tensor:
        """
        Initialize the hidden state dynamically for the given batch size and device.

        Args:
            batch_size (int): The batch size for which the hidden state is initialized.
            device (torch.device): The device (CPU or GPU) where the hidden state should reside.

        Returns:
            Tensor: Initialized hidden state of shape (batch_size, hidden_size).
        """
        hidden_state = torch.zeros(batch_size, self.hidden_size).to(
            device
        )
        logger.info(
            f"Initialized hidden state of shape {hidden_state.shape} on {device}"
        )
        return hidden_state


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) Layer

    Args:
        num_experts (int): Number of experts.
        expert_size (int): Size of each expert layer.
        output_size (int): Output size for gating network.
    """

    def __init__(
        self, num_experts: int, expert_size: int, output_size: int
    ):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList(
            [
                nn.Linear(expert_size, output_size)
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Linear(expert_size, num_experts)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MoE layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            Tensor: Output of the mixture of experts.
        """
        gate_outputs = F.softmax(self.gate(x), dim=1)
        logger.debug(f"Gate outputs: {gate_outputs}")

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )
        logger.debug(f"Expert outputs: {expert_outputs}")

        output = torch.einsum(
            "be,bec->bc", gate_outputs, expert_outputs
        )
        return output


class TransformerLayerWithLiquid(nn.Module):
    """
    A single transformer block integrated with a Liquid Neural Network Cell and Mixture of Experts.

    Args:
        embed_size (int): Size of embedding.
        num_heads (int): Number of attention heads.
        num_experts (int): Number of experts in the MoE layer.
        expert_size (int): Size of each expert layer.
    """

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        num_experts: int,
        expert_size: int,
    ):
        super(TransformerLayerWithLiquid, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        self.liquid_cell = LiquidCell(embed_size, embed_size)
        self.moe = MixtureOfExperts(
            num_experts, embed_size, embed_size
        )
        # self.moe = MixtureOfExperts(
        #     dim = embed_size,
        #     num_experts=num_experts,
        # )
        self.layernorm = nn.LayerNorm(embed_size)

    def forward(self, x: Tensor, hidden_state: Tensor) -> Tensor:
        """
        Forward pass of the Transformer layer with Liquid Cell and Mixture of Experts.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, embed_size).
            hidden_state (Tensor): Hidden state tensor for the liquid cell (batch_size, embed_size).

        Returns:
            Tensor: Output of the transformer layer.
        """
        logger.debug(
            f"Input shape to TransformerLayerWithLiquid: {x.shape}"
        )

        # Self-attention
        attention_output, _ = self.attention(x, x, x)
        logger.debug(
            f"Attention output shape: {attention_output.shape}"
        )

        # Liquid Neural Network Cell
        hidden_state = self.liquid_cell(
            attention_output.mean(dim=0), hidden_state
        )
        logger.debug(
            f"Updated hidden state from LiquidCell: {hidden_state.shape}"
        )

        # Mixture of Experts
        moe_output = self.moe(hidden_state)
        logger.debug(f"MoE output shape: {moe_output.shape}")

        # Layer Norm and Residual Connection
        output = self.layernorm(
            attention_output + moe_output.unsqueeze(0)
        )
        return output


class LiquidTransformer(nn.Module):
    """
    Transformer with multiple layers of liquid neural network cells and mixture of experts.

    Args:
        embed_size (int): Size of embedding.
        num_heads (int): Number of attention heads.
        num_experts (int): Number of experts in each MoE layer.
        expert_size (int): Size of each expert.
        num_layers (int): Number of transformer layers.
    """

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        num_experts: int,
        expert_size: int,
        num_layers: int,
    ):
        super(LiquidTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayerWithLiquid(
                    embed_size, num_heads, num_experts, expert_size
                )
                for _ in range(num_layers)
            ]
        )
        self.hidden_state = torch.zeros(1, embed_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Liquid Transformer.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, embed_size).

        Returns:
            Tensor: Output of the transformer network.
        """
        for layer in self.layers:
            x = layer(x, self.hidden_state)
        return x


# # Example usage
# if __name__ == "__main__":
#     seq_len, batch_size, embed_size = 10, 2, 64
#     num_heads, num_experts, expert_size, num_layers = 8, 4, 64, 6

#     # Create the model
#     model = LiquidTransformer(embed_size, num_heads, num_experts, expert_size, num_layers)

#     # Example input tensor
#     x = torch.randn(seq_len, batch_size, embed_size)

#     # Forward pass
#     output = model(x)
#     logger.info(f"Model output shape: {output.shape}")
