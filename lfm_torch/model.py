import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Optional, Tuple
from torch.nn.functional as F

class AdaptiveLinear(nn.Module):
    """
    Adaptive Linear layer whose weight and bias adapt based on input.
    """

    def __init__(
        self, in_features: int, out_features: int, adapt_dim: int
    ):
        super(AdaptiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.randn(out_features))

        # Linear transformation for adapting the weight based on input
        self.adapt = nn.Linear(adapt_dim, out_features * in_features)

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        adapt_weight = self.adapt(adapt_input).view(
            self.out_features, self.in_features
        )
        weight = self.weight + adapt_weight
        return F.linear(x, weight, self.bias)


class TokenMixing(nn.Module):
    """
    Token mixing layer that performs token-wise interactions using adaptive linear layers.
    Operates across the sequence dimension (sequence_length).
    """

    def __init__(self, token_dim: int, adapt_dim: int):
        super(TokenMixing, self).__init__()
        self.token_mixing = AdaptiveLinear(
            token_dim, token_dim, adapt_dim
        )

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        # x: [batch_size, sequence_length, embedding_dim]
        batch_size, seq_length, embed_dim = x.shape
        x = x.view(
            batch_size * seq_length, embed_dim
        )  # Flatten sequence for linear transformation
        x_mixed = self.token_mixing(x, adapt_input)
        return x_mixed.view(batch_size, seq_length, embed_dim)


class ChannelMixing(nn.Module):
    """
    Channel mixing layer that performs cross-channel interactions using adaptive linear layers.
    Operates across the embedding dimension (embedding_dim).
    """

    def __init__(self, channel_dim: int, adapt_dim: int):
        super(ChannelMixing, self).__init__()
        self.channel_mixing = AdaptiveLinear(
            channel_dim, channel_dim, adapt_dim
        )

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        # x: [batch_size, sequence_length, embedding_dim]
        return self.channel_mixing(x, adapt_input)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) module that dynamically selects experts based on input.
    Operates after channel and token mixing.
    """

    def __init__(
        self, expert_dim: int, num_experts: int, adapt_dim: int
    ):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList(
            [
                AdaptiveLinear(expert_dim, expert_dim, adapt_dim)
                for _ in range(num_experts)
            ]
        )
        self.gating = nn.Linear(adapt_dim, num_experts)

    def forward(
        self, x: torch.Tensor, adapt_input: torch.Tensor
    ) -> torch.Tensor:
        gate_scores = F.softmax(self.gating(adapt_input), dim=-1)
        output = sum(
            gate_scores[:, i].unsqueeze(1) * expert(x, adapt_input)
            for i, expert in enumerate(self.experts)
        )
        return output


class LFModel(nn.Module):
    """
    Custom LF Model architecture combining token mixing, channel mixing, and MoE.
    Accepts 3D input tensor: [batch_size, sequence_length, embedding_dim].
    """

    def __init__(
        self,
        token_dim: int,
        channel_dim: int,
        expert_dim: int,
        adapt_dim: int,
        num_experts: int,
    ):
        super(LFModel, self).__init__()
        self.featurizer = nn.Linear(token_dim, adapt_dim)
        self.token_mixer = TokenMixing(token_dim, adapt_dim)
        self.channel_mixer = ChannelMixing(channel_dim, adapt_dim)
        self.moe = MixtureOfExperts(
            expert_dim, num_experts, adapt_dim
        )
        self.output_layer = nn.Linear(expert_dim, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.info("Input shape: {}", x.shape)

        # Featurization stage
        batch_size, seq_length, embed_dim = x.shape
        adapt_input = self.featurizer(
            x.mean(dim=1)
        )  # Aggregate across sequence for adaptation
        logger.info(
            "Featurization complete. Shape: {}", adapt_input.shape
        )

        # Token Mixing
        token_mixed = self.token_mixer(x, adapt_input)
        logger.info(
            "Token mixing complete. Shape: {}", token_mixed.shape
        )

        # Channel Mixing
        channel_mixed = self.channel_mixer(token_mixed, adapt_input)
        logger.info(
            "Channel mixing complete. Shape: {}", channel_mixed.shape
        )

        # Mixture of Experts
        expert_output = self.moe(channel_mixed, adapt_input)
        logger.info(
            "Mixture of Experts complete. Shape: {}",
            expert_output.shape,
        )

        # Final Output
        output = self.output_layer(expert_output)
        logger.info("Output shape: {}", output.shape)
        return output

