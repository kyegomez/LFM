import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Optional, Tuple

class AdaptiveLinear(nn.Module):
    """
    Adaptive Linear layer whose weight and bias adapt based on input.
    Supports multiple adaptation methods.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        adapt_dim: int,
        adapt_method: str = "add",  # Adaptation method: 'add', 'multiply', 'gate'
    ):
        super(AdaptiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adapt_method = adapt_method

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        self.adapt = nn.Linear(adapt_dim, out_features * in_features)

        if self.adapt_method == "gate":
            self.gate = nn.Linear(adapt_dim, out_features * in_features)


    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        adapt_weight = self.adapt(adapt_input).view(self.out_features, self.in_features)

        if self.adapt_method == "add":
            weight = self.weight + adapt_weight
        elif self.adapt_method == "multiply":
            weight = self.weight * (adapt_weight + 1)
        elif self.adapt_method == "gate":
            gate = torch.sigmoid(self.gate(adapt_input)).view(self.out_features, self.in_features)
            weight = self.weight * gate + adapt_weight * (1 - gate)
        else:
            raise ValueError(f"Invalid adaptation method: {self.adapt_method}")

        return F.linear(x, weight, self.bias)


class TokenMixing(nn.Module):
    def __init__(self, token_dim: int, adapt_dim: int):
        super(TokenMixing, self).__init__()
        self.token_mixing = AdaptiveLinear(token_dim, token_dim, adapt_dim)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, embed_dim = x.shape
        x = x.view(batch_size * seq_length, embed_dim)
        x_mixed = self.token_mixing(x, adapt_input)
        return x_mixed.view(batch_size, seq_length, embed_dim)


class ChannelMixing(nn.Module):
    def __init__(self, channel_dim: int, adapt_dim: int):
        super(ChannelMixing, self).__init__()
        self.channel_mixing = AdaptiveLinear(channel_dim, channel_dim, adapt_dim)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        return self.channel_mixing(x, adapt_input)


class MixtureOfExperts(nn.Module):
    def __init__(self, expert_dim: int, num_experts: int, adapt_dim: int):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([AdaptiveLinear(expert_dim, expert_dim, adapt_dim) for _ in range(num_experts)])
        self.gating = nn.Linear(adapt_dim, num_experts)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        gate_scores = F.softmax(self.gating(adapt_input), dim=-1)
        output = sum(gate_scores[:, i].unsqueeze(1) * expert(x, adapt_input) for i, expert in enumerate(self.experts))
        return output


class LFModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        token_dim: int,
        channel_dim: int,
        expert_dim: int,
        adapt_dim: int,
        num_experts: int,
        dropout_rate: float = 0.1,
        adapt_method: str = "add",
        activation_function: str = "relu"
    ):
        super(LFModel, self).__init__()

        self.activation_function = activation_function.lower()

        self.input_embedding = nn.Linear(input_dim, token_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.adaptor_norm = nn.LayerNorm(adapt_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.channel_norm = nn.LayerNorm(channel_dim)
        self.pre_moe_linear = nn.Linear(channel_dim, expert_dim)
        self.moe_norm = nn.LayerNorm(expert_dim)
        self.output_layer = nn.Linear(expert_dim, token_dim) # Added output layer


        self.featurizer = AdaptiveLinear(token_dim, adapt_dim, adapt_dim, adapt_method=adapt_method)
        self.token_mixer = TokenMixing(token_dim, adapt_dim)
        self.channel_mixer = ChannelMixing(channel_dim, adapt_dim)
        self.moe = MixtureOfExperts(expert_dim, num_experts, adapt_dim)

        # Initialize AdaptiveLinear layers with specified adaptation method for MoE
        for i in range(len(self.moe.experts)):
            self.moe.experts[i] = AdaptiveLinear(expert_dim, expert_dim, adapt_dim, adapt_method=adapt_method)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding(x)
        x = self.dropout(x)

        adapt_input = self.featurizer(x.mean(dim=1), x.mean(dim=1)) # Pass input to featurizer
        adapt_input = self.adaptor_norm(adapt_input)

        token_mixed = self.token_mixer(x, adapt_input)
        token_mixed = self.token_norm(token_mixed)

        channel_mixed = self.channel_mixer(token_mixed, adapt_input)
        channel_mixed = self.channel_norm(channel_mixed)

        moe_input = self.pre_moe_linear(channel_mixed)
        expert_output = self.moe(moe_input, adapt_input)
        expert_output = self.moe_norm(expert_output)

        if self.activation_function == "relu":
            expert_output = F.relu(expert_output)
        elif self.activation_function == "gelu":
            expert_output = F.gelu(expert_output)
        elif self.activation_function == "swish":
            expert_output = expert_output * torch.sigmoid(expert_output)

        output = self.output_layer(expert_output) # Pass through output layer
        return output
