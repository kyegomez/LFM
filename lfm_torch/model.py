import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from collections import OrderedDict


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
        super().__init__()
        if in_features <= 0 or out_features <= 0 or adapt_dim <= 0:
            raise ValueError("Dimensions must be positive integers.")
        if adapt_method not in ["add", "multiply", "gate"]:
            raise ValueError(f"Invalid adaptation method: {adapt_method}")

        self.in_features = in_features
        self.out_features = out_features
        self.adapt_method = adapt_method

        self.weight = nn.Parameter(torch.empty(out_features, in_features).kaiming_uniform_(a=math.sqrt(5)))  # Kaiming init
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.adapt = nn.Linear(adapt_dim, out_features * in_features)

        if self.adapt_method == "gate":
            self.gate = nn.Linear(adapt_dim, out_features * in_features)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Input tensor has incorrect number of features. Expected {self.in_features}, got {x.shape[-1]}.")
        adapt_weight = self.adapt(adapt_input).view(self.out_features, self.in_features)

        if self.adapt_method == "add":
            weight = self.weight + adapt_weight
        elif self.adapt_method == "multiply":
            weight = self.weight * (adapt_weight + 1)
        elif self.adapt_method == "gate":
            gate = torch.sigmoid(self.gate(adapt_input)).view(self.out_features, self.in_features)
            weight = self.weight * gate + adapt_weight * (1 - gate)

        return F.linear(x, weight, self.bias)


class TokenMixing(nn.Module):
    def __init__(self, token_dim: int, adapt_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.linear1 = nn.Linear(token_dim, token_dim)
        self.linear2 = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.adapt_linear = AdaptiveLinear(token_dim, token_dim, adapt_dim, adapt_method="add")


    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.adapt_linear(x, adapt_input)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ChannelMixing(nn.Module):
    def __init__(self, channel_dim: int, adapt_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channel_dim)
        self.linear1 = nn.Linear(channel_dim, channel_dim)
        self.linear2 = nn.Linear(channel_dim, channel_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.adapt_linear = AdaptiveLinear(channel_dim, channel_dim, adapt_dim, adapt_method="add")

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.adapt_linear(x, adapt_input)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MoE(nn.Module):
    def __init__(self, input_dim: int, expert_dim: int, num_experts: int, dropout_rate: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_weights = F.softmax(self.gate(x), dim=-1)
        expert_outputs = [expert(x) for expert in self.experts]
        output = torch.stack(expert_outputs, dim=-2)
        output = torch.sum(output * gate_weights.unsqueeze(-1), dim=-2)
        output = self.dropout(output)
        return output


class LFModel(nn.Module):
    ACTIVATION_FUNCTIONS = OrderedDict([
        ("relu", F.relu),
        ("gelu", F.gelu),
        ("swish", lambda x: x * torch.sigmoid(x))
    ])

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
        super().__init__()
        if activation_function not in self.ACTIVATION_FUNCTIONS:
            raise ValueError(f"Invalid activation function: {activation_function}")

        self.activation_function = self.ACTIVATION_FUNCTIONS[activation_function]

        self.token_mixing = TokenMixing(token_dim, adapt_dim, dropout_rate)
        self.channel_mixing = ChannelMixing(channel_dim, adapt_dim, dropout_rate)
        self.moe = MoE(input_dim, expert_dim, num_experts, dropout_rate)
        self.output_layer = nn.Linear(expert_dim, input_dim)


    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        x = self.token_mixing(x, adapt_input)
        x = self.channel_mixing(x, adapt_input)
        x = self.moe(x)
        output = self.output_layer(x)
        return output
