import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import List, Dict
class AdaptiveLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, adapt_dim: int):
        super(AdaptiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.adapt = nn.Linear(adapt_dim, out_features * in_features)
        
    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        adapt_weight = self.adapt(adapt_input).view(self.out_features, self.in_features)
        weight = self.weight + adapt_weight
        return F.linear(x, weight, self.bias)

class TokenMixing(nn.Module):
    def __init__(self, token_dim: int, adapt_dim: int):
        super(TokenMixing, self).__init__()
        self.token_mixing = AdaptiveLinear(token_dim, token_dim, adapt_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        batch_size, seq_length, embed_dim = x.shape
        x = x.view(batch_size * seq_length, embed_dim)
        x_mixed = self.token_mixing(x, adapt_input)
        x_mixed = x_mixed.view(batch_size, seq_length, embed_dim)
        return residual + self.dropout(x_mixed)

class ChannelMixing(nn.Module):
    def __init__(self, channel_dim: int, adapt_dim: int):
        super(ChannelMixing, self).__init__()
        self.channel_mixing = AdaptiveLinear(channel_dim, channel_dim, adapt_dim)
        self.norm = nn.LayerNorm(channel_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_mixed = self.channel_mixing(x, adapt_input)
        return residual + self.dropout(x_mixed)

class KolmogorovArnoldExpert(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super(KolmogorovArnoldExpert, self).__init__()
        self.phi_functions = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(input_dim)])
        self.psi_function = nn.Linear(input_dim * hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi_outputs = [phi(x[:, i].unsqueeze(1)) for i, phi in enumerate(self.phi_functions)]
        concatenated = torch.cat(phi_outputs, dim=1)
        return self.psi_function(concatenated)

class MixtureOfExperts(nn.Module):
    def __init__(self, expert_dim: int, num_experts: int, adapt_dim: int, token_dim: int, channel_dim: int, hidden_dim: int):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([
            AdaptiveLinear(expert_dim, expert_dim, adapt_dim)
            for _ in range(num_experts - 2)
        ])
        self.lf_submodel = LFModel(token_dim, channel_dim, expert_dim, adapt_dim, num_experts, num_layers=2)
        self.ka_expert = KolmogorovArnoldExpert(expert_dim, expert_dim, hidden_dim)
        self.gating = nn.Linear(adapt_dim, num_experts)
        self.max_recursion = 2
        
    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor, recursion_level: int = 0) -> torch.Tensor:
        gate_scores = F.softmax(self.gating(adapt_input), dim=-1)
        
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(gate_scores[:, i].unsqueeze(1) * expert(x, adapt_input))
        
        # LF submodel expert
        lf_output = self.lf_submodel(x)
        expert_outputs.append(gate_scores[:, -2].unsqueeze(1) * lf_output)
        
        # Kolmogorov-Arnold expert
        ka_output = self.ka_expert(x)
        expert_outputs.append(gate_scores[:, -1].unsqueeze(1) * ka_output)
        
        output = sum(expert_outputs)
        
        if recursion_level < self.max_recursion:
            return self.forward(output, adapt_input, recursion_level + 1)
        else:
            return output

class Attention(nn.Module):
    def __init__(self, dim: int):
        super(Attention, self).__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return attn @ v

class LFModel(nn.Module):
    def __init__(
        self,
        token_dim: int,
        channel_dim: int,
        expert_dim: int,
        adapt_dim: int,
        num_experts: int,
        num_layers: int = 3,
        hidden_dim: int = 64
    ):
        super(LFModel, self).__init__()
        self.featurizer = nn.Linear(token_dim, adapt_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'token_mixer': TokenMixing(token_dim, adapt_dim),
                'channel_mixer': ChannelMixing(channel_dim, adapt_dim),
                'moe': MixtureOfExperts(expert_dim, num_experts, adapt_dim, token_dim, channel_dim, hidden_dim),
                'attention': Attention(token_dim)
            }) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(expert_dim, token_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.info("Input shape: {}", x.shape)
        
        adapt_input = self.featurizer(x.mean(dim=1))
        logger.info("Featurization complete. Shape: {}", adapt_input.shape)
        
        for i, layer in enumerate(self.layers):
            x = layer['token_mixer'](x, adapt_input)
            logger.info(f"Layer {i+1} Token mixing complete. Shape: {x.shape}")
            
            x = layer['channel_mixer'](x, adapt_input)
            logger.info(f"Layer {i+1} Channel mixing complete. Shape: {x.shape}")
            
            x = layer['moe'](x, adapt_input)
            logger.info(f"Layer {i+1} Mixture of Experts complete. Shape: {x.shape}")
            
            x = x + layer['attention'](x)
            logger.info(f"Layer {i+1} Attention complete. Shape: {x.shape}")
        
        output = self.output_layer(x)
        logger.info("Output shape: {}", output.shape)
        
        return output

class AdaptiveConfiguration(nn.Module):
    def __init__(self, adapt_dim: int):
        super(AdaptiveConfiguration, self).__init__()
        self.config_net = nn.Sequential(
            nn.Linear(adapt_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 outputs: max_recursion, num_experts, num_layers, hidden_dim
        )
        
    def forward(self, adapt_input: torch.Tensor) -> Dict[str, int]:
        config = self.config_net(adapt_input)
        return {
            'max_recursion': max(1, int(config[0].item())),
            'num_experts': max(2, int(config[1].item())),
            'num_layers': max(1, int(config[2].item())),
            'hidden_dim': max(16, int(config[3].item()))
        }

class AdaptiveLFModel(nn.Module):
    def __init__(
        self,
        token_dim: int,
        channel_dim: int,
        expert_dim: int,
        adapt_dim: int,
        initial_num_experts: int = 8,
        initial_num_layers: int = 3,
        initial_hidden_dim: int = 64
    ):
        super(AdaptiveLFModel, self).__init__()
        self.adaptive_config = AdaptiveConfiguration(adapt_dim)
        self.token_dim = token_dim
        self.channel_dim = channel_dim
        self.expert_dim = expert_dim
        self.adapt_dim = adapt_dim
        self.lf_model = LFModel(
            token_dim, channel_dim, expert_dim, adapt_dim,
            initial_num_experts, initial_num_layers, initial_hidden_dim
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapt_input = self.lf_model.featurizer(x.mean(dim=1))
        config = self.adaptive_config(adapt_input)
        
        # Обновляем конфигурацию модели
        self.lf_model = LFModel(
            self.token_dim, self.channel_dim, self.expert_dim, self.adapt_dim,
            config['num_experts'], config['num_layers'], config['hidden_dim']
        )
        
        for layer in self.lf_model.layers:
            layer['moe'].max_recursion = config['max_recursion']
        
        return self.lf_model(x)

# Пример использования
if __name__ == "__main__":
    model = AdaptiveLFModel(
        token_dim=512,
        channel_dim=512,
        expert_dim=512,
        adapt_dim=256
    )
    
    dummy_input = torch.randn(32, 100, 512)  # [batch_size, sequence_length, embedding_dim]
    output = model(dummy_input)
    print(f"Final output shape: {output.shape}")
