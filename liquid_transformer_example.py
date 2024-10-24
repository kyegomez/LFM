import torch
from loguru import logger

from lfm_torch.liquid_t_moe import LiquidTransformer

# Example usage
if __name__ == "__main__":
    seq_len, batch_size, embed_size = 10, 2, 64
    num_heads, num_experts, expert_size, num_layers = 8, 4, 64, 6

    # Create the model
    model = LiquidTransformer(embed_size, num_heads, num_experts, expert_size, num_layers)

    # Example input tensor
    x = torch.randn(seq_len, batch_size, embed_size)

    # Forward pass
    output = model(x)
    logger.info(f"Model output shape: {output.shape}")
