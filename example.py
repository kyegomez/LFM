import torch
from lfm_torch.model import LFModel
from loguru import logger

# Instantiate and test the model
if __name__ == "__main__":
    batch_size, seq_length, embedding_dim = 32, 128, 512
    token_dim, channel_dim, expert_dim, adapt_dim, num_experts = (
        embedding_dim,
        embedding_dim,
        embedding_dim,
        128,
        4,
    )
    model = LFModel(
        token_dim, channel_dim, expert_dim, adapt_dim, num_experts
    )

    input_tensor = torch.randn(
        batch_size, seq_length, embedding_dim
    )  # 3D text tensor
    output = model(input_tensor)
    logger.info("Model forward pass complete.")
