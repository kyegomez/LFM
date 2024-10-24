
# Liquid Foundation Models [LFMs]

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

**Welcome to the open-source implementation of Liquid Foundation Models (LFMs)** — pioneering the frontier of real-time learning in AI. LFMs are designed to adapt and learn on-the-fly, continuously evolving their knowledge and capabilities as they interact with new data. This real-time learning approach allows LFMs to stay current and relevant in rapidly changing environments, making them ideal for applications that require up-to-the-minute intelligence and adaptability. Whether processing streaming text, analyzing live audio, interpreting real-time video feeds, or responding to dynamic image inputs, LFMs excel at absorbing and applying new information instantaneously. [Discover more about the model from the original article](https://www.liquid.ai/liquid-foundation-models)

## Installation
```bash
$ pip3 install -U lfm-torch
```

## Usage

```python
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
```


## Liquid Transformer 
A novel neural architecture combining Liquid Neural Networks, Transformer attention mechanisms, and Mixture of Experts (MoE) for enhanced adaptive processing and dynamic state updates. Very experimental and early! We're working on a training script [here](./liquid_transformer_train.py). It still needs an actual tokenizer like llama's tokenizer but it's getting there. If you can help with this then let me know.


### Architecture Overview

```mermaid
flowchart TB
    subgraph "Liquid Transformer"
        Input["Input Sequence"] --> TL["Transformer Layer"]
        
        subgraph "Transformer Layer"
            direction TB
            MHA["Multi-Head Attention"] --> LC["Liquid Cell"]
            LC --> MOE["Mixture of Experts"]
            MOE --> LN["Layer Norm + Residual"]
        end
        
        subgraph "Liquid Cell Details"
            direction LR
            HS["Hidden State"] --> WH["W_h Linear"]
            Input2["Input"] --> WI["W_in Linear"]
            WH --> Add((+))
            WI --> Add
            Add --> Act["Activation"]
            Act --> LN2["LayerNorm"]
            LN2 --> DO["Dropout"]
        end
        
        subgraph "MoE Details"
            direction TB
            Input3["Input"] --> Gate["Gating Network"]
            Input3 --> E1["Expert 1"]
            Input3 --> E2["Expert 2"]
            Input3 --> E3["Expert N"]
            Gate --> Comb["Weighted Combination"]
            E1 --> Comb
            E2 --> Comb
            E3 --> Comb
        end
        
        TL --> Output["Output Sequence"]
    end
```



```python
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
```


# Citations
- All credit for the liquid transformer architecture goes to the original authors: [Google](https://arxiv.org/abs/2402.05385)


# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
