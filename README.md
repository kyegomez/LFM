<!-- [![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503) -->

# Liquid Foundation Models [LFMs]

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

**Welcome to the open-source implementation of Liquid Foundation Models (LFMs)** — the next generation of multi-modal, adaptive AI systems. LFMs represent a breakthrough in model architecture design, specifically tailored to meet the evolving demands of real-world applications across different modalities such as text, audio, image, and video. [The article provides some implementation details](https://www.liquid.ai/liquid-foundation-models)


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

## Overview

The Liquid Foundation Models are a series of generative AI models designed with an entirely new architecture, optimized for performance, scalability, and efficiency across a wide variety of hardware platforms and modalities. Unlike traditional GPT-based models, LFMs introduce a novel computational paradigm that redefines how we approach token and channel mixing, weight sharing, feature sharing, and adaptive computation, making them suitable for a broader range of applications.

LFMs are built to support multi-modal inputs, which allows them to process text, audio, images, and other data types seamlessly. This flexibility makes them ideal for scenarios where combining various forms of data can yield more powerful insights and predictions.

## Key Features of LFMs

### 1. **Multi-Modal Support**
   LFMs natively support multiple data modalities such as text, audio, images, and video. This is achieved through a robust featurization process that converts raw data into structured feature representations. This architecture allows LFMs to unify different types of data and process them under a single framework, enabling richer and more complex tasks like multi-modal reasoning, understanding, and generation.

### 2. **Token Mixing & Channel Mixing**
   The architecture of LFMs incorporates specialized computational units that perform **token mixing** and **channel mixing**:
   
   - **Token Mixing**: This operation focuses on how the model interacts with and mixes embeddings within a sequence of tokens. By leveraging adaptive linear operations, LFMs dynamically modulate the way tokens interact, leading to more efficient and expressive token representations.
   - **Channel Mixing**: Channel mixing refers to interactions between different layers or channels within the model. By introducing adaptive mechanisms at the channel level, LFMs enhance the ability to model complex relationships across different layers, ensuring efficient information flow throughout the network.

### 3. **Adaptive Linear Operators**
   At the heart of LFMs are **adaptive linear operators**. These operators adjust their computational actions based on the input they receive, allowing the model to be more flexible and contextually aware of the data it processes. This adaptiveness reduces the computational overhead typically seen in traditional linear operations while improving the model's performance on diverse data types.

### 4. **Weight & Feature Sharing**
   LFMs incorporate **weight sharing** across depth groups to improve parameter efficiency and reduce memory consumption without sacrificing performance. In addition, **feature sharing** between different featurizer interconnections allows the model to seamlessly transfer learned features between different modalities, resulting in faster convergence and better generalization.

### 5. **Mixture of Experts (MoE)**
   One of the key innovations in LFMs is the **Mixture of Experts (MoE)** layer. This mechanism dynamically selects a subset of experts (sub-models) to process specific inputs. This results in a more efficient model, as only a fraction of the network is used for any given task, significantly reducing computational requirements while maintaining high performance. LFMs with MoE can scale to larger parameter counts while preserving compute efficiency, making them suitable for both cloud and edge deployments.

### 6. **Hardware Optimization**
   LFMs are built with hardware efficiency in mind. They can be optimized for a range of platforms, including Qualcomm, Apple, Cerebras, and AMD, ensuring that the model delivers high throughput regardless of the deployment environment. This optimization not only boosts inference speeds but also enables LFMs to run on cost-effective hardware, democratizing access to high-performance AI.

### 7. **Scalable Architecture**
   With models ranging from 1 billion to over 40 billion parameters, LFMs offer scalability without compromise. The architecture has been designed to maximize performance at every size, and the combination of adaptive computation and MoE allows LFMs to outperform models that are significantly larger in size, offering both efficiency and precision.

## Benefits of LFMs

### 1. **Multi-Modal Capabilities**
   LFMs’ ability to seamlessly integrate and process multi-modal data makes them a powerful tool for applications requiring cross-modal reasoning, such as video analysis with audio context, or multi-modal document understanding. This allows enterprises to leverage a single model architecture to handle diverse data types, reducing the need for separate models and increasing operational efficiency.

### 2. **Superior Performance at Reduced Costs**
   By incorporating MoE and adaptive linear operations, LFMs are able to maintain performance levels comparable to models much larger than themselves. This reduces the cost of inference and training, making LFMs an ideal choice for organizations looking to scale AI without incurring prohibitive hardware costs.

### 3. **Optimized for Real-World Use**
   LFMs are designed with real-world applications in mind, with targeted optimizations for deployment on edge devices, cloud environments, and specialized hardware like GPUs and AI accelerators. Their hardware adaptability ensures smooth operation across diverse deployment environments, from mobile devices to large-scale data centers.

### 4. **Efficient Training and Inference**
   LFMs' unique architecture reduces the overall number of computations required per task, speeding up both training and inference times. Enterprises can expect faster model iterations, quicker deployment of new features, and a more efficient feedback loop for continuous improvement.

### 5. **Scalability and Flexibility**
   With LFMs, enterprises can scale their AI capabilities flexibly. Whether working with smaller 1B parameter models or larger 40B models, LFMs provide scalability that is driven by the needs of the task rather than hardware constraints. This allows businesses to balance performance with cost-effectiveness depending on their specific application.

### 6. **Enterprise-Ready Design**
   LFMs are architected with a focus on deployment at scale. The model’s ability to optimize for different hardware platforms means enterprises can quickly integrate LFMs into their existing workflows without major infrastructure overhauls. Moreover, the inclusion of the Mixture of Experts layer provides higher throughput and lower latency, crucial for time-sensitive applications.

## Conclusion

Liquid Foundation Models (LFMs) mark the next step in the evolution of generative AI models. With their innovative architecture, LFMs provide unmatched performance across multi-modal tasks while being highly efficient and scalable. Their adaptive computational units, token and channel mixing, and Mixture of Experts (MoE) technology offer a new balance between model size, performance, and hardware compatibility.

LFMs are the foundation for the future of AI systems, designed to be deployed in enterprise environments, across diverse hardware setups, and to handle real-world tasks that span multiple data types.

This open-source implementation provides a glimpse into the powerful potential of LFMs, enabling businesses and researchers to experiment with the most advanced model architectures available today.

For more information on the Liquid Foundation Models and their broader applications, visit [Liquid AI](https://www.liquid.ai/liquid-foundation-models).

---
