import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from loguru import logger
from pydantic import BaseModel

logger.add(
    "model_log.log", format="{time} {level} {message}", level="DEBUG"
)


# Helper class for managing configuration using Pydantic
class ModelConfig(BaseModel):
    input_dim: int
    num_layers: int
    sparsity: float
    cluster_size: int
    hidden_dim: int
    num_clusters: int
    num_classes: int
    memory_size: int


# Sparse Information Extraction Layer
class SparseInformationExtraction(nn.Module):
    """
    This layer selects a sparse subset of tokens based on their importance.
    """

    def __init__(self, input_dim: int, sparsity: float):
        """
        Initializes the sparse selection layer.
        Args:
            input_dim (int): Dimension of input tokens.
            sparsity (float): Fraction of tokens to select (between 0 and 1).
        """
        super(SparseInformationExtraction, self).__init__()
        self.input_dim = input_dim
        self.sparsity = sparsity

    def forward(self, x: Tensor) -> Tensor:
        """
        Select a sparse subset of tokens based on their magnitudes.
        Args:
            x (Tensor): Input token embeddings of shape (batch_size, seq_len, input_dim).
        Returns:
            Tensor: Sparsely selected tokens.
        """
        logger.debug(f"Original input shape: {x.shape}")

        # Compute the L2 norm across the token embeddings
        token_norms = torch.norm(x, p=2, dim=-1)
        logger.debug(f"Token norms shape: {token_norms.shape}")

        # Select top-k tokens based on sparsity value
        k = int(self.sparsity * x.size(1))
        _, topk_indices = torch.topk(token_norms, k, dim=1)

        # Gather the top-k tokens
        sparse_x = torch.gather(
            x,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.input_dim),
        )
        logger.debug(f"Sparse input shape: {sparse_x.shape}")

        return sparse_x


# Hierarchical Clustering Layer
class HierarchicalClustering(nn.Module):
    """
    Hierarchically clusters the input tokens into fewer groups.
    """

    def __init__(self, cluster_size: int):
        """
        Initializes the clustering layer.
        Args:
            cluster_size (int): Number of clusters to group tokens into.
        """
        super(HierarchicalClustering, self).__init__()
        self.cluster_size = cluster_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Cluster tokens hierarchically by reshaping and reducing their dimension.
        Args:
            x (Tensor): Sparse input tokens of shape (batch_size, seq_len, input_dim).
        Returns:
            Tensor: Clustered tokens.
        """
        logger.debug(f"Input before clustering: {x.shape}")
        batch_size, seq_len, input_dim = x.shape
        num_clusters = seq_len // self.cluster_size
        x = x.view(
            batch_size, num_clusters, self.cluster_size * input_dim
        )
        logger.debug(f"Input after clustering: {x.shape}")
        return x


# Dynamic Activation Layer
class DynamicMaskingActivation(nn.Module):
    """
    Activates only a subset of neurons based on dynamic masking.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, mask_fraction: float
    ):
        """
        Initializes the dynamic activation layer.
        Args:
            input_dim (int): Dimension of input layer (matches the output of clustering layer).
            hidden_dim (int): Dimension of hidden layer.
            mask_fraction (float): Fraction of neurons to activate.
        """
        super(DynamicMaskingActivation, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mask_fraction = mask_fraction
        self.fc = nn.Linear(
            input_dim, hidden_dim
        )  # Adjusted to take input_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dynamic masking to the hidden layer.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            Tensor: Masked activation output.
        """
        logger.debug(f"Input before dynamic masking: {x.shape}")
        batch_size, seq_len, input_dim = x.shape

        # Compute the number of neurons to activate
        k = int(self.mask_fraction * self.hidden_dim)

        # Apply linear transformation
        x = self.fc(x)

        # Mask out a random subset of neurons
        mask = torch.zeros_like(x).bernoulli_(self.mask_fraction)
        x = x * mask
        logger.debug(f"Masked output shape: {x.shape}")

        return F.relu(x)


# Sparse Recursion-Based Memory Layer
class SparseMemory(nn.Module):
    """
    Implements a recursive memory mechanism for sequence compression.
    """

    def __init__(self, input_dim: int, memory_size: int):
        """
        Initializes the memory mechanism.
        Args:
            input_dim (int): Dimension of input embeddings.
            memory_size (int): Size of memory (number of stored representations).
        """
        super(SparseMemory, self).__init__()
        self.memory_size = memory_size
        self.fc = nn.Linear(input_dim, memory_size)

    def forward(
        self, x: Tensor, memory: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Update and compress the memory state.
        Args:
            x (Tensor): Current input tensor of shape (batch_size, seq_len, input_dim).
            memory (Tensor): Previous memory state of shape (batch_size, memory_size).
        Returns:
            Tuple[Tensor, Tensor]: Updated input and memory state.
        """
        logger.debug(f"Input before memory update: {x.shape}")

        # Compress sequence length to match memory size (batch_size, memory_size)
        x_compressed = torch.mean(
            x, dim=1
        )  # Compress along the sequence dimension
        logger.debug(f"Compressed input shape: {x_compressed.shape}")

        # Update the memory state by combining previous memory and new compressed input
        updated_memory = F.relu(self.fc(x_compressed) + memory)
        logger.debug(f"Updated memory shape: {updated_memory.shape}")

        return x_compressed, updated_memory


# Main SDCI Model Architecture
class SDCIModel(nn.Module):
    """
    Main model combining Sparse Information Extraction, Clustering, Masking, and Memory.
    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the SDCI model.
        Args:
            config (ModelConfig): Configuration object containing model parameters.
        """
        super(SDCIModel, self).__init__()
        self.sparse_extraction = SparseInformationExtraction(
            config.input_dim, config.sparsity
        )
        self.clustering = HierarchicalClustering(config.cluster_size)
        self.dynamic_activation = DynamicMaskingActivation(
            input_dim=config.cluster_size
            * config.input_dim,  # Match the clustered output
            hidden_dim=config.hidden_dim,
            mask_fraction=config.sparsity,
        )
        self.memory = SparseMemory(
            config.hidden_dim, config.memory_size
        )
        self.fc_out = nn.Linear(
            config.memory_size, config.num_classes
        )

    def forward(
        self, x: Tensor, memory: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the model.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            memory (Tensor): Memory tensor of shape (batch_size, memory_size).
        Returns:
            Tuple[Tensor, Tensor]: Output predictions and updated memory.
        """
        logger.debug("Starting forward pass of the model.")

        # Step 1: Sparse information extraction
        x = self.sparse_extraction(x)

        # Step 2: Hierarchical clustering
        x = self.clustering(x)

        # Step 3: Dynamic masking and activation
        x = self.dynamic_activation(x)

        # Step 4: Recursive memory update
        x, memory = self.memory(x, memory)

        # Step 5: Output layer for classification
        output = self.fc_out(memory)

        return output, memory


import time
import matplotlib.pyplot as plt

# Example configuration
config = ModelConfig(
    input_dim=128,
    num_layers=4,
    sparsity=0.5,
    cluster_size=4,
    hidden_dim=256,
    num_clusters=16,
    num_classes=10,
    memory_size=128,
)

# Initialize the model and memory
model = SDCIModel(config)


# Function to benchmark the model with different input sizes
def benchmark_model(
    model: nn.Module, input_sizes: List[int], batch_size: int = 32
):
    times = []
    memory = torch.zeros(
        batch_size, config.memory_size
    )  # Initialize memory

    for input_size in input_sizes:
        input_tensor = torch.randn(
            batch_size, input_size, config.input_dim
        )  # Generate random input

        # Measure time for forward pass
        start_time = time.time()
        with torch.no_grad():  # Disable gradients for benchmarking
            _ = model(input_tensor, memory)
        elapsed_time = time.time() - start_time

        times.append(elapsed_time)
        logger.info(
            f"Input size {input_size} - Elapsed time: {elapsed_time:.6f} seconds"
        )

    return times


# Define the range of input sizes to test
input_sizes = [128, 256, 512, 1024, 2048]

# Run the benchmark
execution_times = benchmark_model(model, input_sizes)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(
    input_sizes,
    execution_times,
    label="Model Execution Time",
    marker="o",
)
plt.plot(
    input_sizes,
    [size for size in input_sizes],
    label="Linear Time (O(N))",
    linestyle="--",
)
plt.plot(
    input_sizes,
    [size**2 for size in input_sizes],
    label="Quadratic Time (O(N^2))",
    linestyle="--",
)
plt.xlabel("Input Sequence Length (N)")
plt.ylabel("Execution Time (seconds)")
plt.title("Benchmark: Model Execution Time vs Input Size")
plt.legend()
plt.grid(True)
plt.show()
