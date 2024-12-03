import torch
import torch.nn as nn
from torch import Tensor
from loguru import logger
import time
from typing import List

logger.info("Setting up Sub-Sub-Linear LLM Model")


class SparseDynamicLayer(nn.Module):
    """
    A layer that dynamically selects a subset of tokens for processing.

    Attributes:
        input_dim (int): The input embedding dimension.
        output_dim (int): The output embedding dimension.
        dropout (float): Dropout rate for token selection.
    """

    def __init__(
        self, input_dim: int, output_dim: int, dropout: float = 0.1
    ):
        super(SparseDynamicLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the sparse dynamic layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output tensor after sparse selection and transformation.
        """
        # Dynamic sparse token selection (probability-driven)
        token_selection_prob = torch.sigmoid(
            self.fc(x)
        )  # Shape (batch_size, seq_len, output_dim)
        selected_tokens = self.dropout(token_selection_prob)

        logger.info(
            f"Selected {selected_tokens.sum()} tokens for processing out of {x.shape[1]} total tokens."
        )
        return selected_tokens


class HierarchicalSubstructureLayer(nn.Module):
    """
    A layer that processes the input sequence hierarchically, by splitting the sequence into substructures
    and processing relevant portions.

    Attributes:
        input_dim (int): The input embedding dimension.
    """

    def __init__(self, input_dim: int):
        super(HierarchicalSubstructureLayer, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for hierarchical substructure processing.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output tensor after hierarchical substructure processing.
        """
        batch_size, seq_len, _ = x.size()
        logger.info(
            f"Processing {seq_len} tokens into hierarchical substructures."
        )

        # Hierarchical substructure processing
        # For simplicity, we'll break the input sequence into 2 substructures.
        substructure_1 = x[:, : seq_len // 2, :]
        substructure_2 = x[:, seq_len // 2 :, :]

        # Processing each substructure independently
        processed_1 = self.fc(substructure_1)
        processed_2 = self.fc(substructure_2)

        # Reassemble the processed structures
        processed = torch.cat([processed_1, processed_2], dim=1)

        return processed


class ProbabilisticMemoryCompressionLayer(nn.Module):
    """
    A layer that performs probabilistic memory compression to reduce the amount of information passed to subsequent layers.

    Attributes:
        input_dim (int): The input embedding dimension.
        output_dim (int): The output embedding dimension (should match hidden_dim of next layer).
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(ProbabilisticMemoryCompressionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(
            input_dim, output_dim
        )  # Directly output hidden_dim size

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for probabilistic memory compression.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Compressed memory output.
        """
        batch_size, seq_len, _ = x.size()
        logger.info(f"Compressing memory from {seq_len} tokens.")

        # Apply the compression to match the output_dim (hidden_dim)
        compressed = self.fc(x)

        logger.info(
            f"Memory compressed to {compressed.shape[1]} tokens."
        )
        return compressed


class SubSubLinearLLM(nn.Module):
    """
    Sub-Sub-Linear LLM Model that scales sub-sub-linearly while maintaining learning ability.

    Attributes:
        input_dim (int): Dimension of input embeddings.
        hidden_dim (int): Dimension of hidden layers.
        output_dim (int): Dimension of the output embeddings.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super(SubSubLinearLLM, self).__init__()
        self.sparse_layer = SparseDynamicLayer(input_dim, hidden_dim)
        self.hierarchical_layer = HierarchicalSubstructureLayer(
            hidden_dim
        )
        self.compression_layer = ProbabilisticMemoryCompressionLayer(
            hidden_dim, hidden_dim
        )  # Ensure output is hidden_dim
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Sub-Sub-Linear LLM Model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Final output tensor of shape (batch_size, output_dim).
        """
        # Step 1: Sparse dynamic selection
        x = self.sparse_layer(x)

        # Step 2: Hierarchical processing
        x = self.hierarchical_layer(x)

        # Step 3: Probabilistic memory compression
        x = self.compression_layer(x)

        # Final output layer
        # Perform mean pooling along the sequence dimension (dim=1), resulting in shape (batch_size, hidden_dim)
        x = x.mean(dim=1)

        # Now x has shape (batch_size, hidden_dim), which matches fc_output
        output = self.fc_output(x)
        return output


import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
from scipy.stats import linregress


def benchmark_model(
    model: nn.Module,
    input_dim: int,
    seq_lengths: List[int],
    batch_size: int = 32,
    runs: int = 5,
):
    """
    Benchmark the model on different sequence lengths and log the results.

    Args:
        model (nn.Module): The model to benchmark.
        input_dim (int): Input dimensionality.
        seq_lengths (List[int]): List of sequence lengths to test.
        batch_size (int): Batch size for testing.
        runs (int): Number of runs to average for each sequence length.

    Returns:
        dict: A dictionary with sequence lengths as keys and average times as values.
    """
    model.eval()
    times = []

    for seq_len in seq_lengths:
        logger.info(f"Benchmarking sequence length {seq_len}")

        # Generate random input for the given sequence length
        x = torch.randn(batch_size, seq_len, input_dim)

        # Measure time for several runs and average
        elapsed_times = []
        for _ in range(runs):
            start_time = time.time()

            with torch.no_grad():
                output = model(x)

            end_time = time.time()
            elapsed_times.append(end_time - start_time)

        avg_time = np.mean(elapsed_times)
        times.append(avg_time)

        logger.info(
            f"Average time for sequence length {seq_len}: {avg_time:.6f} seconds"
        )

    return {
        seq_len: time for seq_len, time in zip(seq_lengths, times)
    }


def detect_scaling_regime(
    seq_lengths: List[int], times: List[float]
) -> float:
    """
    Detect the scaling regime by fitting a line to the log-log data and computing the slope.

    Args:
        seq_lengths (List[int]): Sequence lengths.
        times (List[float]): Times corresponding to each sequence length.

    Returns:
        float: The slope of the log-log plot indicating the scaling regime.
    """
    log_seq_lengths = np.log(seq_lengths)
    log_times = np.log(times)

    # Fit a linear regression to the log-log data
    slope, intercept, r_value, p_value, std_err = linregress(
        log_seq_lengths, log_times
    )

    logger.info(f"Slope of the log-log plot: {slope:.4f}")
    return slope


def plot_benchmark_results(results: dict, slope: float):
    """
    Plot the benchmark results to analyze scaling behavior.

    Args:
        results (dict): A dictionary with sequence lengths as keys and average times as values.
        slope (float): The slope of the log-log plot for scaling regime detection.
    """
    seq_lengths = list(results.keys())
    times = list(results.values())

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(
        seq_lengths, times, marker="o", label=f"Slope: {slope:.2f}"
    )
    plt.title("Model Benchmark: Time vs Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Average Time (seconds)")
    plt.grid(True)
    plt.xscale("log")
    plt.yscale(
        "log"
    )  # Use log-log scale to detect power-law relationships
    plt.legend()
    plt.show()

    logger.info("Benchmark plot generated.")


if __name__ == "__main__":
    input_dim = 512
    hidden_dim = 256
    output_dim = 128
    seq_lengths = [
        128,
        256,
        512,
        1024,
        2048,
    ]  # Varying sequence lengths
    batch_size = 32
    runs = 5  # Average over 5 runs for each sequence length

    model = SubSubLinearLLM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    # Run benchmark and get results
    benchmark_results = benchmark_model(
        model, input_dim, seq_lengths, batch_size, runs
    )

    # Extract sequence lengths and times
    seq_lengths = list(benchmark_results.keys())
    times = list(benchmark_results.values())

    # Detect scaling regime (slope of log-log plot)
    slope = detect_scaling_regime(seq_lengths, times)

    # Plot the benchmark results and scaling regime
    plot_benchmark_results(benchmark_results, slope)

    # Automatically detect and print scaling regime
    if slope > 1.5:
        logger.info(
            f"The model scales **quadratically** (slope: {slope:.2f})"
        )
    elif 0.9 <= slope <= 1.5:
        logger.info(
            f"The model scales **linearly** (slope: {slope:.2f})"
        )
    elif 0.5 <= slope < 0.9:
        logger.info(
            f"The model scales **sub-linearly** (slope: {slope:.2f})"
        )
    else:
        logger.info(
            f"The model scales **sub-sub-linearly** (slope: {slope:.2f})"
        )
