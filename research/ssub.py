# hcen.py

import torch
import torch.nn as nn
from torch import Tensor
from loguru import logger

logger.add("hcen.log", rotation="1 MB")  # Log file configuration


class EncodingFunction(nn.Module):
    """
    Encoding function f that maps sequences of varying lengths to a fixed-dimensional vector space.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(EncodingFunction, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the encoding function.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: Encoded tensor of shape (batch_size, hidden_dim).
        """
        # Simple mean pooling followed by a linear layer
        x = x.mean(dim=1)  # Shape: (batch_size, input_dim)
        encoded = self.encoder(x)  # Shape: (batch_size, hidden_dim)
        return encoded


class ImportanceScoring(nn.Module):
    """
    Importance scoring function I(C_l) to select the most informative segments.
    """

    def __init__(self, hidden_dim: int):
        super(ImportanceScoring, self).__init__()
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute importance scores for each compressed representation.

        Args:
            x (Tensor): Tensor of shape (batch_size, num_segments, hidden_dim).

        Returns:
            Tensor: Importance scores of shape (batch_size, num_segments).
        """
        scores = self.scorer(x).squeeze(
            -1
        )  # Shape: (batch_size, num_segments)
        return scores


class AggregationFunction(nn.Module):
    """
    Aggregation function g to combine two compressed representations.
    """

    def __init__(self, hidden_dim: int):
        super(AggregationFunction, self).__init__()
        self.aggregator = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Aggregate two compressed representations.

        Args:
            x1 (Tensor): Tensor of shape (batch_size, hidden_dim).
            x2 (Tensor): Tensor of shape (batch_size, hidden_dim).

        Returns:
            Tensor: Aggregated tensor of shape (batch_size, hidden_dim).
        """
        combined = torch.cat(
            [x1, x2], dim=-1
        )  # Shape: (batch_size, hidden_dim * 2)
        aggregated = self.aggregator(
            combined
        )  # Shape: (batch_size, hidden_dim)
        return aggregated


class OutputFunction(nn.Module):
    """
    Output function h to produce the final output from the root compressed representation.
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        super(OutputFunction, self).__init__()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the final output.

        Args:
            x (Tensor): Root compressed representation of shape (batch_size, hidden_dim).

        Returns:
            Tensor: Final output tensor of shape (batch_size, output_dim).
        """
        output = self.output_layer(
            x
        )  # Shape: (batch_size, output_dim)
        return output


class HCEN(nn.Module):
    """
    Hierarchical Compressed Encoding Network (HCEN).
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, k: int
    ):
        super(HCEN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k  # Number of segments to select at each level
        self.encoding_function = EncodingFunction(
            input_dim, hidden_dim
        )
        self.importance_scoring = ImportanceScoring(hidden_dim)
        self.aggregation_function = AggregationFunction(hidden_dim)
        self.output_function = OutputFunction(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of HCEN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Tensor: Final output tensor of shape (batch_size, output_dim).
        """
        batch_size, seq_len, _ = x.size()
        logger.info(f"Input shape: {x.shape}")

        # Initialize segments with the entire sequence
        segments = [
            x
        ]  # List of tensors of shape (batch_size, seq_len_i, input_dim)
        level = 0

        while True:
            logger.info(
                f"Processing level {level} with {len(segments)} segments"
            )

            compressed_reps = []
            # Encode each segment
            for segment in segments:
                if segment.size(-1) == self.input_dim:
                    # Segment is unencoded, so encode it
                    encoded = self.encoding_function(
                        segment
                    )  # Shape: (batch_size, hidden_dim)
                    compressed_reps.append(encoded)
                elif segment.size(-1) == self.hidden_dim:
                    # Segment is already encoded
                    compressed_reps.append(segment)
                else:
                    raise ValueError(
                        f"Unexpected segment size: {segment.size()}"
                    )

            compressed_reps = torch.stack(
                compressed_reps, dim=1
            )  # Shape: (batch_size, num_segments, hidden_dim)

            # If only one compressed representation remains, we can stop
            if compressed_reps.size(1) == 1:
                root_representation = compressed_reps.squeeze(
                    1
                )  # Shape: (batch_size, hidden_dim)
                break

            # Compute importance scores
            importance_scores = self.importance_scoring(
                compressed_reps
            )  # Shape: (batch_size, num_segments)
            logger.debug(
                f"Importance scores shape: {importance_scores.shape}"
            )

            # Select top-k segments based on importance scores
            k = min(self.k, compressed_reps.size(1))
            _, indices = torch.topk(
                importance_scores, k, dim=1
            )  # Indices of top-k segments
            logger.info(f"Selected top-{k} segments at level {level}")

            # Gather selected compressed representations
            batch_indices = (
                torch.arange(batch_size).unsqueeze(-1).expand(-1, k)
            )
            selected_reps = compressed_reps[
                batch_indices, indices
            ]  # Shape: (batch_size, k, hidden_dim)

            # Aggregate selected representations pairwise
            aggregated_reps = []
            i = 0
            while i < selected_reps.size(1):
                x1 = selected_reps[
                    :, i, :
                ]  # Shape: (batch_size, hidden_dim)
                if i + 1 < selected_reps.size(1):
                    x2 = selected_reps[
                        :, i + 1, :
                    ]  # Shape: (batch_size, hidden_dim)
                    aggregated = self.aggregation_function(x1, x2)
                else:
                    # If there's an odd number of representations, carry the last one forward
                    aggregated = x1
                aggregated_reps.append(aggregated)
                i += 2

            # Prepare for next level
            segments = aggregated_reps  # Each segment is a tensor of shape (batch_size, hidden_dim)

            level += 1

        # Final output
        output = self.output_function(
            root_representation
        )  # Shape: (batch_size, output_dim)
        logger.info(f"Output shape: {output.shape}")
        return output


# test_hcen.py

# import torch
# # from hcen import HCEN
# import time
# import matplotlib.pyplot as plt

# # def test_hcen_sublinear_scaling():
# #     """
# #     Test the HCEN model to verify sub-linear computational complexity.
# #     """
# #     input_dim = 128
# #     hidden_dim = 64
# #     output_dim = 10
# #     k = 5  # Number of segments to select at each level
# #     batch_size = 32

# #     sequence_lengths = [2 ** i for i in range(5, 15)]  # Sequence lengths from 32 to 16384
# #     times = []

# #     for seq_len in sequence_lengths:
# #         model = HCEN(input_dim, hidden_dim, output_dim, k)
# #         x = torch.randn(batch_size, seq_len, input_dim)

# #         start_time = time.time()
# #         output = model(x)
# #         end_time = time.time()

# #         elapsed_time = end_time - start_time
# #         times.append(elapsed_time)
# #         print(f"Sequence Length: {seq_len}, Time Taken: {elapsed_time:.6f} seconds")

# #     # Plotting the results
# #     plt.figure(figsize=(10, 6))
# #     plt.plot(sequence_lengths, times, marker='o')
# #     plt.xlabel('Sequence Length (N)')
# #     plt.ylabel('Time Taken (seconds)')
# #     plt.title('HCEN Computational Time vs Sequence Length')
# #     plt.xscale('log')
# #     plt.yscale('log')
# #     plt.grid(True)
# #     plt.show()

# # if __name__ == "__main__":
# #     test_hcen_sublinear_scaling()


# # # Transformer Model (Quadratic Scaling)
# # class TransformerModel(nn.Module):
# #     def __init__(self, input_dim: int, num_heads: int, num_layers: int, output_dim: int):
# #         super(TransformerModel, self).__init__()
# #         self.transformer = nn.Transformer(
# #             d_model=input_dim,
# #             nhead=num_heads,
# #             num_encoder_layers=num_layers,
# #             num_decoder_layers=num_layers,
# #             dim_feedforward=4 * input_dim,
# #             batch_first=True,
# #         )
# #         self.output_layer = nn.Linear(input_dim, output_dim)

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         # Transformer requires both src and tgt; for simplicity, we'll use the same input
# #         output = self.transformer(x, x)
# #         # Take the mean across the sequence length
# #         output = output.mean(dim=1)
# #         output = self.output_layer(output)
# #         return output

# # # RNN Model (Linear Scaling)
# # class RNNModel(nn.Module):
# #     def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
# #         super(RNNModel, self).__init__()
# #         self.rnn = nn.RNN(
# #             input_size=input_dim,
# #             hidden_size=hidden_dim,
# #             num_layers=num_layers,
# #             batch_first=True,
# #         )
# #         self.output_layer = nn.Linear(hidden_dim, output_dim)

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         # RNN returns output and hidden state; we'll use the final hidden state
# #         _, hn = self.rnn(x)
# #         # hn shape: (num_layers, batch_size, hidden_dim)
# #         hn = hn[-1]  # Take the output from the last layer
# #         output = self.output_layer(hn)
# #         return output

# # def benchmark_models():
# #     """
# #     Benchmark HCEN, Transformer, and RNN models to compare computational scaling.
# #     """
# #     input_dim = 128
# #     hidden_dim = 64
# #     output_dim = 10
# #     k = 5  # Number of segments to select at each level in HCEN
# #     num_heads = 8
# #     num_layers = 2
# #     batch_size = 32

# #     sequence_lengths = [2 ** i for i in range(5, 14)]  # Sequence lengths from 32 to 8192
# #     hcen_times = []
# #     transformer_times = []
# #     rnn_times = []

# #     for seq_len in sequence_lengths:
# #         x = torch.randn(batch_size, seq_len, input_dim)

# #         # HCEN Model
# #         hcen_model = HCEN(input_dim, hidden_dim, output_dim, k)
# #         start_time = time.time()
# #         hcen_output = hcen_model(x)
# #         end_time = time.time()
# #         hcen_elapsed = end_time - start_time
# #         hcen_times.append(hcen_elapsed)

# #         # Transformer Model
# #         # transformer_model = TransformerModel(input_dim, num_heads, num_layers, output_dim)
# #         # start_time = time.time()
# #         # transformer_output = transformer_model(x)
# #         # end_time = time.time()
# #         # transformer_elapsed = end_time - start_time
# #         # transformer_times.append(transformer_elapsed)

# #         # RNN Model
# #         rnn_model = RNNModel(input_dim, hidden_dim, num_layers, output_dim)
# #         start_time = time.time()
# #         rnn_output = rnn_model(x)
# #         end_time = time.time()
# #         rnn_elapsed = end_time - start_time
# #         rnn_times.append(rnn_elapsed)

# #         print(f"Sequence Length: {seq_len}, HCEN Time: {hcen_elapsed:.6f}s, "
# #               f"RNN Time: {rnn_elapsed:.6f}s")

# #     # Plotting the results
# #     plt.figure(figsize=(12, 8))
# #     plt.plot(sequence_lengths, hcen_times, marker='o', label='HCEN (Sub-Linear)')
# #     # plt.plot(sequence_lengths, transformer_times, marker='o', label='Transformer (Quadratic)')
# #     plt.plot(sequence_lengths, rnn_times, marker='o', label='RNN (Linear)')

# #     # Reference lines for O(N), O(N log N), O(N^2)
# #     N = np.array(sequence_lengths)
# #     plt.plot(N, N / N.max() * max(hcen_times + transformer_times + rnn_times), 'k--', label='O(N)')
# #     plt.plot(N, np.log(N) / np.log(N.max()) * max(hcen_times + transformer_times + rnn_times), 'g--', label='O(log N)')
# #     plt.plot(N, (N ** 2) / (N.max() ** 2) * max(hcen_times + transformer_times + rnn_times), 'r--', label='O(N^2)')

# #     plt.xlabel('Sequence Length (N)')
# #     plt.ylabel('Time Taken (seconds)')
# #     plt.title('Model Computational Time vs Sequence Length')
# #     plt.xscale('log')
# #     plt.yscale('log')
# #     plt.legend()
# #     plt.grid(True)
# #     plt.show()

# # if __name__ == "__main__":
# #     benchmark_models()
