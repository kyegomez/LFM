import torch
import torch.nn as nn
from loguru import logger

logger.add("liquid_neural_net.log", rotation="500 MB", level="INFO")


class LiquidNeuron(nn.Module):
    """
    A single neuron in a liquid neural network with time-varying dynamics.

    Attributes:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        tau (float): Time constant to control the neuron dynamics.
    """

    def __init__(
        self, input_size: int, hidden_size: int, tau: float = 0.1
    ):
        """
        Initialize the LiquidNeuron with the given input and hidden size.

        Args:
            input_size (int): Size of the input.
            hidden_size (int): Size of the hidden state.
            tau (float): Time constant that controls the update speed of the neuron state.
        """
        super(LiquidNeuron, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau  # Time constant for neuron dynamics

        # Parameters: weights and biases for input and hidden connections
        self.W_input = nn.Parameter(
            torch.randn(hidden_size, input_size)
        )
        self.W_hidden = nn.Parameter(
            torch.randn(hidden_size, hidden_size)
        )
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        # Initial hidden state (zero-initialized)
        self.state = torch.zeros(hidden_size)

    def forward(
        self, x: torch.Tensor, previous_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the liquid neuron.

        The state of the neuron evolves dynamically based on the input and the previous state.

        Equation: s(t+1) = (1 - tau) * s(t) + tau * tanh(W_input * x(t) + W_hidden * s(t) + b)
        Reference: Hasani, Ramin, et al. "Liquid time-constant networks" (2021).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            previous_state (torch.Tensor): The previous state of the neuron.

        Returns:
            torch.Tensor: The updated state of the neuron.
        """
        # Dynamic state update based on a differential equation for liquid neuron behavior
        new_state = (
            1 - self.tau
        ) * previous_state + self.tau * torch.tanh(
            torch.matmul(x, self.W_input.T)
            + torch.matmul(previous_state, self.W_hidden.T)
            + self.bias
        )
        return new_state


class LiquidRNN(nn.Module):
    """
    A recurrent neural network (RNN) built using liquid neurons.

    Attributes:
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden state.
        output_size (int): Size of the output (vocabulary size).
        tau (float): Time constant for neuron dynamics.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        tau: float = 0.1,
    ):
        """
        Initialize the LiquidRNN with the given input size, hidden size, and output size.

        Args:
            input_size (int): Size of the input.
            hidden_size (int): Size of the hidden state.
            output_size (int): Size of the output (vocabulary size).
            tau (float): Time constant for neuron dynamics (controls neuron update speed).
        """
        super(LiquidRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # Liquid neuron layer
        self.liquid_neuron = LiquidNeuron(
            input_size, hidden_size, tau
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Initialize hidden state
        self.hidden_state = torch.zeros(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LiquidRNN.

        Processes each timestep sequentially, evolving hidden states based on the liquid neuron dynamics.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_size).
        """
        batch_size, seq_len, _ = x.shape
        outputs = []
        hidden_state = self.hidden_state

        logger.info(
            f"Starting forward pass with batch_size: {batch_size}, sequence_length: {seq_len}"
        )

        for t in range(seq_len):
            hidden_state = self.liquid_neuron(
                x[:, t, :], hidden_state
            )
            output = self.output_layer(hidden_state)
            outputs.append(output)

        return torch.stack(outputs, dim=1)

    def generate_text(
        self, start_token: torch.Tensor, max_len: int = 100
    ) -> str:
        """
        Generates text using the trained LiquidRNN model.

        Args:
            start_token (torch.Tensor): The starting token for text generation.
            max_len (int): Maximum length of the generated sequence.

        Returns:
            str: The generated text as a string of tokens.
        """
        generated_tokens = [start_token.item()]
        hidden_state = self.hidden_state.unsqueeze(0)

        logger.info(f"Generating text with max length {max_len}")

        # Generate text by predicting one token at a time
        for _ in range(max_len - 1):
            output = self(
                start_token.unsqueeze(0).unsqueeze(0)
            )  # Add batch and sequence dimensions
            next_token = torch.argmax(output, dim=-1)
            generated_tokens.append(next_token.item())
            start_token = next_token.squeeze(0)

        return "".join(map(str, generated_tokens))


# Assuming the LiquidRNN class has been defined as shown earlier
# Here is a simple forward pass on CPU without using GPUs.


def cpu_forward_pass_example():
    """
    Performs a forward pass with the LiquidRNN model using a CPU.
    """
    logger.info("Starting forward pass on CPU...")

    # Example configuration
    input_size = 128  # Input size (e.g., embedding dimension or one-hot encoding size)
    hidden_size = 256  # Size of the hidden state
    output_size = 128  # Output size (e.g., vocabulary size)

    # Create a dummy input tensor (batch_size=2, sequence_length=10, input_size=128)
    batch_size = 2
    sequence_length = 10
    dummy_input = torch.randn(batch_size, sequence_length, input_size)

    # Initialize the LiquidRNN model
    model = LiquidRNN(input_size, hidden_size, output_size)

    # Move the model to CPU (this is already the default)
    device = torch.device("cpu")
    model = model.to(device)

    # Perform the forward pass on the dummy input
    output = model(dummy_input)

    # Log output information
    logger.info(
        f"Output shape: {output.shape}"
    )  # Output shape should be (batch_size, sequence_length, output_size)
    logger.info("Forward pass on CPU completed.")

    return output


# Run the CPU forward pass example
output = cpu_forward_pass_example()

# Output will be printed in the logs
