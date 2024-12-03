import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple


class ScalingBenchmark:
    def __init__(
        self,
        models: List[nn.Module],
        scaling_factor: float = 1.1,
        input_size_start: int = 16,
        num_tests: int = 10,
    ):
        """
        Initialize the benchmark.

        :param models: A list of models to test.
        :param scaling_factor: How much to increase input size each iteration.
        :param input_size_start: Starting size of input.
        :param num_tests: Number of tests to run.
        """
        logger.info(
            f"Initializing ScalingBenchmark with {len(models)} models"
        )

        self.models = models
        self.scaling_factor = scaling_factor
        self.input_size_start = input_size_start
        self.num_tests = num_tests

    def _generate_input(self, input_size: int) -> torch.Tensor:
        """
        Generates random input tensor of a given size.

        :param input_size: Size of the input tensor to generate.
        :return: Random tensor of shape (input_size, input_size).
        """
        logger.debug(f"Generating input tensor of size {input_size}")
        return torch.randn(input_size, input_size)

    def _test_model(
        self, model: nn.Module, input_size: int
    ) -> Tuple[float, float]:
        """
        Test a model with a specific input size and measure the forward pass time and output.

        :param model: The model to test.
        :param input_size: Size of the input tensor.
        :return: The time taken for the forward pass and the model's output mean.
        """
        logger.debug(f"Testing model with input size {input_size}")

        input_tensor = self._generate_input(input_size)

        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            output = model(input_tensor)
            end_time.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            elapsed_time = start_time.elapsed_time(
                end_time
            )  # in milliseconds
            output_mean = output.mean().item()

            logger.debug(
                f"Model test completed: elapsed time {elapsed_time} ms, output mean {output_mean}"
            )

            return elapsed_time, output_mean

    def run_benchmark(self) -> None:
        """
        Run the scaling benchmark on all models.
        Categorizes the models as linear, quadratic, or sub-linear based on performance scaling.
        """
        logger.info("Starting benchmark tests")

        performance_data = {model: [] for model in self.models}

        for i in tqdm(range(self.num_tests), desc="Benchmarking"):
            current_input_size = int(
                self.input_size_start * (self.scaling_factor**i)
            )
            logger.info(
                f"Running test {i + 1}/{self.num_tests} with input size {current_input_size}"
            )

            for model in self.models:
                elapsed_time, output_mean = self._test_model(
                    model, current_input_size
                )
                performance_data[model].append(
                    (current_input_size, elapsed_time)
                )

        self._categorize_models(performance_data)

    def _categorize_models(self, performance_data: dict) -> None:
        """
        Categorize models based on how their performance scales with input size.

        :param performance_data: Dictionary containing performance data for each model.
        """
        logger.info("Categorizing models based on scaling behavior")

        for model, data in performance_data.items():
            input_sizes, times = zip(*data)
            input_sizes = np.array(input_sizes)
            times = np.array(times)

            # Fit to a polynomial of degree 2 (quadratic), 1 (linear), or sub-linear
            quadratic_fit = np.polyfit(input_sizes, times, 2)
            linear_fit = np.polyfit(input_sizes, times, 1)

            quadratic_error = np.sum(
                (np.polyval(quadratic_fit, input_sizes) - times) ** 2
            )
            linear_error = np.sum(
                (np.polyval(linear_fit, input_sizes) - times) ** 2
            )

            logger.info(
                f"Model {model.__class__.__name__} fit results: quadratic_error={quadratic_error}, linear_error={linear_error}"
            )

            if quadratic_error < linear_error:
                logger.success(
                    f"Model {model.__class__.__name__} scales quadratically."
                )
            elif (
                linear_error < quadratic_error
                and linear_error < 0.1 * quadratic_error
            ):
                logger.success(
                    f"Model {model.__class__.__name__} scales linearly."
                )
            else:
                logger.success(
                    f"Model {model.__class__.__name__} scales sub-linearly."
                )
