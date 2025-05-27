import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import models

from supersayan.logging_config import configure_logging, get_logger
from supersayan.remote.client import SupersayanClient

configure_logging(level="INFO", disable_file_logging=True)

logger = get_logger(__name__)


def find_free_port():
    """Find a free port to use for the server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def run_server(host, port, models_dir):
    """Run the supersayan server in a separate process."""
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scripts", "run_server.py"
    )

    os.makedirs(models_dir, exist_ok=True)

    cmd = [
        sys.executable,
        script_path,
        "--host",
        host,
        "--port",
        str(port),
        "--models-dir",
        models_dir,
    ]
    logger.info(f"Starting server with command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    start_time = time.time()
    timeout = 30
    server_url = f"http://{host}:{port}"
    startup_message = f"Uvicorn running on http://{host}:{port}"

    logger.info(f"Waiting for server to start on {host}:{port}...")

    while time.time() - start_time < timeout:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(f"Server process exited with code {process.returncode}")
            logger.error(f"Server stdout: {stdout}")
            logger.error(f"Server stderr: {stderr}")
            raise RuntimeError(
                f"Server failed to start, exit code: {process.returncode}"
            )

        line = process.stderr.readline().strip()
        if line:
            logger.warning(f"Server stderr: {line}")
            if startup_message in line:
                logger.info(f"Server started successfully at {server_url}")
                break

    def log_server_output():
        while process.poll() is None:
            stdout = process.stdout.readline()
            if stdout:
                logger.info(f"Server stdout: {stdout.strip()}")
            stderr = process.stderr.readline()
            if stderr:
                logger.warning(f"Server stderr: {stderr.strip()}")

    log_thread = threading.Thread(target=log_server_output, daemon=True)
    log_thread.start()

    if time.time() - start_time < timeout:
        return process

    process.terminate()
    process.wait(timeout=5)
    stdout, stderr = process.communicate()
    logger.error(f"Server stdout: {stdout}")
    logger.error(f"Server stderr: {stderr}")
    raise TimeoutError(f"Server didn't start within {timeout} seconds")


@pytest.fixture(scope="module")
def server_fixture():
    """Fixture to start and stop the server for testing."""
    models_dir = tempfile.mkdtemp(prefix="supersayan_test_models_")
    logger.info(f"Created temporary models directory: {models_dir}")

    try:
        host = "127.0.0.1"
        port = find_free_port()
        server_url = f"http://{host}:{port}"

        logger.info(f"Starting server on {host}:{port}")
        server_process = run_server(host, port, models_dir)

        yield server_url

    except Exception as e:
        logger.error(f"Error in server fixture: {e}")
        raise
    finally:
        logger.info("Shutting down server")
        if "server_process" in locals():
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()

            stdout, stderr = server_process.communicate()
            if stdout:
                logger.info(f"Final server stdout: {stdout}")
            if stderr:
                logger.warning(f"Final server stderr: {stderr}")

        try:
            import shutil

            shutil.rmtree(models_dir, ignore_errors=True)
            logger.info(f"Removed temporary models directory: {models_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up models directory: {e}")


def test_hybrid_house_price_regression(server_fixture):
    """Test the client-server architecture with a house price regression model."""
    server_url = server_fixture
    logger.info(f"Testing client against server at {server_url}")

    class HousePriceRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(5, 16)
            self.relu1 = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.linear2 = nn.Linear(16, 8)
            self.relu2 = nn.ReLU()
            self.linear3 = nn.Linear(8, 1)

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu1(x)
            x = self.dropout(x)
            x = self.linear2(x)
            x = self.relu2(x)
            x = self.linear3(x)
            return x

    torch_model = HousePriceRegressor()
    test_x = torch.rand(5, 5, dtype=torch.float32)

    torch_pred = torch_model(test_x)
    torch_values = torch_pred.detach().numpy()

    logger.info("Creating hybrid client...")
    client_hybrid = SupersayanClient(
        server_url=server_url,
        torch_model=torch_model,
        fhe_modules=[nn.Linear],
    )

    try:
        logger.info("Running forward pass...")
        client_hybrid_pred = client_hybrid(test_x)
        client_hybrid_values = client_hybrid_pred.detach().numpy()

        logger.info("Original PyTorch model predictions:")
        logger.info(torch_values)
        logger.info("Hybrid SupersayanClient model predictions:")
        logger.info(client_hybrid_values)

        mean_diff = np.mean(np.abs(torch_values - client_hybrid_values))
        logger.info(f"Mean absolute difference: {mean_diff}")

        assert mean_diff < 1000.0, f"Model predictions differ too much: {mean_diff}"

    except Exception as e:
        logger.error(f"Error during client test: {e}", exc_info=True)
        raise


def test_hybrid_small_cnn(server_fixture):
    """Test the client-server architecture with a small convolutional network."""
    server_url = server_fixture
    logger.info(f"Testing CNN client against server at {server_url}")

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1
            )
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(2 * 8 * 8, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

    torch_model = SmallCNN()
    test_x = torch.rand(2, 1, 8, 8, dtype=torch.float32)

    torch_pred = torch_model(test_x)
    torch_values = torch_pred.detach().numpy()

    logger.info("Creating hybrid CNN client...")
    client_hybrid = SupersayanClient(
        server_url=server_url,
        torch_model=torch_model,
        fhe_modules=[nn.Conv2d],
    )

    try:
        logger.info("Running CNN forward pass...")
        client_hybrid_pred = client_hybrid(test_x)
        client_hybrid_values = client_hybrid_pred.detach().numpy()

        logger.info("Original PyTorch model predictions:")
        logger.info(torch_values)
        logger.info("Hybrid SupersayanClient model predictions:")
        logger.info(client_hybrid_values)

        mean_diff = np.mean(np.abs(torch_values - client_hybrid_values))
        logger.info(f"Mean absolute difference: {mean_diff}")

        assert mean_diff < 1.0, f"Model predictions differ too much: {mean_diff}"

    except Exception as e:
        logger.error(f"Error during CNN client test: {e}", exc_info=True)
        raise


def test_resnet18_random_input(server_fixture):
    """Test the client-server architecture with a ResNet18 model on random input."""
    server_url = server_fixture
    logger.info(f"Testing ResNet18 client against server at {server_url}")

    torch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    torch_model.eval()

    test_x = torch.rand(1, 3, 1, 1, dtype=torch.float32)

    torch_pred = torch_model(test_x)
    torch_values = torch_pred.detach().numpy()

    logger.info("Creating hybrid ResNet18 client...")
    client_hybrid = SupersayanClient(
        server_url=server_url,
        torch_model=torch_model,
        fhe_modules=[nn.Conv2d, nn.Linear],
    )

    try:
        logger.info("Running ResNet18 forward pass...")
        client_hybrid_pred = client_hybrid(test_x)
        client_hybrid_values = client_hybrid_pred.detach().numpy()

        logger.info("Original PyTorch model predictions:")
        logger.info(torch_values)
        logger.info("Hybrid SupersayanClient model predictions:")
        logger.info(client_hybrid_values)

        mean_diff = np.mean(np.abs(torch_values - client_hybrid_values))
        logger.info(f"Mean absolute difference: {mean_diff}")

        assert mean_diff < 1.0, f"Model predictions differ too much: {mean_diff}"

    except Exception as e:
        logger.error(f"Error during ResNet18 client test: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as models_dir:
        host = "127.0.0.1"
        port = find_free_port()
        server_url = f"http://{host}:{port}"

        server_process = run_server(host, port, models_dir)

        try:
            test_resnet18_random_input(server_url)
            logger.info("Test completed successfully")
        except Exception as e:
            logger.error(f"Error running test: {e}", exc_info=True)
        finally:
            logger.info("Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
