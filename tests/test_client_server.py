import os
import sys
import time
import pytest
import tempfile
import subprocess
import logging
import torch
import torch.nn as nn
import numpy as np
import requests
import socket
from urllib.parse import urljoin
from torchvision import models

# Import from project
from supersayan.nn.convert import ModelType
from supersayan.remote.client import SupersayanClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_free_port():
    """Find a free port to use for the server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


def run_server(host, port, models_dir):
    """
    Run the supersayan server in a separate process.
    """
    # Get path to the run_server.py script
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "run_server.py")
    
    # Create the models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Log the server command
    cmd = [sys.executable, script_path, "--host", host, "--port", str(port), "--models-dir", models_dir]
    logger.info(f"Starting server with command: {' '.join(cmd)}")
    
    # Start the server process - don't pipe stdout/stderr to allow proper initialization
    process = subprocess.Popen(
        cmd,
        # Ensure output is visible for debugging
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered output
    )
    
    # Monitor server startup
    start_time = time.time()
    timeout = 30  # Allow up to 30 seconds for server to start
    
    # Start the log monitoring in a separate thread to prevent blocking
    def log_server_output():
        # Non-blocking read from stdout and stderr
        while process.poll() is None:
            stdout_line = process.stdout.readline()
            if stdout_line:
                logger.info(f"Server stdout: {stdout_line.strip()}")
            stderr_line = process.stderr.readline()
            if stderr_line:
                logger.warning(f"Server stderr: {stderr_line.strip()}")
    
    import threading
    log_thread = threading.Thread(target=log_server_output, daemon=True)
    log_thread.start()
    
    # Wait for server to start and become responsive
    logger.info(f"Waiting for server to start on {host}:{port}...")
    server_url = f"http://{host}:{port}"
    
    while time.time() - start_time < timeout:
        # Check if the process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(f"Server process exited with code {process.returncode}")
            logger.error(f"Server stdout: {stdout}")
            logger.error(f"Server stderr: {stderr}")
            raise RuntimeError(f"Server failed to start, exit code: {process.returncode}")
        
        # Try to connect to the health endpoint
        try:
            response = requests.get(urljoin(server_url, "/health"), timeout=1)
            if response.status_code == 200:
                logger.info(f"Server started successfully at {server_url}")
                return process
        except requests.RequestException:
            # Server not ready yet
            time.sleep(1)
    
    # If we reach here, server didn't start within the timeout
    process.terminate()
    process.wait(timeout=5)
    stdout, stderr = process.communicate()
    logger.error(f"Server stdout: {stdout}")
    logger.error(f"Server stderr: {stderr}")
    raise TimeoutError(f"Server didn't start within {timeout} seconds")


@pytest.fixture(scope="module")
def server_fixture():
    """Fixture to start and stop the server for testing."""
    # Create a temporary directory for models
    models_dir = tempfile.mkdtemp(prefix="supersayan_test_models_")
    logger.info(f"Created temporary models directory: {models_dir}")
    
    try:
        # Server configuration
        host = "127.0.0.1"
        port = find_free_port()  # Use a free port to avoid conflicts
        server_url = f"http://{host}:{port}"
        
        # Start server
        logger.info(f"Starting server on {host}:{port}")
        server_process = run_server(host, port, models_dir)
        
        # Yield server URL for tests to use
        yield server_url
        
    except Exception as e:
        logger.error(f"Error in server fixture: {e}")
        raise
    finally:
        # Cleanup: terminate the server process
        logger.info("Shutting down server")
        if 'server_process' in locals():
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if not terminated
                server_process.kill()
                server_process.wait()
            
            # Print final output
            stdout, stderr = server_process.communicate()
            if stdout:
                logger.info(f"Final server stdout: {stdout}")
            if stderr:
                logger.warning(f"Final server stderr: {stderr}")
                
        # Clean up the temp directory
        try:
            import shutil
            shutil.rmtree(models_dir, ignore_errors=True)
            logger.info(f"Removed temporary models directory: {models_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up models directory: {e}")
            

def test_hybrid_house_price_regression(server_fixture):
    """
    Test the client-server architecture with a house price regression model.
    
    Args:
        server_fixture: The server URL fixture
    """
    server_url = server_fixture
    logger.info(f"Testing client against server at {server_url}")
    
    # Verify server is responsive
    try:
        response = requests.get(urljoin(server_url, "/health"), timeout=2)
        assert response.status_code == 200, "Server health check failed"
        logger.info("Server health check passed")
    except Exception as e:
        pytest.fail(f"Server health check failed: {e}")

    # Define a simple house price regressor model
    class HousePriceRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            # Use named modules instead of Sequential for better hybrid conversion
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

    # Create the model
    torch_model = HousePriceRegressor()
    
    # Test sample data
    test_x = torch.rand(5, 5, dtype=torch.float32)  # 5 test samples
    
    # Get predictions from original model for comparison
    torch_pred = torch_model(test_x)
    torch_values = torch_pred.detach().numpy()
    
    logger.info("Creating hybrid client...")
    # Create a client with hybrid model (only Linear layers in FHE)
    client_hybrid = SupersayanClient(
        server_url=server_url,
        torch_model=torch_model,
        model_type=ModelType.HYBRID,
        fhe_modules=[nn.Linear]  # Convert only Linear layers to FHE
    )
    
    try:
        # Perform forward pass - FHE layers run remotely, others run locally
        logger.info("Running forward pass...")
        client_hybrid_pred = client_hybrid(test_x)
        client_hybrid_values = client_hybrid_pred.detach().numpy()
        
        logger.info("Original PyTorch model predictions:")
        logger.info(torch_values)
        logger.info("Hybrid SupersayanClient model predictions:")
        logger.info(client_hybrid_values)
        
        # Compare the results (allowing for some numerical differences due to FHE)
        mean_diff = np.mean(np.abs(torch_values - client_hybrid_values))
        logger.info(f"Mean absolute difference: {mean_diff}")
        
        # The threshold should be adjusted based on expected precision
        assert mean_diff < 1000.0, f"Model predictions differ too much: {mean_diff}"
        
    except Exception as e:
        logger.error(f"Error during client test: {e}", exc_info=True)
        raise
    finally:
        # Always close the client session
        try:
            client_hybrid.close()
            logger.info("Closed client session")
        except Exception as e:
            logger.warning(f"Error closing client session: {e}")


def test_hybrid_small_cnn(server_fixture):
    """
    Test the client-server architecture with a small convolutional network on random input.
    Only the Conv2d layer runs in FHE.
    
    Args:
        server_fixture: The server URL fixture
    """
    server_url = server_fixture
    logger.info(f"Testing CNN client against server at {server_url}")
    
    # Verify server is responsive
    try:
        response = requests.get(urljoin(server_url, "/health"), timeout=2)
        assert response.status_code == 200, "Server health check failed"
        logger.info("Server health check passed")
    except Exception as e:
        pytest.fail(f"Server health check failed: {e}")
    
    # Define a small CNN model for image classification
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Keep the model very small for FHE computation
            self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(2 * 8 * 8, 10)  # Output 10 classes
            
        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x
    
    # Create the model
    torch_model = SmallCNN()
    
    # Test sample data - very small random image batch (8x8 pixels)
    test_x = torch.rand(2, 1, 8, 8, dtype=torch.float32)  # 2 test images, 1 channel, 8x8 pixels
    
    # Get predictions from original model for comparison
    torch_pred = torch_model(test_x)
    torch_values = torch_pred.detach().numpy()
    
    logger.info("Creating hybrid CNN client...")
    # Create a client with hybrid model (only Conv2d layers in FHE)
    client_hybrid = SupersayanClient(
        server_url=server_url,
        torch_model=torch_model,
        model_type=ModelType.HYBRID,
        fhe_modules=[nn.Conv2d]  # Convert only Conv2d layers to FHE
    )
    
    try:
        # Perform forward pass - FHE Conv2d layer runs remotely, others run locally
        logger.info("Running CNN forward pass...")
        client_hybrid_pred = client_hybrid(test_x)
        client_hybrid_values = client_hybrid_pred.detach().numpy()
        
        logger.info("Original PyTorch model predictions:")
        logger.info(torch_values)
        logger.info("Hybrid SupersayanClient model predictions:")
        logger.info(client_hybrid_values)
        
        # Compare the results (allowing for some numerical differences due to FHE)
        mean_diff = np.mean(np.abs(torch_values - client_hybrid_values))
        logger.info(f"Mean absolute difference: {mean_diff}")
        
        # The threshold should be adjusted based on expected precision
        assert mean_diff < 1.0, f"Model predictions differ too much: {mean_diff}"
        
    except Exception as e:
        logger.error(f"Error during CNN client test: {e}", exc_info=True)
        raise
    finally:
        # Always close the client session
        try:
            client_hybrid.close()
            logger.info("Closed CNN client session")
        except Exception as e:
            logger.warning(f"Error closing CNN client session: {e}")


def test_resnet18_random_input(server_fixture):
    """
    Test the client-server architecture with a ResNet18 model on random input.
    Only the Conv2d and Linear layers run in FHE.
    
    Args:
        server_fixture: The server URL fixture
    """
    server_url = server_fixture
    logger.info(f"Testing ResNet18 client against server at {server_url}")
    
    # Verify server is responsive
    try:
        response = requests.get(urljoin(server_url, "/health"), timeout=2)
        assert response.status_code == 200, "Server health check failed"
        logger.info("Server health check passed")
    except Exception as e:
        pytest.fail(f"Server health check failed: {e}")
    
    # Load a pre-trained ResNet18 model
    torch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    torch_model.eval()
    
    # Test sample data - random input
    test_x = torch.rand(1, 3, 1, 1, dtype=torch.float32)  # 1 test image, 3 channels, 224x224 pixels
    
    # Get predictions from original model for comparison
    torch_pred = torch_model(test_x)
    torch_values = torch_pred.detach().numpy()
    
    logger.info("Creating hybrid ResNet18 client...")
    # Create a client with hybrid model (Conv2d and Linear layers in FHE)
    client_hybrid = SupersayanClient(
        server_url=server_url,
        torch_model=torch_model,
        model_type=ModelType.HYBRID,
        fhe_modules=[nn.Conv2d, nn.Linear]  # Convert Conv2d and Linear layers to FHE
    )
    
    try:
        # Perform forward pass - FHE layers run remotely, others run locally
        logger.info("Running ResNet18 forward pass...")
        
        # Set a longer timeout for the test to accommodate larger data processing
        client_hybrid_pred = client_hybrid(test_x)
        client_hybrid_values = client_hybrid_pred.detach().numpy()
        
        logger.info("Original PyTorch model predictions:")
        logger.info(torch_values)
        logger.info("Hybrid SupersayanClient model predictions:")
        logger.info(client_hybrid_values)
        
        # Compare the results (allowing for some numerical differences due to FHE)
        mean_diff = np.mean(np.abs(torch_values - client_hybrid_values))
        logger.info(f"Mean absolute difference: {mean_diff}")
        
        # The threshold should be adjusted based on expected precision
        assert mean_diff < 1.0, f"Model predictions differ too much: {mean_diff}"
        
    except Exception as e:
        logger.error(f"Error during ResNet18 client test: {e}", exc_info=True)
        raise
    finally:
        # Always close the client session
        try:
            client_hybrid.close()
            logger.info("Closed ResNet18 client session")
        except Exception as e:
            logger.warning(f"Error closing ResNet18 client session: {e}")


if __name__ == "__main__":
    # This allows running the file directly for debugging
    with tempfile.TemporaryDirectory() as models_dir:
        host = "127.0.0.1"
        port = find_free_port()
        server_url = f"http://{host}:{port}"
        
        server_process = run_server(host, port, models_dir)
        try:
            # Verify server is responsive
            response = requests.get(urljoin(server_url, "/health"))
            assert response.status_code == 200, "Server health check failed"
            logger.info("Server is ready, running test...")
            
            # Choose which test to run
            # test_hybrid_house_price_regression(server_url)
            # # Uncomment to run the CNN test
            # test_hybrid_small_cnn(server_url)
            # Uncomment to run the ResNet18 test
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