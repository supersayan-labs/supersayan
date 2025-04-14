import os
import sys
import time
import pytest
import tempfile
import subprocess
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
from torch.utils.data import DataLoader, TensorDataset
import socket
from urllib.parse import urljoin

# Import from project
from supersayan.nn.convert import ModelType
from supersayan.remote.client import SupersayanClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simulated dataset: [rooms, area, location_score, age, has_garage]
X = torch.rand(500, 5, dtype=torch.float32)  # 500 houses
y = (
    50000 * X[:, 0] +     # rooms
    300 * X[:, 1] * 100 + # area
    100000 * X[:, 2] +    # location_score
    -2000 * X[:, 3] * 100 + # age
    20000 * X[:, 4]       # has_garage
).unsqueeze(1) + torch.randn(500, 1, dtype=torch.float32) * 10000  # add some noise

# DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model: simple DNN with named modules for hybrid conversion
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


def find_free_port():
    """Find a free port to use for the server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


def run_server(host, port, models_dir):
    """
    Run the SuperSayan server in a separate process.
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


def train_model():
    """
    Train the house price regressor model for testing.
    
    Returns:
        torch_model: The trained PyTorch model
    """
    # Create and train original PyTorch model
    torch_model = HousePriceRegressor()
    
    # Training setup
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(torch_model.parameters(), lr=1e-3)
    
    # Training loop - brief training just for demonstration
    logger.info("Training original PyTorch model:")
    for epoch in range(3):  # Reduced epochs for faster testing
        total_loss = 0
        batches = 0
        for batch_x, batch_y in loader:
            pred = torch_model(batch_x)
            loss = loss_fn(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        avg_loss = total_loss / batches
        logger.info(f"Epoch {epoch}: Avg Loss = {avg_loss:.2f}")
    
    return torch_model


def test_hybrid_client_server(server_fixture):
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
    
    # Train the model
    torch_model = train_model()
    
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
            
            test_hybrid_client_server(server_url)
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