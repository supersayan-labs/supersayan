import logging
import os
import subprocess
import tempfile
import json
import shutil
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class DockerDeployment:
    """
    Handles deployment of SuperSayan models in Docker containers.
    
    Provides functionality to create Docker images, deploy containers,
    and manage their lifecycle.
    """
    def __init__(
        self, 
        image_name: str = "supersayan-server",
        tag: str = "latest",
        port: int = 8000,
        models_dir: str = "/tmp/supersayan/models"
    ):
        """
        Initialize Docker deployment.
        
        Args:
            image_name: Name for the Docker image
            tag: Tag for the Docker image
            port: Port to expose for the server
            models_dir: Directory for storing models
        """
        self.image_name = image_name
        self.tag = tag
        self.port = port
        self.models_dir = models_dir
        self.containers = {}  # container_id -> container_info
        
    def _create_dockerfile(self, temp_dir: str) -> str:
        """
        Create a Dockerfile for the SuperSayan server.
        
        Args:
            temp_dir: Temporary directory for building the image
            
        Returns:
            Path to the created Dockerfile
        """
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")
        
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for Julia
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
    build-essential \\
    ca-certificates \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install Julia
RUN curl -LO https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz && \\
    tar -xzf julia-1.8.5-linux-x86_64.tar.gz && \\
    rm julia-1.8.5-linux-x86_64.tar.gz && \\
    mv julia-1.8.5 /opt/julia && \\
    ln -s /opt/julia/bin/julia /usr/local/bin/julia

# Copy SuperSayan package
COPY supersayan /app/supersayan/

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install /app/supersayan

# Install FastAPI for the server
RUN pip install fastapi uvicorn

# Copy server script
COPY server.py /app/

# Create directory for models
RUN mkdir -p {self.models_dir}

# Set environment variables
ENV PYTHONPATH=/app
ENV MODELS_DIR={self.models_dir}

EXPOSE {self.port}

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "{self.port}"]
"""
        
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        return dockerfile_path
    
    def _create_server_script(self, temp_dir: str) -> str:
        """
        Create a FastAPI server script.
        
        Args:
            temp_dir: Temporary directory for building the image
            
        Returns:
            Path to the created server script
        """
        server_path = os.path.join(temp_dir, "server.py")
        
        server_content = """
import logging
import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

from supersayan.remote.server import SupersayanServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the server
models_dir = os.environ.get("MODELS_DIR", "/tmp/supersayan/models")
server = SupersayanServer(storage_dir=models_dir)

# Create FastAPI app
app = FastAPI(title="SuperSayan FHE Server")

# Request models
class UploadModelRequest(BaseModel):
    session_id: str
    model_data: str

class InferenceRequest(BaseModel):
    session_id: str
    encrypted_input: str

class SessionRequest(BaseModel):
    session_id: str

# API routes
@app.post("/models/upload")
async def upload_model(request: UploadModelRequest):
    response = server.handle_upload_model(
        request.session_id,
        request.model_data
    )
    
    if isinstance(response, tuple):
        response_data, status_code = response
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=response_data["error"])
        return response_data
    return response

@app.get("/models/{model_id}/structure")
async def get_model_structure(model_id: str):
    response = server.handle_get_model_structure(model_id)
    
    if isinstance(response, tuple):
        response_data, status_code = response
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=response_data["error"])
        return response_data
    return response

@app.post("/inference/{model_id}/{layer_name}")
async def inference(model_id: str, layer_name: str, request: InferenceRequest):
    response = server.handle_inference(
        request.session_id,
        model_id,
        layer_name,
        request.encrypted_input
    )
    
    if isinstance(response, tuple):
        response_data, status_code = response
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=response_data["error"])
        return response_data
    return response

@app.post("/sessions/{session_id}/close")
async def close_session(session_id: str, request: SessionRequest):
    response = server.handle_close_session(request.session_id)
    
    if isinstance(response, tuple):
        response_data, status_code = response
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=response_data["error"])
        return response_data
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        
        with open(server_path, "w") as f:
            f.write(server_content)
        
        return server_path
    
    def build_image(self, supersayan_path: Optional[str] = None) -> str:
        """
        Build a Docker image for the SuperSayan server.
        
        Args:
            supersayan_path: Path to the SuperSayan package (defaults to current directory)
            
        Returns:
            Image ID
            
        Raises:
            RuntimeError: If the build fails
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Dockerfile
            dockerfile_path = self._create_dockerfile(temp_dir)
            
            # Create server script
            server_path = self._create_server_script(temp_dir)
            
            # Copy SuperSayan package
            if supersayan_path is None:
                supersayan_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../../..")
                )
            
            temp_supersayan_dir = os.path.join(temp_dir, "supersayan")
            shutil.copytree(supersayan_path, temp_supersayan_dir)
            
            # Copy requirements.txt
            requirements_path = os.path.join(supersayan_path, "requirements.txt")
            if os.path.exists(requirements_path):
                shutil.copy(requirements_path, os.path.join(temp_dir, "requirements.txt"))
            else:
                # Create minimal requirements if not found
                with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
                    f.write("numpy\ntorch\njulia\n")
            
            # Build image
            image_tag = f"{self.image_name}:{self.tag}"
            logger.info(f"Building Docker image {image_tag}")
            
            try:
                subprocess.check_call(
                    ["docker", "build", "-t", image_tag, temp_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Get image ID
                result = subprocess.check_output(
                    ["docker", "images", "-q", image_tag]
                ).decode("utf-8").strip()
                
                logger.info(f"Successfully built Docker image {image_tag} with ID {result}")
                return result
            except subprocess.CalledProcessError as e:
                logger.error(f"Docker build failed: {e}")
                raise RuntimeError(f"Failed to build Docker image: {e}")
    
    def run_container(self, image_id: Optional[str] = None) -> str:
        """
        Run a Docker container with the SuperSayan server.
        
        Args:
            image_id: ID of the image to run (builds a new image if not provided)
            
        Returns:
            Container ID
            
        Raises:
            RuntimeError: If the container fails to start
        """
        # Build image if not provided
        if image_id is None:
            image_id = self.build_image()
        
        image_tag = f"{self.image_name}:{self.tag}"
        
        # Run container
        logger.info(f"Starting Docker container for {image_tag}")
        
        try:
            container_id = subprocess.check_output(
                [
                    "docker", "run", "-d",
                    "-p", f"{self.port}:{self.port}",
                    "--name", f"supersayan-server-{os.urandom(4).hex()}",
                    image_tag
                ]
            ).decode("utf-8").strip()
            
            logger.info(f"Started container {container_id} on port {self.port}")
            
            # Store container info
            self.containers[container_id] = {
                "image_id": image_id,
                "port": self.port
            }
            
            return container_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker run failed: {e}")
            raise RuntimeError(f"Failed to start Docker container: {e}")
    
    def stop_container(self, container_id: str) -> bool:
        """
        Stop a Docker container.
        
        Args:
            container_id: ID of the container to stop
            
        Returns:
            True if the container was stopped, False otherwise
        """
        if container_id not in self.containers:
            logger.warning(f"Container {container_id} not found")
            return False
        
        try:
            subprocess.check_call(
                ["docker", "stop", container_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            subprocess.check_call(
                ["docker", "rm", container_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            del self.containers[container_id]
            logger.info(f"Stopped and removed container {container_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop container {container_id}: {e}")
            return False