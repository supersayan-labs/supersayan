import logging
import uuid
import math
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

logger = logging.getLogger(__name__)

# Default chunk size: 5MB
DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024

class ChunkManager:
    """
    Unified manager for chunked data transfers.
    
    Handles both sending and receiving chunked data with a simple interface.
    """
    def __init__(self):
        """Initialize a chunk manager."""
        # Dictionary to store transfers: transfer_id -> {chunks, metadata}
        self.transfers = {}
    
    def needs_chunking(self, data: str) -> bool:
        """
        Check if data needs to be chunked.
        
        Args:
            data: The data to check
            
        Returns:
            True if data exceeds DEFAULT_CHUNK_SIZE, False otherwise
        """
        return len(data) > DEFAULT_CHUNK_SIZE
    
    def create_transfer(self, data: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Create a new chunked transfer from data.
        
        Args:
            data: Data to split into chunks
            
        Returns:
            Tuple containing:
                - transfer_id: Unique ID for this chunked transfer
                - chunks: List of dictionaries, each containing chunk data and metadata
        """
        # Generate a transfer ID
        transfer_id = str(uuid.uuid4())
        
        # Calculate number of chunks needed
        total_chunks = math.ceil(len(data) / DEFAULT_CHUNK_SIZE)
        
        # Create chunks
        chunks = []
        for i in range(total_chunks):
            start = i * DEFAULT_CHUNK_SIZE
            end = min(start + DEFAULT_CHUNK_SIZE, len(data))
            
            chunks.append({
                "transfer_id": transfer_id,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "data": data[start:end]
            })
        
        logger.info(f"Created transfer {transfer_id} with {total_chunks} chunks")
        return transfer_id, chunks
    
    def register_transfer(self, transfer_id: str, total_chunks: int, metadata: Dict[str, Any] = None) -> None:
        """
        Register a new incoming transfer.
        
        Args:
            transfer_id: The ID of the transfer
            total_chunks: Expected total number of chunks
            metadata: Optional metadata to store with the transfer
        """
        self.transfers[transfer_id] = {
            "chunks": {},
            "total_chunks": total_chunks,
            "received_chunks": 0,
            **(metadata or {})
        }
    
    def add_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Add a chunk to an existing transfer.
        
        Args:
            chunk: A chunk dictionary with transfer_id, chunk_index, total_chunks, and data
            
        Returns:
            True if all chunks have been received, False otherwise
        """
        transfer_id = chunk.get("transfer_id")
        chunk_index = chunk.get("chunk_index")
        total_chunks = chunk.get("total_chunks")
        data = chunk.get("data")
        
        if not transfer_id or chunk_index is None or not data:
            logger.error("Invalid chunk: missing required fields")
            return False
        
        # Initialize transfer if not already registered
        if transfer_id not in self.transfers:
            self.register_transfer(transfer_id, total_chunks)
        
        # Get transfer record
        transfer = self.transfers[transfer_id]
        
        # Store the chunk if not already received
        if chunk_index not in transfer["chunks"]:
            transfer["chunks"][chunk_index] = data
            transfer["received_chunks"] += 1
        
        # Return True if all chunks received
        return transfer["received_chunks"] == transfer["total_chunks"]
    
    def get_assembled_data(self, transfer_id: str) -> Optional[str]:
        """
        Get assembled data for a transfer if all chunks received.
        
        Args:
            transfer_id: ID of the transfer
            
        Returns:
            Assembled data string or None if not all chunks received
        """
        if transfer_id not in self.transfers:
            return None
            
        transfer = self.transfers[transfer_id]
        
        # Check if all chunks received
        if transfer["received_chunks"] != transfer["total_chunks"]:
            return None
        
        # Assemble data from chunks
        chunks = transfer["chunks"]
        total_chunks = transfer["total_chunks"]
        
        assembled_data = ""
        for i in range(total_chunks):
            if i in chunks:
                assembled_data += chunks[i]
            else:
                logger.error(f"Missing chunk {i} for transfer {transfer_id}")
                return None
        
        return assembled_data
    
    def get_metadata(self, transfer_id: str) -> Dict[str, Any]:
        """
        Get metadata for a transfer.
        
        Args:
            transfer_id: ID of the transfer
            
        Returns:
            Dictionary of metadata for the transfer, or empty dict if not found
        """
        if transfer_id not in self.transfers:
            return {}
            
        # Return all keys except chunks
        return {k: v for k, v in self.transfers[transfer_id].items() 
                if k not in ["chunks", "received_chunks", "total_chunks"]}
    
    def cleanup_transfer(self, transfer_id: str) -> bool:
        """
        Clean up a transfer.
        
        Args:
            transfer_id: ID of the transfer to clean up
            
        Returns:
            True if transfer was found and removed, False otherwise
        """
        if transfer_id in self.transfers:
            del self.transfers[transfer_id]
            return True
        return False
    
    def send_chunked(self, 
                     data: str, 
                     send_chunk_fn: Callable[[Dict[str, Any]], Any],
                     metadata: Dict[str, Any] = None) -> Tuple[str, Any]:
        """
        Send data in chunks using the provided send function.
        
        Args:
            data: Data to send
            send_chunk_fn: Function to call to send each chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            Tuple containing:
                - transfer_id: ID of the transfer
                - result: Return value from the last chunk sent
        """
        # Create transfer
        transfer_id, chunks = self.create_transfer(data)
        
        # Add metadata to each chunk if provided
        if metadata:
            for chunk in chunks:
                chunk.update(metadata)
        
        # Send each chunk
        result = None
        for chunk in chunks:
            result = send_chunk_fn(chunk)
        
        return transfer_id, result
    
    def receive_chunked(self, 
                       transfer_id: str, 
                       total_chunks: int,
                       get_chunk_fn: Callable[[int], Dict[str, Any]],
                       metadata: Dict[str, Any] = None) -> str:
        """
        Receive chunked data by requesting each chunk.
        
        Args:
            transfer_id: ID of the transfer
            total_chunks: Total number of chunks to receive
            get_chunk_fn: Function to call to get each chunk, takes chunk_index
            metadata: Optional metadata to store with the transfer
            
        Returns:
            Assembled data
        """
        # Register the transfer
        self.register_transfer(transfer_id, total_chunks, metadata)
        
        # Request each chunk
        for i in range(total_chunks):
            chunk_data = get_chunk_fn(i)
            
            # Create proper chunk structure
            chunk = {
                "transfer_id": transfer_id,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "data": chunk_data.get("data", "")
            }
            
            # Add to the transfer
            self.add_chunk(chunk)
        
        # Return assembled data
        return self.get_assembled_data(transfer_id) 