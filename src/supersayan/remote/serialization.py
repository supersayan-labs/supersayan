"""
Serialization utilities for SuperSayan remote execution.

This module provides a clean API for serializing and deserializing
objects that implement the Serializable protocol.
"""

import base64
import json
import numpy as np
from typing import Any, Dict, List, Union, TypeVar, Type, Optional

from supersayan.core.types import Serializable, SerializableArray, LWE

T = TypeVar('T', bound=Serializable)

def serialize_data(data: Any) -> str:
    """
    Serialize data to a Base64-encoded JSON string.
    
    Args:
        data: The data to serialize (must be Serializable or convertible to Serializable)
        
    Returns:
        Base64-encoded JSON string representation of the data
    """
    # Convert data to a serializable format
    serializable = convert_to_serializable(data)
    
    # Convert to JSON and then Base64 encode
    json_str = json.dumps(serializable)
    return base64.b64encode(json_str.encode('utf-8')).decode('utf-8')

def deserialize_data(data_base64: str) -> Any:
    """
    Deserialize data from a Base64-encoded JSON string.
    
    Args:
        data_base64: Base64-encoded JSON string
        
    Returns:
        Deserialized object - primarily focused on returning numpy arrays
    """
    # Decode Base64 and parse JSON
    json_str = base64.b64decode(data_base64).decode('utf-8')
    serializable = json.loads(json_str)
    
    # Convert from serializable format to objects
    result = convert_from_serializable(serializable)
    
    # If the deserialized result is a SerializableArray, return the underlying numpy array
    if isinstance(result, SerializableArray):
        return result.array
        
    return result

def convert_to_serializable(data: Any) -> Dict[str, Any]:
    """
    Convert any data to a serializable format.
    
    This function handles the following types:
    - Serializable objects (including LWE)
    - numpy arrays (converting them to SerializableArray)
    - Julia objects with mask and masked attributes (LWE)
    - Dictionaries and lists (recursively converting values)
    - Primitive types (pass-through)
    
    Args:
        data: The data to convert
        
    Returns:
        JSON-serializable representation of the data
    """
    # Handle None
    if data is None:
        return {"type": "null", "data": None}
        
    # Handle Serializable objects directly
    if isinstance(data, Serializable):
        return data.to_dict()
    
    # Handle Julia objects with mask and masked attributes (LWE objects)
    if hasattr(data, "mask") and hasattr(data, "masked"):
        # Convert Julia LWE to Python LWE, then to dict
        return LWE(data.mask, data.masked).to_dict()
    
    # Handle numpy arrays by wrapping them in SerializableArray
    if isinstance(data, np.ndarray):
        # If the array contains Julia objects, convert them to Python LWE objects first
        if data.dtype == object:
            # Try to convert array of Julia LWE objects to Python LWE objects
            try:
                converted_array = np.empty(data.shape, dtype=object)
                for index, item in np.ndenumerate(data):
                    if hasattr(item, "mask") and hasattr(item, "masked"):
                        # Convert Julia LWE to Python LWE
                        converted_array[index] = LWE(item.mask, item.masked)
                    else:
                        converted_array[index] = item
                return SerializableArray(converted_array).to_dict()
            except Exception as e:
                # If conversion fails, try to serialize as is
                pass
        
        return SerializableArray(data).to_dict()
    
    # Handle Julia wrapper objects (jlwrap)
    if hasattr(data, "__class__") and data.__class__.__name__ == "jlwrap":
        # Try to handle different types of Julia wrapper objects
        try:
            # If it looks like an LWE object
            if hasattr(data, "mask") and hasattr(data, "masked"):
                return LWE(data.mask, data.masked).to_dict()
            # If it's an array-like object
            elif hasattr(data, "__len__") and hasattr(data, "__getitem__"):
                # Try to convert to a numpy array if possible
                arr = np.array(data)
                return SerializableArray(arr).to_dict()
            else:
                # Last resort - convert to string
                return {"type": "string", "data": str(data)}
        except Exception as e:
            # If all else fails, convert to string
            return {"type": "string", "data": str(data)}
    
    # Handle dictionaries
    if isinstance(data, dict):
        # If it's already a serialized object, pass it through
        if "type" in data and isinstance(data["type"], str):
            return data
        # Otherwise serialize each value
        return {k: convert_to_serializable(v) for k, v in data.items()}
    
    # Handle lists
    if isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    
    # Handle primitive types directly
    if isinstance(data, (str, int, float, bool)):
        return data
    
    # For anything else, convert to string
    return {"type": "string", "data": str(data)}

def convert_from_serializable(data: Any) -> Any:
    """
    Convert from a serializable format back to Python objects.
    
    Args:
        data: The serialized data
        
    Returns:
        Deserialized Python objects - primarily focused on numpy arrays
    """
    # Handle primitive types directly
    if not isinstance(data, dict) or "type" not in data:
        # Pass through primitive types and non-typed dictionaries
        if isinstance(data, dict):
            return {k: convert_from_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_from_serializable(item) for item in data]
        else:
            return data
    
    # Handle typed dictionaries
    type_name = data["type"]
    
    # Handle LWE objects
    if type_name == "LWE":
        return LWE.from_dict(data)
    
    # Handle arrays (both regular and of objects)
    if type_name in ["array", "array_of_objects"]:
        return SerializableArray.from_dict(data)
    
    # Handle null values
    if type_name == "null":
        return None
    
    # Handle string representations
    if type_name == "string":
        return data["data"]
    
    # Unknown type - return as is
    return data