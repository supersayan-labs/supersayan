import numpy as np
from typing import (
    Dict,
    Any,
    Protocol,
    Union,
    List,
    Type,
    TypeVar,
    Generic,
    Tuple,
    runtime_checkable,
)
import sys


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dictionaries."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a serializable dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Create an object from a dictionary representation."""
        ...


T = TypeVar("T")


class SerializableArray(Serializable, Generic[T]):
    """
    A wrapper for numpy arrays that makes them serializable.

    This class handles the serialization of numpy arrays, including
    arrays of Serializable objects like LWE.
    """

    def __init__(self, array: np.ndarray, element_type: Type[T] = None):
        """
        Initialize a SerializableArray.

        Args:
            array: The numpy array to wrap
            element_type: Optional type of the elements in the array
        """
        self.array = array
        self.element_type = element_type

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the array to a serializable dictionary.

        Returns:
            dict: A dictionary representation of the array
        """
        # Handle array of Serializable objects
        if self.array.dtype == object and isinstance(
            self.array.flatten()[0], Serializable
        ):
            # Convert each Serializable object to a dict
            items = [item.to_dict() for item in self.array.flatten()]
            return {
                "data": items,
                "shape": self.array.shape,
                "type": "array_of_objects",
            }
        # Handle regular numpy arrays
        else:
            return {
                "data": self.array.tolist(),
                "shape": self.array.shape,
                "type": "array",
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SerializableArray":
        """
        Create a SerializableArray from a dictionary representation.

        Args:
            data: Dictionary representation of the array

        Returns:
            SerializableArray: A SerializableArray containing the deserialized data
        """
        if data.get("type") not in ["array", "array_of_objects"]:
            raise ValueError(
                f"Expected array or array_of_objects type, got {data.get('type')}"
            )

        shape = tuple(data.get("shape", (len(data["data"]),)))

        if data.get("type") == "array_of_objects":
            # If we're deserializing objects, we need to know their type
            # First element should indicate the type through its "type" field
            if not data["data"]:  # Empty array
                return cls(np.array([], dtype=object).reshape(shape))

            first_item = data["data"][0]
            item_type = first_item.get("type")

            if item_type == "LWE":
                # Convert each dict to an LWE object
                from supersayan.core.types import (
                    LWE,
                )  # Local import to avoid circular reference

                items = [LWE.from_dict(item) for item in data["data"]]
                return cls(np.array(items, dtype=object).reshape(shape), LWE)
            else:
                # Unknown object type, just use the data as is
                return cls(np.array(data["data"], dtype=object).reshape(shape))
        else:
            # Regular array
            return cls(np.array(data["data"]).reshape(shape))

    def __array__(self) -> np.ndarray:
        """
        Return the underlying numpy array.
        This allows the SerializableArray to be used as a numpy array.
        """
        return self.array

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying numpy array.
        """
        if name.startswith("__") and name.endswith("__"):
            return super().__getattr__(name)
        return getattr(self.array, name)

    def __getitem__(self, idx):
        """
        Support numpy-style indexing.
        """
        return self.array[idx]

    def __setitem__(self, idx, value):
        """
        Support numpy-style item assignment.
        """
        self.array[idx] = value

    def __repr__(self):
        return f"SerializableArray(shape={self.array.shape}, element_type={self.element_type})"


class LWE(Serializable):
    """
    Python representation of an LWE ciphertext.

    An LWE ciphertext consists of a mask (vector of Float64) and a masked value (Float64).
    """

    def __init__(self, mask: Union[List[float], np.ndarray], masked: float):
        """
        Initialize an LWE ciphertext.

        Args:
            mask: The vector of floats representing the mask
            masked: The masked value (float)
        """
        self.mask = (
            mask if isinstance(mask, np.ndarray) else np.array(mask, dtype=np.float64)
        )
        self.masked = float(masked)

    @classmethod
    def from_julia(cls, julia_lwe):
        """
        Create an LWE instance from a Julia LWE object.

        Args:
            julia_lwe: The Julia LWE object

        Returns:
            LWE: A Python LWE instance
        """
        return cls(
            mask=np.array(julia_lwe.mask, dtype=np.float64),
            masked=float(julia_lwe.masked),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the LWE object to a dictionary for serialization.

        Returns:
            dict: A dictionary representation of the LWE object
        """
        return {"mask": self.mask.tolist(), "masked": self.masked, "type": "LWE"}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LWE":
        """
        Create an LWE instance from a dictionary.

        Args:
            data: Dictionary with mask and masked values

        Returns:
            LWE: A Python LWE instance
        """
        if data.get("type") != "LWE":
            raise ValueError(f"Expected LWE type, got {data.get('type')}")

        return cls(mask=data["mask"], masked=data["masked"])

    def __repr__(self):
        mask_size = sys.getsizeof(self.mask)
        masked_size = sys.getsizeof(self.masked)
        return f"LWE(mask_size={len(self.mask)} ({mask_size:,} bytes), masked={self.masked} ({masked_size:,} bytes))"


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