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
