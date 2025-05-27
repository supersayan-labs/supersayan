try:
    from ._setup import _auto_setup

    _auto_setup()
except Exception:
    pass

from .core import encryption, keygen, bindings
from .nn import layers
from .nn.convert import (
    SupersayanModel,
    ModelType,
)
from .nn.layers import Linear, Conv2d
from .remote import SupersayanClient, SupersayanServer, socket_utils
from .logging_config import configure_logging, get_logger, set_log_level
