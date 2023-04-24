from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {"configuration_lamda": ["LaMDAConfig",]}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_lamda"] = ["LaMDATokenizer"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_lamda"] = [
        "LaMDAForCausalLM",
        "LaMDAModel",
        "LaMDAPreTrainedModel",
        "load_tf_weights_in_lamda",
    ]

if TYPE_CHECKING:
    from .configuration_lamda import LaMDAConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_lamda import LaMDATokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_lamda import (
            LaMDAForCausalLM,
            LaMDAModel,
            LaMDAPreTrainedModel,
            load_tf_weights_in_lamda,
        )
        
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
