import os

if os.environ.get("FLASH_ATTN_TF_USE_AS_DEPENDENCY") == "1":
    from flash_attn_tf import flash_mha
    __all__ = ["flash_mha"]

else:
    from flash_attn_tf import (
        flash_mha,
    )
    from flash_attn_tf._version import __version__

    __all__ = [
        "flash_mha",
        "__version__",
    ]
