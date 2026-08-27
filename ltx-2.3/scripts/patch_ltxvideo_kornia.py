#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
PYRAMID_BLENDING = (
    COMFYUI_HOME / "custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py"
)
SHIM = (
    "# Kornia >=0.8.3 no longer re-exports pad from "
    "kornia.geometry.transform.pyramid.\n"
    "pad = F.pad\n"
)


def patch_ltxvideo_kornia(path: Path = PYRAMID_BLENDING) -> bool:
    if not path.is_file():
        raise FileNotFoundError(f"ComfyUI-LTXVideo ausente: {path}")

    text = path.read_text(encoding="utf-8")
    patched = text.replace("    pad,\n", "")
    if "pad = F.pad" not in patched:
        patched = patched.replace(
            "import torch.nn.functional as F\n",
            f"import torch.nn.functional as F\n\n{SHIM}",
            1,
        )
    if patched == text:
        return False
    path.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    changed = patch_ltxvideo_kornia()
    action = "corrigido" if changed else "ja corrigido"
    print(f"ComfyUI-LTXVideo Kornia shim {action}: {PYRAMID_BLENDING}")


if __name__ == "__main__":
    main()
