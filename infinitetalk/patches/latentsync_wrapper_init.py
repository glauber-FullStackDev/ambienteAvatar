from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .latentsync_stable_node import LatentSyncStableNode


NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS,
    "LatentSyncStableNode": LatentSyncStableNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS,
    "LatentSyncStableNode": "LatentSync 1.6 Stable (InfiniteTalk)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
