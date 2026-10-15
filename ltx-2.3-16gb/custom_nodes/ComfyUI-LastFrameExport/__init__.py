"""Small ComfyUI node used by the bundled IA2V personal-LoRA preset."""


class LastFrameFromBatch:
    """Return only the final image in a ComfyUI IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_frame",)
    FUNCTION = "extract"
    CATEGORY = "image/postprocessing"

    def extract(self, images):
        if images is None or len(images) == 0:
            raise ValueError("não há frames para extrair")
        return (images[-1:].contiguous(),)


NODE_CLASS_MAPPINGS = {"LastFrameFromBatch": LastFrameFromBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"LastFrameFromBatch": "Extract Last Video Frame"}
