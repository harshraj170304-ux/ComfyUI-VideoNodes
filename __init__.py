"""
ComfyUI-VideoNodes
Custom nodes for video upload and combine operations in ComfyUI.
Supports both GPU and CPU processing.
Based on VideoHelperSuite architecture with GPU encoder support.
"""

from .video_nodes import (
    VideoUpload,
    VideoUploadPath,
    VideoCombine,
    VideoInfo,
    GPUEncoderInfo,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

# Version info
__version__ = "2.0.0"
__author__ = "ComfyUI-VideoNodes"

# Web directory for custom JavaScript (if needed in future)
WEB_DIRECTORY = "./web"
