"""
ComfyUI Video Nodes
Custom nodes for video upload (loading) and video combine (encoding)
Supports both GPU and CPU processing
"""

import os
import folder_paths
import torch
import numpy as np

from .video_utils import (
    load_video_frames,
    encode_video,
    get_video_info,
    resize_frames_tensor,
    get_device,
    get_best_encoder,
    check_gpu_encoder
)


class VideoUpload:
    """
    Load a video file and convert it to a batch of image frames.
    Supports GPU acceleration for frame processing.
    """
    
    CATEGORY = "video"
    FUNCTION = "load_video"
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("images", "frame_count", "fps", "width", "height")
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        
        # Get video files from input directory
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif', '.m4v', '.wmv', '.flv')):
                    files.append(f)
        
        return {
            "required": {
                "video": (sorted(files) if files else [""], {"video_upload": True}),
            },
            "optional": {
                "force_rate": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 120,
                    "step": 0.1,
                    "tooltip": "Target frame rate. 0 = keep original rate"
                }),
                "force_size": (["Disabled", "256x256", "512x512", "768x768", "1024x1024", 
                               "512x?", "?x512", "768x?", "?x768", "1024x?", "?x1024"],
                              {"default": "Disabled"}),
                "frame_load_cap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Maximum frames to load. 0 = no limit"
                }),
                "skip_first_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Number of frames to skip from the start"
                }),
                "select_every_nth": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Select every nth frame"
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        """Check if the video file has changed."""
        video_path = folder_paths.get_annotated_filepath(video)
        if os.path.exists(video_path):
            return os.path.getmtime(video_path)
        return float("nan")
    
    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        """Validate that the video file exists."""
        if not video:
            return "No video file selected"
        video_path = folder_paths.get_annotated_filepath(video)
        if not os.path.exists(video_path):
            return f"Video file not found: {video}"
        return True
    
    def _parse_force_size(self, force_size, original_width, original_height):
        """Parse the force_size option and return target dimensions."""
        if force_size == "Disabled":
            return None
        
        if "x?" in force_size:
            # Width specified, calculate height from aspect ratio
            target_width = int(force_size.replace("x?", ""))
            aspect = original_height / original_width
            target_height = int(target_width * aspect)
            # Round to nearest multiple of 8 for encoding compatibility
            target_height = (target_height // 8) * 8
            return (target_width, target_height)
        
        elif "?x" in force_size:
            # Height specified, calculate width from aspect ratio
            target_height = int(force_size.replace("?x", ""))
            aspect = original_width / original_height
            target_width = int(target_height * aspect)
            target_width = (target_width // 8) * 8
            return (target_width, target_height)
        
        else:
            # Both dimensions specified
            parts = force_size.split("x")
            return (int(parts[0]), int(parts[1]))
    
    def load_video(self, video, force_rate=0, force_size="Disabled", 
                   frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
        """
        Load video and return frames as IMAGE batch.
        """
        video_path = folder_paths.get_annotated_filepath(video)
        
        # Get video info first
        info = get_video_info(video_path)
        
        # Parse force_size
        target_size = self._parse_force_size(force_size, info["width"], info["height"])
        
        # Load frames
        frames, frame_count, fps = load_video_frames(
            video_path,
            force_rate=force_rate,
            frame_load_cap=frame_load_cap,
            skip_first_frames=skip_first_frames,
            select_every_nth=select_every_nth,
            force_size=target_size
        )
        
        # Get output dimensions
        output_height = frames.shape[1]
        output_width = frames.shape[2]
        
        return (frames, frame_count, fps, output_width, output_height)


class VideoUploadPath:
    """
    Load a video from a file path.
    Similar to VideoUpload but accepts a string path instead of uploaded file.
    """
    
    CATEGORY = "video"
    FUNCTION = "load_video"
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("images", "frame_count", "fps", "width", "height")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Full path to the video file"
                }),
            },
            "optional": {
                "force_rate": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 120,
                    "step": 0.1
                }),
                "force_size": (["Disabled", "256x256", "512x512", "768x768", "1024x1024",
                               "512x?", "?x512", "768x?", "?x768", "1024x?", "?x1024"],
                              {"default": "Disabled"}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }
    
    def _parse_force_size(self, force_size, original_width, original_height):
        """Parse the force_size option and return target dimensions."""
        if force_size == "Disabled":
            return None
        
        if "x?" in force_size:
            target_width = int(force_size.replace("x?", ""))
            aspect = original_height / original_width
            target_height = int(target_width * aspect)
            target_height = (target_height // 8) * 8
            return (target_width, target_height)
        
        elif "?x" in force_size:
            target_height = int(force_size.replace("?x", ""))
            aspect = original_width / original_height
            target_width = int(target_height * aspect)
            target_width = (target_width // 8) * 8
            return (target_width, target_height)
        
        else:
            parts = force_size.split("x")
            return (int(parts[0]), int(parts[1]))
    
    def load_video(self, video_path, force_rate=0, force_size="Disabled",
                   frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
        """Load video from path and return frames."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        info = get_video_info(video_path)
        target_size = self._parse_force_size(force_size, info["width"], info["height"])
        
        frames, frame_count, fps = load_video_frames(
            video_path,
            force_rate=force_rate,
            frame_load_cap=frame_load_cap,
            skip_first_frames=skip_first_frames,
            select_every_nth=select_every_nth,
            force_size=target_size
        )
        
        output_height = frames.shape[1]
        output_width = frames.shape[2]
        
        return (frames, frame_count, fps, output_width, output_height)


class VideoCombine:
    """
    Combine image frames into a video file.
    Supports GPU-accelerated encoding via NVENC/AMF when available.
    """
    
    CATEGORY = "video"
    FUNCTION = "combine_video"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {
                    "default": 24,
                    "min": 1,
                    "max": 120,
                    "step": 0.1,
                    "tooltip": "Output video frame rate"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI_video",
                    "tooltip": "Prefix for output filename"
                }),
            },
            "optional": {
                "format": (["mp4_h264", "mp4_hevc", "webm_vp9", "gif"], {
                    "default": "mp4_h264",
                    "tooltip": "Output video format and codec"
                }),
                "quality": ("INT", {
                    "default": 23,
                    "min": 0,
                    "max": 51,
                    "step": 1,
                    "tooltip": "CRF quality (lower = better quality, larger file)"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU encoding if available (NVENC/AMF)"
                }),
                "loop_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of times to loop the video (0 = no loop)"
                }),
                "pingpong": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Play forward then backward for seamless loop"
                }),
                "pix_fmt": (["yuv420p", "yuv444p", "yuv420p10le"], {
                    "default": "yuv420p",
                    "tooltip": "Pixel format (yuv420p is most compatible)"
                }),
            }
        }
    
    def combine_video(self, images, frame_rate, filename_prefix, format="mp4_h264",
                      quality=23, use_gpu=True, loop_count=0, pingpong=False, pix_fmt="yuv420p"):
        """
        Combine images into a video file.
        """
        # Parse format
        format_parts = format.split("_")
        container = format_parts[0]
        codec = format_parts[1] if len(format_parts) > 1 else "h264"
        
        # Determine extension
        ext_map = {"mp4": ".mp4", "webm": ".webm", "gif": ".gif"}
        extension = ext_map.get(container, ".mp4")
        
        # Get output directory
        output_dir = folder_paths.get_output_directory()
        
        # Generate unique filename
        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:05d}{extension}"
            output_path = os.path.join(output_dir, filename)
            if not os.path.exists(output_path):
                break
            counter += 1
        
        # Apply pingpong if enabled
        frames = images
        if pingpong and frames.shape[0] > 1:
            # Reverse frames (excluding first and last to avoid duplicates)
            reversed_frames = torch.flip(frames[1:-1], dims=[0])
            frames = torch.cat([frames, reversed_frames], dim=0)
        
        # Apply loop
        if loop_count > 0:
            frames = frames.repeat(loop_count + 1, 1, 1, 1)
        
        # Encode video
        output_file = encode_video(
            frames,
            output_path,
            frame_rate=frame_rate,
            codec=codec,
            quality=quality,
            use_gpu=use_gpu,
            pix_fmt=pix_fmt
        )
        
        # Return result with preview
        return {"ui": {"videos": [{"filename": filename, "type": "output"}]}, 
                "result": (output_file,)}


class VideoInfo:
    """
    Get information about a video file.
    """
    
    CATEGORY = "video"
    FUNCTION = "get_info"
    RETURN_TYPES = ("FLOAT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("fps", "frame_count", "width", "height", "duration")
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif', '.m4v')):
                    files.append(f)
        
        return {
            "required": {
                "video": (sorted(files) if files else [""],),
            }
        }
    
    def get_info(self, video):
        """Get video information."""
        video_path = folder_paths.get_annotated_filepath(video)
        info = get_video_info(video_path)
        
        return (info["fps"], info["frame_count"], info["width"], 
                info["height"], info["duration"])


class GPUEncoderInfo:
    """
    Display available GPU encoders.
    Useful for debugging GPU encoding support.
    """
    
    CATEGORY = "video"
    FUNCTION = "get_encoder_info"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }
    
    def get_encoder_info(self):
        """Get information about available encoders."""
        device = get_device()
        encoders = check_gpu_encoder()
        
        lines = [
            f"PyTorch Device: {device}",
            f"CUDA Available: {torch.cuda.is_available()}",
            "",
            "Available GPU Encoders:",
        ]
        
        for encoder, available in encoders.items():
            status = "✓" if available else "✗"
            lines.append(f"  {status} {encoder}")
        
        lines.append("")
        lines.append(f"Recommended H.264 Encoder: {get_best_encoder('h264', True)}")
        lines.append(f"Recommended HEVC Encoder: {get_best_encoder('hevc', True)}")
        
        info_text = "\n".join(lines)
        print(info_text)
        
        return (info_text,)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "VideoUpload": VideoUpload,
    "VideoUploadPath": VideoUploadPath,
    "VideoCombine": VideoCombine,
    "VideoInfo": VideoInfo,
    "GPUEncoderInfo": GPUEncoderInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoUpload": "Video Upload",
    "VideoUploadPath": "Video Upload (Path)",
    "VideoCombine": "Video Combine",
    "VideoInfo": "Video Info",
    "GPUEncoderInfo": "GPU Encoder Info",
}
