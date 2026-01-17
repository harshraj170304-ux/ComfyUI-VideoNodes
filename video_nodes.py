"""
ComfyUI Video Nodes - Enhanced Version
Custom nodes for video upload (loading) and video combine (encoding)
Supports both GPU and CPU processing with direct ffmpeg integration
Based on VideoHelperSuite architecture with GPU encoding support
"""

import os
import sys
import json
import subprocess
import numpy as np
import re
import datetime
from typing import List, Optional, Tuple, Iterator, Generator
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pathlib import Path

import folder_paths
from comfy.utils import ProgressBar

from .video_utils import (
    get_device,
    check_gpu_encoder,
    get_best_encoder,
    CV2_AVAILABLE,
)

# Try to import cv2
try:
    import cv2
except ImportError:
    cv2 = None

# Constants
BIGMAX = 2**53 - 1
ENCODE_ARGS = ("utf-8", "backslashreplace")

def get_ffmpeg_path():
    """Get ffmpeg executable path."""
    # Try imageio-ffmpeg first
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    
    # Try system ffmpeg
    for name in ["ffmpeg", "ffmpeg.exe"]:
        try:
            result = subprocess.run([name, "-version"], capture_output=True)
            if result.returncode == 0:
                return name
        except FileNotFoundError:
            continue
    
    return None

ffmpeg_path = get_ffmpeg_path()


def tensor_to_bytes(tensor):
    """Convert tensor to uint8 numpy array."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    return (tensor * 255 + 0.5).clip(0, 255).astype(np.uint8)


def get_video_formats():
    """Get available video formats with GPU options."""
    formats = [
        "video/h264-mp4",
        "video/h264-mp4-gpu",
        "video/hevc-mp4",
        "video/hevc-mp4-gpu",
        "video/vp9-webm",
        "video/gif",
        "image/gif",
        "image/webp",
    ]
    return formats


def get_format_config(format_name: str, use_gpu: bool = True) -> dict:
    """Get ffmpeg configuration for a format."""
    configs = {
        "h264-mp4": {
            "extension": "mp4",
            "codec": "libx264",
            "codec_gpu": "h264_nvenc",
            "pix_fmt": "yuv420p",
            "extra_args": ["-preset", "medium"],
            "extra_args_gpu": ["-preset", "p4", "-tune", "hq"],
        },
        "h264-mp4-gpu": {
            "extension": "mp4",
            "codec": "h264_nvenc",
            "codec_gpu": "h264_nvenc",
            "pix_fmt": "yuv420p",
            "extra_args": ["-preset", "p4", "-tune", "hq"],
            "extra_args_gpu": ["-preset", "p4", "-tune", "hq"],
        },
        "hevc-mp4": {
            "extension": "mp4",
            "codec": "libx265",
            "codec_gpu": "hevc_nvenc",
            "pix_fmt": "yuv420p",
            "extra_args": ["-preset", "medium"],
            "extra_args_gpu": ["-preset", "p4", "-tune", "hq"],
        },
        "hevc-mp4-gpu": {
            "extension": "mp4",
            "codec": "hevc_nvenc",
            "codec_gpu": "hevc_nvenc",
            "pix_fmt": "yuv420p",
            "extra_args": ["-preset", "p4", "-tune", "hq"],
            "extra_args_gpu": ["-preset", "p4", "-tune", "hq"],
        },
        "vp9-webm": {
            "extension": "webm",
            "codec": "libvpx-vp9",
            "codec_gpu": "libvpx-vp9",  # No GPU encoder for VP9
            "pix_fmt": "yuv420p",
            "extra_args": ["-deadline", "good", "-cpu-used", "2"],
            "extra_args_gpu": ["-deadline", "good", "-cpu-used", "2"],
        },
        "gif": {
            "extension": "gif",
            "codec": "gif",
            "codec_gpu": "gif",
            "pix_fmt": "rgb8",
            "extra_args": ["-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"],
            "extra_args_gpu": ["-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"],
        },
    }
    
    format_key = format_name.replace("video/", "").replace("image/", "")
    config = configs.get(format_key, configs["h264-mp4"])
    
    # Check if GPU encoder is available
    if use_gpu and "gpu" in format_key:
        encoders = check_gpu_encoder()
        if config["codec_gpu"] in encoders and encoders[config["codec_gpu"]]:
            config["use_gpu"] = True
        else:
            config["use_gpu"] = False
            print(f"[VideoNodes] GPU encoder {config['codec_gpu']} not available, falling back to CPU")
    else:
        config["use_gpu"] = False
    
    return config


def ffmpeg_process(args: list, file_path: str, env: dict) -> Generator:
    """Generator that pipes frames to ffmpeg."""
    frame_data = yield
    total_frames = 0
    
    with subprocess.Popen(
        args + [file_path],
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        env=env
    ) as proc:
        try:
            while frame_data is not None:
                proc.stdin.write(frame_data)
                frame_data = yield
                total_frames += 1
            proc.stdin.flush()
            proc.stdin.close()
            res = proc.stderr.read()
        except BrokenPipeError:
            res = proc.stderr.read()
            raise Exception(f"FFmpeg error: {res.decode(*ENCODE_ARGS)}")
    
    yield total_frames
    if len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)


def to_pingpong(images):
    """Create pingpong loop from images."""
    if not hasattr(images, "__getitem__"):
        images = list(images)
    yield from images
    for i in range(len(images) - 2, 0, -1):
        yield images[i]


class VideoUpload:
    """
    Load a video file and convert it to a batch of image frames.
    Supports GPU acceleration for frame processing.
    """
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "load_video"
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT", "AUDIO")
    RETURN_NAMES = ("images", "frame_count", "fps", "width", "height", "audio")
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        
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
                    "max": BIGMAX,
                    "step": 1,
                    "tooltip": "Maximum frames to load. 0 = no limit"
                }),
                "skip_first_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": BIGMAX,
                    "step": 1,
                }),
                "select_every_nth": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        video_path = folder_paths.get_annotated_filepath(video)
        if os.path.exists(video_path):
            return os.path.getmtime(video_path)
        return float("nan")
    
    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not video:
            return "No video file selected"
        video_path = folder_paths.get_annotated_filepath(video)
        if not os.path.exists(video_path):
            return f"Video file not found: {video}"
        return True
    
    def _parse_force_size(self, force_size, original_width, original_height):
        if force_size == "Disabled":
            return None
        
        if "x?" in force_size:
            target_width = int(force_size.replace("x?", ""))
            aspect = original_height / original_width
            target_height = int(target_width * aspect)
            target_height = (target_height // 2) * 2  # Ensure even
            return (target_width, target_height)
        elif "?x" in force_size:
            target_height = int(force_size.replace("?x", ""))
            aspect = original_width / original_height
            target_width = int(target_height * aspect)
            target_width = (target_width // 2) * 2
            return (target_width, target_height)
        else:
            parts = force_size.split("x")
            return (int(parts[0]), int(parts[1]))
    
    def load_video(self, video, force_rate=0, force_size="Disabled", 
                   frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python")
        
        video_path = folder_paths.get_annotated_filepath(video)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        target_size = self._parse_force_size(force_size, width, height)
        
        # Calculate frame skip for rate conversion
        rate_skip = 1
        if force_rate > 0 and original_fps > 0:
            rate_skip = max(1, int(round(original_fps / force_rate)))
        
        frames = []
        frame_idx = 0
        loaded_count = 0
        
        if skip_first_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
            frame_idx = skip_first_frames
        
        pbar = ProgressBar(min(frame_load_cap if frame_load_cap > 0 else total_frames, total_frames))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_idx - skip_first_frames) % (rate_skip * select_every_nth) == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if target_size is not None:
                    frame_rgb = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LANCZOS4)
                
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
                frames.append(frame_normalized)
                loaded_count += 1
                pbar.update(1)
                
                if frame_load_cap > 0 and loaded_count >= frame_load_cap:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames were loaded from the video")
        
        frames_array = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_array)
        
        output_height = frames_tensor.shape[1]
        output_width = frames_tensor.shape[2]
        
        # Audio placeholder (would need ffmpeg to extract properly)
        audio = None
        
        return (frames_tensor, len(frames), original_fps, output_width, output_height, audio)


class VideoUploadPath:
    """Load a video from a file path."""
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "load_video"
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT", "AUDIO")
    RETURN_NAMES = ("images", "frame_count", "fps", "width", "height", "audio")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 120, "step": 0.1}),
                "force_size": (["Disabled", "256x256", "512x512", "768x768", "1024x1024",
                               "512x?", "?x512", "768x?", "?x768", "1024x?", "?x1024"],
                              {"default": "Disabled"}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }
    
    def load_video(self, video_path, force_rate=0, force_size="Disabled",
                   frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
        # Reuse VideoUpload logic
        uploader = VideoUpload()
        # Temporarily override the path resolution
        original_get_annotated = folder_paths.get_annotated_filepath
        folder_paths.get_annotated_filepath = lambda x: video_path
        try:
            result = uploader.load_video("", force_rate, force_size, frame_load_cap, 
                                         skip_first_frames, select_every_nth)
        finally:
            folder_paths.get_annotated_filepath = original_get_annotated
        return result


class VideoCombine:
    """
    Combine image frames into a video file.
    Supports GPU-accelerated encoding via NVENC when available.
    Similar to VideoHelperSuite VHS_VideoCombine with GPU support.
    """
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {
                    "default": 8,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                }),
                "loop_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "format": (get_video_formats(),),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "crf": ("INT", {
                    "default": 19,
                    "min": 0,
                    "max": 51,
                    "step": 1,
                    "tooltip": "Quality (lower = better, 0 = lossless)"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    def combine_video(
        self,
        images,
        frame_rate: float,
        loop_count: int,
        filename_prefix: str = "ComfyUI",
        format: str = "video/h264-mp4",
        pingpong: bool = False,
        save_output: bool = True,
        audio=None,
        crf: int = 19,
        prompt=None,
        extra_pnginfo=None,
    ):
        if ffmpeg_path is None:
            raise ProcessLookupError(
                "ffmpeg is required for video outputs.\n"
                "Install with: pip install imageio-ffmpeg\n"
                "Or install ffmpeg and add to system PATH."
            )
        
        if images is None or (isinstance(images, torch.Tensor) and images.size(0) == 0):
            return ((save_output, []),)
        
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
        
        # Get first image for dimensions
        first_image = images[0]
        dimensions = (first_image.shape[1], first_image.shape[0])  # width, height
        
        # Ensure dimensions are even (required by most codecs)
        if dimensions[0] % 2 or dimensions[1] % 2:
            pad_w = dimensions[0] % 2
            pad_h = dimensions[1] % 2
            dimensions = (dimensions[0] + pad_w, dimensions[1] + pad_h)
        
        # Get output directory
        output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )
        output_files = []
        
        # Counter for unique filenames
        max_counter = 0
        matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                file_counter = int(match.group(1))
                if file_counter > max_counter:
                    max_counter = file_counter
        counter = max_counter + 1
        
        # Save first frame as PNG with metadata
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        
        first_image_file = f"{filename}_{counter:05}.png"
        png_path = os.path.join(full_output_folder, first_image_file)
        Image.fromarray(tensor_to_bytes(first_image)).save(png_path, pnginfo=metadata, compress_level=4)
        output_files.append(png_path)
        
        # Parse format
        format_type, format_ext = format.split("/")
        
        # Handle image formats (gif, webp) with Pillow
        if format_type == "image":
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            
            if pingpong:
                images = list(to_pingpong(images))
            
            def frames_gen(imgs):
                for img in imgs:
                    pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(img))
            
            frames = frames_gen(images)
            first_pil = next(frames)
            
            save_kwargs = {"duration": round(1000 / frame_rate), "loop": loop_count}
            if format_ext == "gif":
                save_kwargs["disposal"] = 2
            elif format_ext == "webp":
                save_kwargs["lossless"] = True
            
            first_pil.save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                **save_kwargs
            )
            output_files.append(file_path)
        
        else:
            # Use ffmpeg for video formats
            use_gpu = "gpu" in format_ext.lower()
            config = get_format_config(format, use_gpu)
            
            codec = config["codec_gpu"] if config.get("use_gpu") else config["codec"]
            extra_args = config["extra_args_gpu"] if config.get("use_gpu") else config["extra_args"]
            
            file = f"{filename}_{counter:05}.{config['extension']}"
            file_path = os.path.join(full_output_folder, file)
            
            # Build ffmpeg command
            args = [
                ffmpeg_path, "-y", "-v", "error",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{dimensions[0]}x{dimensions[1]}",
                "-r", str(frame_rate),
                "-i", "-",
            ]
            
            # Add loop filter if needed
            if loop_count > 0:
                args += ["-vf", f"loop=loop={loop_count}:size={num_frames}"]
            
            # Add codec and quality settings
            args += ["-c:v", codec]
            
            # Add CRF/quality
            if "nvenc" in codec:
                args += ["-cq", str(crf), "-rc", "vbr"]
            else:
                args += ["-crf", str(crf)]
            
            args += extra_args
            args += ["-pix_fmt", config["pix_fmt"]]
            
            # Prepare images
            if pingpong:
                images = list(to_pingpong(images))
            
            env = os.environ.copy()
            
            # Run ffmpeg
            with subprocess.Popen(
                args + [file_path],
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                env=env
            ) as proc:
                try:
                    for img in images:
                        frame_bytes = tensor_to_bytes(img).tobytes()
                        proc.stdin.write(frame_bytes)
                        pbar.update(1)
                    proc.stdin.flush()
                    proc.stdin.close()
                    stderr = proc.stderr.read()
                    if proc.wait() != 0:
                        raise Exception(f"FFmpeg error: {stderr.decode(*ENCODE_ARGS)}")
                except BrokenPipeError:
                    stderr = proc.stderr.read()
                    raise Exception(f"FFmpeg pipe error: {stderr.decode(*ENCODE_ARGS)}")
            
            output_files.append(file_path)
            
            # Handle audio muxing if provided
            if audio is not None:
                try:
                    waveform = audio.get('waveform')
                    sample_rate = audio.get('sample_rate', 44100)
                    if waveform is not None:
                        # Save audio and mux with video
                        audio_file = os.path.join(full_output_folder, f"{filename}_{counter:05}_audio.wav")
                        output_with_audio = os.path.join(full_output_folder, f"{filename}_{counter:05}_final.{config['extension']}")
                        
                        # This is a simplified audio handling - full implementation would need proper audio encoding
                        pass
                except Exception as e:
                    print(f"[VideoNodes] Audio muxing failed: {e}")
        
        return {"ui": {"gifs": [{"filename": os.path.basename(f), "subfolder": subfolder, "type": "output"} for f in output_files[1:]]},
                "result": ((save_output, output_files),)}


class VideoInfo:
    """Get information about a video file."""
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
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
        if cv2 is None:
            raise ImportError("OpenCV is required")
        
        video_path = folder_paths.get_annotated_filepath(video)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return (fps, frame_count, width, height, duration)


class GPUEncoderInfo:
    """Display available GPU encoders."""
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "get_encoder_info"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    
    def get_encoder_info(self):
        device = get_device()
        encoders = check_gpu_encoder()
        
        lines = [
            f"PyTorch Device: {device}",
            f"CUDA Available: {torch.cuda.is_available()}",
            f"FFmpeg Path: {ffmpeg_path}",
            "",
            "GPU Encoders:",
        ]
        
        for encoder, available in encoders.items():
            status = "✓" if available else "✗"
            lines.append(f"  {status} {encoder}")
        
        lines.append("")
        lines.append(f"Recommended H.264: {get_best_encoder('h264', True)}")
        lines.append(f"Recommended HEVC: {get_best_encoder('hevc', True)}")
        
        info_text = "\n".join(lines)
        print(info_text)
        
        return (info_text,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "VHS_VideoUpload": VideoUpload,
    "VHS_VideoUploadPath": VideoUploadPath,
    "VHS_VideoCombine_GPU": VideoCombine,
    "VHS_VideoInfo": VideoInfo,
    "VHS_GPUEncoderInfo": GPUEncoderInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_VideoUpload": "Video Upload 🎥",
    "VHS_VideoUploadPath": "Video Upload (Path) 🎥",
    "VHS_VideoCombine_GPU": "Video Combine (GPU) 🎥",
    "VHS_VideoInfo": "Video Info 🎥",
    "VHS_GPUEncoderInfo": "GPU Encoder Info 🎥",
}
