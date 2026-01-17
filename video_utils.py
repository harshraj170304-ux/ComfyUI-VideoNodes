"""
Video utility functions for ComfyUI-VideoNodes
Provides GPU/CPU compatible video processing functions
"""

import os
import subprocess
import numpy as np
import torch

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import imageio
    import imageio_ffmpeg
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


def get_device():
    """
    Detect and return the available device (GPU or CPU).
    
    Returns:
        torch.device: The device to use for tensor operations
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def check_gpu_encoder():
    """
    Check for available GPU video encoders.
    
    Returns:
        dict: Available encoders with their names
    """
    encoders = {
        "h264_nvenc": False,  # NVIDIA
        "hevc_nvenc": False,  # NVIDIA
        "h264_amf": False,    # AMD
        "hevc_amf": False,    # AMD
        "h264_qsv": False,    # Intel QuickSync
    }
    
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe() if IMAGEIO_AVAILABLE else "ffmpeg"
        result = subprocess.run(
            [ffmpeg_path, "-encoders"],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout + result.stderr
        
        for encoder in encoders.keys():
            if encoder in output:
                encoders[encoder] = True
                
    except Exception:
        pass
    
    return encoders


def get_best_encoder(preferred_codec="h264", use_gpu=True):
    """
    Get the best available encoder for the given codec.
    
    Args:
        preferred_codec: The codec type (h264, hevc)
        use_gpu: Whether to prefer GPU encoding
        
    Returns:
        str: The encoder name to use
    """
    if not use_gpu:
        return "libx264" if preferred_codec == "h264" else "libx265"
    
    encoders = check_gpu_encoder()
    
    if preferred_codec == "h264":
        if encoders["h264_nvenc"]:
            return "h264_nvenc"
        elif encoders["h264_amf"]:
            return "h264_amf"
        elif encoders["h264_qsv"]:
            return "h264_qsv"
        else:
            return "libx264"
    elif preferred_codec == "hevc":
        if encoders["hevc_nvenc"]:
            return "hevc_nvenc"
        elif encoders["hevc_amf"]:
            return "hevc_amf"
        else:
            return "libx265"
    
    return "libx264"


def load_video_frames(video_path, force_rate=0, frame_load_cap=0, skip_first_frames=0, 
                      select_every_nth=1, force_size=None):
    """
    Load frames from a video file.
    
    Args:
        video_path: Path to the video file
        force_rate: Target frame rate (0 = keep original)
        frame_load_cap: Maximum frames to load (0 = no limit)
        skip_first_frames: Number of frames to skip from start
        select_every_nth: Select every nth frame
        force_size: Tuple (width, height) or None
        
    Returns:
        tuple: (frames_tensor, frame_count, original_fps)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for video loading. Install with: pip install opencv-python")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame skip for rate conversion
    rate_skip = 1
    if force_rate > 0 and original_fps > 0:
        rate_skip = max(1, int(round(original_fps / force_rate)))
    
    frames = []
    frame_idx = 0
    loaded_count = 0
    
    # Skip initial frames
    if skip_first_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
        frame_idx = skip_first_frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rate and nth selection
        if (frame_idx - skip_first_frames) % (rate_skip * select_every_nth) == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if force_size is not None:
                frame_rgb = cv2.resize(frame_rgb, force_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize to 0-1 range
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frames.append(frame_normalized)
            loaded_count += 1
            
            # Check frame cap
            if frame_load_cap > 0 and loaded_count >= frame_load_cap:
                break
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames were loaded from the video")
    
    # Convert to tensor [B, H, W, C]
    frames_array = np.stack(frames, axis=0)
    frames_tensor = torch.from_numpy(frames_array)
    
    # Move to GPU if available
    device = get_device()
    frames_tensor = frames_tensor.to(device)
    
    return frames_tensor, len(frames), original_fps


def resize_frames_tensor(frames, target_size, mode="bilinear"):
    """
    Resize a batch of frames using PyTorch (GPU/CPU compatible).
    
    Args:
        frames: Tensor of shape [B, H, W, C]
        target_size: Tuple (height, width)
        mode: Interpolation mode
        
    Returns:
        Resized tensor [B, H, W, C]
    """
    # Convert to [B, C, H, W] for torch interpolation
    frames_bchw = frames.permute(0, 3, 1, 2)
    
    # Resize
    resized = torch.nn.functional.interpolate(
        frames_bchw,
        size=target_size,
        mode=mode,
        align_corners=False if mode != "nearest" else None
    )
    
    # Convert back to [B, H, W, C]
    return resized.permute(0, 2, 3, 1)


def encode_video(frames, output_path, frame_rate=24, codec="h264", quality=23, 
                 use_gpu=True, pix_fmt="yuv420p"):
    """
    Encode frames to a video file.
    
    Args:
        frames: Tensor of shape [B, H, W, C] with values 0-1
        output_path: Output video file path
        frame_rate: Output frame rate
        codec: Video codec (h264, hevc, vp9, gif)
        quality: CRF value (0-51, lower = better)
        use_gpu: Whether to use GPU encoding if available
        pix_fmt: Pixel format
        
    Returns:
        str: Path to the output video
    """
    if not IMAGEIO_AVAILABLE:
        raise ImportError("imageio and imageio-ffmpeg are required. Install with: pip install imageio imageio-ffmpeg")
    
    # Move to CPU and convert to numpy
    if isinstance(frames, torch.Tensor):
        frames_np = frames.cpu().numpy()
    else:
        frames_np = frames
    
    # Convert from 0-1 to 0-255 uint8
    frames_uint8 = (frames_np * 255).clip(0, 255).astype(np.uint8)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Get encoder
    if codec == "gif":
        # GIF output
        imageio.mimsave(output_path, frames_uint8, fps=frame_rate, loop=0)
        return output_path
    
    encoder = get_best_encoder(codec, use_gpu)
    
    # Prepare ffmpeg parameters
    ffmpeg_params = []
    
    if "nvenc" in encoder or "amf" in encoder or "qsv" in encoder:
        # Hardware encoder settings
        ffmpeg_params.extend(["-preset", "p4" if "nvenc" in encoder else "balanced"])
        if "nvenc" in encoder:
            ffmpeg_params.extend(["-rc", "vbr", "-cq", str(quality)])
        else:
            ffmpeg_params.extend(["-q:v", str(quality)])
    else:
        # Software encoder settings
        ffmpeg_params.extend(["-preset", "medium", "-crf", str(quality)])
    
    ffmpeg_params.extend(["-pix_fmt", pix_fmt])
    
    # Write video
    writer = imageio.get_writer(
        output_path,
        fps=frame_rate,
        codec=encoder,
        quality=None,  # Use CRF instead
        ffmpeg_params=ffmpeg_params,
        macro_block_size=8
    )
    
    for frame in frames_uint8:
        writer.append_data(frame)
    
    writer.close()
    
    return output_path


def get_video_info(video_path):
    """
    Get information about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        dict: Video information (fps, frame_count, width, height, duration)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    
    cap.release()
    return info
