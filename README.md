# ComfyUI-VideoNodes

Custom nodes for video upload (loading) and video combine (encoding) in ComfyUI with GPU/CPU support.

## Features

- **GPU Acceleration**: Uses NVENC (NVIDIA), AMF (AMD), or QSV (Intel) for video encoding when available
- **CPU Fallback**: Automatically falls back to software encoding (libx264/libx265) on systems without GPU encoders
- **Multiple Formats**: Supports MP4 (H.264/HEVC), WebM (VP9), and GIF output
- **Frame Processing**: All frame operations use PyTorch tensors for GPU-accelerated processing

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone or copy this repository:
   ```bash
   git clone https://github.com/yourusername/ComfyUI-VideoNodes
   # OR copy the ComfyUI-VideoNodes folder directly
   ```

3. Install dependencies:
   ```bash
   cd ComfyUI-VideoNodes
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

## Nodes

### Video Upload
Loads a video file and converts it to a batch of image frames.

**Inputs:**
- `video`: Video file to load (upload via ComfyUI)
- `force_rate`: Target frame rate (0 = keep original)
- `force_size`: Resize options (Disabled, 512x512, 768x?, etc.)
- `frame_load_cap`: Maximum frames to load (0 = no limit)
- `skip_first_frames`: Skip N frames from start
- `select_every_nth`: Select every Nth frame

**Outputs:**
- `images`: Batch of frames as IMAGE tensor
- `frame_count`: Number of frames loaded
- `fps`: Original video frame rate
- `width`: Frame width
- `height`: Frame height

### Video Upload (Path)
Same as Video Upload, but accepts a file path string instead of uploaded file.

### Video Combine
Combines image frames into a video file.

**Inputs:**
- `images`: Batch of images to encode
- `frame_rate`: Output frame rate
- `filename_prefix`: Output filename prefix
- `format`: Output format (mp4_h264, mp4_hevc, webm_vp9, gif)
- `quality`: CRF quality value (0-51, lower = better)
- `use_gpu`: Enable GPU encoding if available
- `loop_count`: Number of times to loop
- `pingpong`: Play forward then backward
- `pix_fmt`: Pixel format (yuv420p recommended)

**Outputs:**
- `file_path`: Path to the saved video

### Video Info
Get information about a video file.

**Outputs:**
- `fps`: Frame rate
- `frame_count`: Total frames
- `width`: Video width
- `height`: Video height
- `duration`: Duration in seconds

### GPU Encoder Info
Displays available GPU encoders for debugging.

## GPU Encoding Support

| GPU | H.264 Encoder | HEVC Encoder |
|-----|---------------|--------------|
| NVIDIA | h264_nvenc | hevc_nvenc |
| AMD | h264_amf | hevc_amf |
| Intel | h264_qsv | - |

If no GPU encoder is available, the nodes automatically fall back to software encoding (libx264/libx265).

## Requirements

- ComfyUI
- Python 3.8+
- PyTorch (installed by ComfyUI)
- OpenCV (`opencv-python`)
- imageio + imageio-ffmpeg

## License

MIT License
