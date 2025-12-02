"""
Unified Vision Processing Module

Core vision processing functionality for Qwen2.5-VL including:
- Image processing, resizing, and format conversion
- Video processing and frame extraction
- EXIF orientation handling
- Data conversion pipeline image processing

Includes ImageProcessor class for data conversion pipeline.
"""

import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from data_conversion.utils.exif_utils import apply_exif_orientation


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 32
MIN_PIXELS = 4 * 32*32
MAX_PIXELS = 768 * 32*32
MAX_RATIO = 200


VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 32*32
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 32*32 * 0.9))
)
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    # Ensure all parameters are integers
    height = int(height)
    width = int(width)
    factor = int(factor)
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(int(height / beta), factor))
        w_bar = max(factor, floor_by_factor(int(width / beta), factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    """Convert arbitrary PIL image to an RGB image with correct EXIF orientation.

    The function first applies EXIF orientation so any orientation stored in the
    image metadata is materialised in the pixel data, ensuring subsequent
    geometric computations align with the annotation space.
    """
    pil_image = apply_exif_orientation(pil_image)
    return pil_image


def _safe_to_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int with a default fallback."""
    try:
        if isinstance(value, (int, float)):
            return int(value)
        elif isinstance(value, str):
            return int(value)
        elif hasattr(value, "__int__"):
            return int(value)
        else:
            return default
    except (ValueError, TypeError):
        return default


def fetch_image(
    ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR
) -> Image.Image:
    """Fetch and resize image from URL or local path.

    Args:
        ele: A dictionary containing image information.
            - image: URL or local path to image, or PIL Image object.
            - resized_height, resized_width: Optional pre-computed dimensions.
        size_factor: Factor to ensure dimensions are divisible by.

    Returns:
        Resized PIL Image.
    """
    # Handle PIL Image directly
    if "image" in ele and isinstance(ele["image"], Image.Image):
        image = ele["image"]
    # Handle URL or path
    elif "image" in ele and isinstance(ele["image"], str):
        image_path = ele["image"]
        # Handle URL
        if image_path.startswith(("http://", "https://")):
            try:
                response = requests.get(image_path, stream=True, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                logger.error(f"Failed to fetch image from URL: {e}")
                # Create a small placeholder image on error
                image = Image.new("RGB", (size_factor, size_factor), color="gray")
        # Handle local path
        else:
            try:
                image = Image.open(image_path)
            except Exception as e:
                logger.error(f"Failed to open image from path: {e}")
                # Create a small placeholder image on error
                image = Image.new("RGB", (size_factor, size_factor), color="gray")
    elif "image_url" in ele and isinstance(ele["image_url"], str):
        image_url = ele["image_url"]
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            logger.error(f"Failed to fetch image from URL: {e}")
            # Create a small placeholder image on error
            image = Image.new("RGB", (size_factor, size_factor), color="gray")
    else:
        # Create a blank image if no valid source
        logger.warning("No valid image source provided, creating placeholder")
        image = Image.new("RGB", (size_factor, size_factor), color="gray")

    # Convert to RGB
    image = to_rgb(image)

    # Resize based on provided dimensions or calculate from image
    if "resized_height" in ele and "resized_width" in ele:
        # Use pre-computed dimensions if provided
        try:
            # Extract height and width values safely
            height_val = _safe_to_int(ele.get("resized_height"), 480)
            width_val = _safe_to_int(ele.get("resized_width"), 640)

            # Ensure values are integers before passing to smart_resize
            resized_height, resized_width = smart_resize(
                height_val,
                width_val,
                factor=size_factor,
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing dimensions: {e}, using defaults")
            resized_height, resized_width = smart_resize(480, 640, factor=size_factor)
    else:
        # Get dimensions from the image
        if hasattr(image, "size") and len(image.size) >= 2:
            # Get dimensions as integers
            width, height = _safe_to_int(image.size[0]), _safe_to_int(image.size[1])

            min_pixels = _safe_to_int(ele.get("min_pixels"), MIN_PIXELS)
            max_pixels = _safe_to_int(ele.get("max_pixels"), MAX_PIXELS)
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        else:
            # Fallback for invalid images
            resized_height, resized_width = size_factor, size_factor

    # Resize the image
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), (
        "Only accept either `fps` or `nframes`"
    )
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> "tuple[torch.Tensor, float]":
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn(
                "torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0."
            )
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, _, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(
        f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(
        f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}"
    )
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    ele: dict,
) -> "tuple[torch.Tensor, float]":
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord

    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    logger.info(
        f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    )
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def is_torchcodec_available() -> bool:
    """Check if torchcodec is available and properly installed."""
    try:
        import importlib.util

        if importlib.util.find_spec("torchcodec") is None:
            return False
        # Import to verify it's available
        import torchcodec.decoders  # type: ignore  # noqa: F401

        return True
    except (ImportError, AttributeError, Exception):
        return False


def _read_video_torchcodec(
    ele: dict,
) -> "tuple[torch.Tensor, float]":
    """read video using torchcodec.decoders.VideoDecoder

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    from torchcodec.decoders import VideoDecoder  # type: ignore

    TORCHCODEC_NUM_THREADS = int(os.environ.get("TORCHCODEC_NUM_THREADS", 8))
    logger.info(f"set TORCHCODEC_NUM_THREADS: {TORCHCODEC_NUM_THREADS}")
    video_path = ele["video"]
    st = time.time()
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames

    # Handle potential None values from metadata
    if total_frames is None:
        total_frames = 1
    if video_fps is None:
        video_fps = 1.0

    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = decoder.get_frames_at(indices=idx).data
    logger.info(
        f"torchcodec:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    )
    return video, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "torchcodec": _read_video_torchcodec,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(
    ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False
) -> Union[
    "torch.Tensor",
    List[Image.Image],
    Tuple["torch.Tensor", float],
    Tuple[List[Image.Image], float]
]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(
                f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}"
            )
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        min_pixels = int(ele.get("min_pixels", VIDEO_MIN_PIXELS))
        total_pixels = int(ele.get("total_pixels", VIDEO_TOTAL_PIXELS))
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
            int(min_pixels * 1.05),
        )
        max_pixels_supposed = int(ele.get("max_pixels", max_pixels))
        if max_pixels_supposed > max_pixels:
            logger.warning(
                f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
            )
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            # Ensure height and width are integers
            resized_height_val = ele["resized_height"]
            resized_width_val = ele["resized_width"]

            # Convert to integers if needed
            if isinstance(resized_height_val, str):
                resized_height_val = int(resized_height_val)
            elif hasattr(resized_height_val, "__int__"):
                resized_height_val = int(resized_height_val)
            else:
                resized_height_val = int(float(resized_height_val))

            if isinstance(resized_width_val, str):
                resized_width_val = int(resized_width_val)
            elif hasattr(resized_width_val, "__int__"):
                resized_width_val = int(resized_width_val)
            else:
                resized_width_val = int(float(resized_width_val))

            resized_height, resized_width = smart_resize(
                resized_height_val,
                resized_width_val,
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                int(height),
                int(width),
                factor=image_factor,
                min_pixels=int(min_pixels),
                max_pixels=int(max_pixels),
            )
        # Use TF.resize instead of functional.resize to avoid type issues
        video = TF.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image(
                {"image": video_element, **process_info}, size_factor=image_factor
            )
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, float(process_info.pop("fps", 2.0))
        return images


def extract_vision_info(conversations: Union[List[dict], List[List[dict]]]) -> List[dict]:
    """Extract vision information from conversations.

    Args:
        conversations: A list of conversations or a list of list of conversations

    Returns:
        A list of dictionaries containing vision information
    """
    vision_infos: list[dict] = []

    # Handle empty input
    if not conversations:
        return vision_infos

    # Determine format and standardize to list of conversations
    # Check if it's a single conversation (list of messages) vs list of conversations
    is_single_conversation = all(isinstance(item, dict) for item in conversations)

    # Convert single conversation to list of conversations for uniform processing
    conv_list = [conversations] if is_single_conversation else conversations

    # Process each conversation
    for conversation in conv_list:
        for message in conversation:
            if isinstance(message, dict) and "content" in message:
                content = message["content"]
                if isinstance(content, list):
                    for ele in content:
                        if isinstance(ele, dict) and (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele.get("type", "") in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)

    return vision_infos


def process_vision_info(
    conversations: Union[List[dict], List[List[dict]]],
    return_video_kwargs: bool = False,
) -> Tuple[
    Union[List[Image.Image], None],
    Union[List[Union["torch.Tensor", List[Image.Image]]], None],
    Union[Dict[str, List[float]], None],
]:
    """Process vision information from conversations.

    Args:
        conversations: A list of conversations or a list of list of conversations
        return_video_kwargs: Whether to return video keywords arguments

    Returns:
        A tuple of (image_inputs, video_inputs, video_kwargs)
    """
    # Extract vision information from conversations
    vision_infos = extract_vision_info(conversations)

    # Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(
                vision_info, return_video_sample_fps=True
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")

    # Handle empty inputs
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    # Return with or without video kwargs
    if return_video_kwargs:
        return image_inputs, video_inputs, {"fps": video_sample_fps_list}

    # Return a 3-tuple with None as the third element to match the return type
    return image_inputs, video_inputs, None


# Data Conversion Pipeline Image Processor
class ImageProcessor:
    """Unified image processing for the data conversion pipeline."""

    def __init__(self, config):
        """Initialize with configuration."""

        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = config.get_dataset_output_dir()
        self.output_image_dir = config.get_dataset_image_dir()

        logger.info(f"ImageProcessor initialized: resize={config.resize}")

    def to_rgb(self, pil_image: Image.Image) -> Image.Image:
        """
        Convert PIL image to RGB with proper EXIF orientation handling.

        Applies EXIF orientation transformation to ensure image display
        matches annotation space, then converts to RGB with white background
        for transparency handling.
        """
        # Apply EXIF orientation transformation (centralized)
        pil_image = apply_exif_orientation(pil_image)
        return pil_image

    def _process_image_legacy(
        self,
        image_path: Path,
        width: int,
        height: int,
        output_base_dir: Optional[Path] = None,
    ) -> Tuple[Path, int, int]:
        """
        Process a single image: copy or resize with coordinate scaling.

        Returns:
            Tuple of (output_image_path, final_width, final_height)
        """
        from data_conversion.utils.file_ops import FileOperations

        if not self.output_image_dir:
            # No processing needed, return original
            return image_path, width, height

        # Calculate output path
        try:
            rel_path = image_path.relative_to(self.input_dir)
        except ValueError:
            # image_path is not relative to input_dir, it might already be in output_dir
            rel_path = image_path.name

        output_path = self.output_image_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if image already exists and has been processed
        if output_path.exists() and output_path != image_path:
            existing_width, existing_height = FileOperations.get_image_dimensions(
                output_path
            )
            logger.debug(
                f"Using existing processed image: {output_path} ({existing_width}x{existing_height})"
            )
            return output_path, existing_width, existing_height

        if self.config.resize:
            # Smart resize using the smart_resize function from this module
            new_height, new_width = smart_resize(
                height=height,
                width=width,
                factor=IMAGE_FACTOR,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )

            with Image.open(image_path) as img:
                # Apply EXIF orientation and convert to RGB
                processed_img = self.to_rgb(img)

                # Resize image
                resized_img = processed_img.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                resized_img.save(output_path)

            logger.debug(
                f"Resized {image_path.name}: {width}x{height} → {new_width}x{new_height} (MAX_PIXELS={MAX_PIXELS})"
            )
            return output_path, new_width, new_height

        else:
            # Copy with EXIF orientation handling
            if not output_path.exists():
                with Image.open(image_path) as img:
                    processed_img = self.to_rgb(img)
                    processed_img.save(output_path)

            logger.debug(f"Copied {image_path.name} with EXIF orientation applied")
            return output_path, width, height

    def process_image(
        self,
        image_path: Path,
        width: int,
        height: int,
        output_base_dir: Optional[Path] = None,
        final_width: Optional[int] = None,
        final_height: Optional[int] = None,
    ) -> Tuple[Path, int, int]:
        """Process a single image using the canonical resize + EXIF pipeline.

        Pipeline:
          1. Apply EXIF orientation so pixels match the annotation space.
          2. If ``final_width``/``final_height`` are provided, resize to that size.
          3. Otherwise, if ``self.config.resize`` is True, apply :func:`smart_resize`
             to the EXIF dimensions.
          4. Otherwise, only EXIF orientation is applied (no geometric resize).

        Returns:
            Tuple of (output_image_path, final_width, final_height)
        """
        from data_conversion.utils.file_ops import FileOperations

        # If no output image directory is configured, fall back to returning the
        # original path with the best-effort dimensions from the caller.
        if not self.output_image_dir:
            out_w = final_width if final_width is not None else width
            out_h = final_height if final_height is not None else height
            return image_path, out_w, out_h

        # Calculate output path
        try:
            rel_path = image_path.relative_to(self.input_dir)
        except ValueError:
            # image_path is not relative to input_dir, it might already be in output_dir
            rel_path = image_path.name

        output_path = self.output_image_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if image already exists and has been processed; trust its size.
        if output_path.exists() and output_path != image_path:
            existing_width, existing_height = FileOperations.get_image_dimensions(
                output_path
            )
            logger.debug(
                f"Using existing processed image: {output_path} ({existing_width}x{existing_height})"
            )
            return output_path, existing_width, existing_height

        # Materialise EXIF orientation before any resizing so that geometry and
        # pixels live in the same coordinate frame.
        with Image.open(image_path) as img:
            processed_img = self.to_rgb(img)
            base_width, base_height = processed_img.size

            # Decide target size
            if (final_width is None) != (final_height is None):
                raise ValueError(
                    "process_image received mismatched final dimensions: "
                    f"final_width={final_width}, final_height={final_height}"
                )

            if final_width is not None and final_height is not None:
                target_width, target_height = final_width, final_height
            elif self.config.resize:
                # Smart resize using the canonical implementation on EXIF size
                target_height, target_width = smart_resize(
                    height=base_height,
                    width=base_width,
                    factor=IMAGE_FACTOR,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                )
            else:
                target_width, target_height = base_width, base_height

            # Resize only if needed
            if target_width != base_width or target_height != base_height:
                resized_img = processed_img.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )
                resized_img.save(output_path)
                logger.debug(
                    f"Resized {image_path.name}: "
                    f"{base_width}x{base_height} → {target_width}x{target_height} "
                    f"(MAX_PIXELS={MAX_PIXELS})"
                )
            else:
                processed_img.save(output_path)
                logger.debug(
                    f"Copied {image_path.name} with EXIF orientation applied "
                    f"(dimensions {base_width}x{base_height})"
                )

        return output_path, target_width, target_height


    def get_relative_image_path(self, absolute_image_path: Path) -> str:
        """Get relative image path for use in JSONL files."""
        try:
            # Make path relative to dataset output directory
            rel_path = absolute_image_path.relative_to(self.output_dir)
            return str(rel_path)
        except ValueError:
            # If path is not relative to output_dir, just return the name under images/
            return f"images/{absolute_image_path.name}"

    def scale_object_coordinates(
        self,
        objects: List[Dict],
        original_width: int,
        original_height: int,
        new_width: int,
        new_height: int,
    ) -> None:
        """Scale bounding box coordinates in-place for resized images."""
        from data_conversion.pipeline.coordinate_manager import CoordinateManager

        if original_width == new_width and original_height == new_height:
            return  # No scaling needed

        for obj in objects:
            bbox = obj["bbox_2d"]
            try:
                scaled_bbox = CoordinateManager.apply_smart_resize_scaling(
                    bbox, original_width, original_height, new_width, new_height
                )
                obj["bbox_2d"] = scaled_bbox
            except ValueError as e:
                logger.error(f"Error scaling bbox {bbox}: {e}")
                if self.config.fail_fast:
                    raise

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of image processing operations."""
        summary = {
            "resize": self.config.resize,
            "input_dir": str(self.input_dir),
            "output_image_dir": str(self.output_image_dir)
            if self.output_image_dir
            else None,
        }

        if self.output_image_dir and self.output_image_dir.exists():
            # Count processed files
            image_files = list(self.output_image_dir.glob("*.{jpeg,jpg}"))
            json_files = list(self.output_image_dir.glob("*.json"))

            summary.update(
                {
                    "processed_images": len(image_files),
                    "processed_jsons": len(json_files),
                }
            )

        return summary
