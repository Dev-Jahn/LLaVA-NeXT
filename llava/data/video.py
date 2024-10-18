import os.path
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Callable, Literal
import json
import logging
import subprocess

import av
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, IterableDataset
from decord import VideoReader, cpu

from .utils import get_video_info, suppress_system, get_video_frame_count

BASE_KEYS = ['video_id', 'url', 'caption']
AVAILABLE_BACKENDS = ['decord', 'pyav', 'opencv', 'ffmpeg']
logger = logging.getLogger(__name__)


def load_video_preserve_ratio(video_path: str,
                              fps: int = None,
                              n_frames: int = None,
                              start: float = None, end: float = None,
                              width=None, height=None,
                              video_backend='decord',
                              debug=False) -> np.array:
    """ Load video from video_path with the specified backend
    Note:
        This function does not match the width and height given.
        Do letterbox pad, center-crop, or resize afterward to match the desired shape.
    Args:
        video_path (str): Path to video file
        fps (int): Frame per second to sample, exclusive with n_frames
        n_frames (int): Number of frames to sample, exclusive with fps
        start (float): Start time in seconds
        end (float): End time in seconds
        width (int): Width of the final frame, original width if None
        height (int): Height of the final frame, original height if None
        video_backend (str): Video backend to use
        debug (bool): Debug mode
    Returns:
        np.array: Video frames
    """
    assert video_backend in AVAILABLE_BACKENDS, f"Invalid video backend {video_backend}"
    assert fps is None or n_frames is None, "fps and n_frames should be exclusive"

    # pre-resize to match shorter side for speed up
    vstream, astream, format_ = get_video_info(video_path)
    origin_w, origin_h = int(vstream['width']), int(vstream['height'])
    if width is None and height is None:
        width, height = origin_w, origin_h
    elif width is None or height is None:
        width = width or round(height * origin_w / origin_h)
        height = height or round(width * origin_h / origin_w)
    else:
        if origin_w < origin_h:
            width, height = (width, round(height * origin_h / origin_w))
        else:
            width, height = (round(width * origin_w / origin_h), height)

    # Sample frame indices with given strategy
    timestamps = _get_video_timestamps(video_path)
    start = max(start or 0., 0.)
    end = min(end or timestamps[-1, 1], timestamps[-1, 1])
    if fps is not None:
        indices = _sample_fps(start, end, fps, timestamps)
    elif n_frames is not None:
        indices = _sample_n_frames(start, end, n_frames, timestamps)
    else:  # load all frames
        indices = list(range(len(timestamps)))

    match video_backend:
        case 'decord':
            return _load_video_decord(video_path, indices, width, height, debug)
        case 'pyav':
            return _load_video_pyav(video_path, indices, width, height, debug)
        case 'opencv':
            return _load_video_opencv(video_path, indices, width, height, debug)
        case 'ffmpeg':
            return _load_video_ffmpeg(video_path, indices, width, height, debug)
        case _:
            raise ValueError(f"Invalid video backend {video_backend}")


def _load_video_decord(video_path: str,
                       indices=List[int],
                       width=None, height=None,
                       debug=False) -> np.array:
    """ Load video using decord """
    vr = VideoReader(video_path, width=width, height=height, ctx=cpu(0))
    # batched resizing
    with suppress_system(not debug):
        if len(indices) <= 512:
            return vr.get_batch(indices).asnumpy()
        else:
            return vr.get_batch(indices).asnumpy()


def _load_video_pyav(video_path: str,
                     indices=List[int],
                     width=None, height=None,
                     debug=False) -> np.array:
    """ Load video using pyav """
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.codec_context.width = width
        stream.codec_context.height = height

        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                img = frame.to_ndarray(format='rgb24')
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                frames.append(img)
            if i >= max(indices):
                break
    return np.stack(frames)


def _load_video_opencv(video_path: str,
                       indices=List[int],
                       width=None, height=None,
                       debug=False) -> np.array:
    """ Load video using opencv """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in indices:
        with suppress_system(not debug):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            logger.warning(f"Frame {idx} not loaded from {video_path}")
            continue
    cap.release()
    return np.stack(frames)


def _load_video_ffmpeg(video_path: str,
                       indices=List[int],
                       width=None, height=None,
                       debug=False) -> np.array:
    """ Load video using ffmpeg """
    select_string = "+".join(f"eq(n,{i})" for i in indices)
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'select=\'{select_string}\',scale={width}:{height}',
        '-vsync', '0',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-'
    ]
    with suppress_system(not debug):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
    if process.returncode != 0:
        logger.error(f"FFmpeg error: {err.decode()}")
        return np.array([])
    frames = np.frombuffer(out, dtype=np.uint8).reshape(-1, height, width, 3)
    return frames


def _get_video_timestamps(video_path: str) -> np.array:
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'packet=pts_time,duration_time',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")
    frames_info = json.loads(result.stdout)
    ts = np.array(
        [(float(p['pts_time']), float(p['pts_time']) + float(p['duration_time'])) for p in frames_info['packets']])
    # Truncate for packet/frame mismatch e.g. VP9-MKV
    return ts[:get_video_frame_count(video_path)]


def _sample_fps(start, end, fps, timestamps):
    duration = end - start
    target_frame_count = int(duration * fps)
    if target_frame_count <= 1:
        return [_get_nearest_idx(timestamps, start)]
    targets = np.linspace(start, end, num=target_frame_count)
    return [_get_nearest_idx(timestamps, t) for t in targets]


def _sample_n_frames(start, end, n_frames, timestamps):
    if n_frames <= 1:
        return [_get_nearest_idx(timestamps, start)]
    targets = np.linspace(start, end, num=n_frames)
    return [_get_nearest_idx(timestamps, t) for t in targets]


def _get_nearest_idx(timestamps, target):
    frame_centers = (timestamps[:, 0] + timestamps[:, 1]) / 2
    return np.abs(frame_centers - target).argmin()


def fast_resize_opencv(image_array, new_shape, letterbox=False):
    if letterbox:
        # TODO
        raise NotImplementedError
    else:
        return cv2.resize(image_array, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)


def resize_loop(video_array, new_shape, letterbox=False):
    if letterbox:
        # TODO
        raise NotImplementedError
    else:
        return np.stack([fast_resize_opencv(frame, new_shape, letterbox) for frame in video_array])


def resize_hack(video_array, new_shape, letterbox=False):
    """ news_shape: (W, H) """
    if letterbox:
        # TODO
        raise NotImplementedError
    else:
        N, H, W, C = video_array.shape
        instack = video_array.transpose((1, 2, 3, 0)).reshape((H, W, C * N))
        outstack = cv2.resize(instack, new_shape, interpolation=cv2.INTER_LINEAR)
        return outstack.reshape((new_shape[1], new_shape[0], C, N)).transpose((3, 0, 1, 2))


class LocalVideoDataset(Dataset):
    def __init__(
        self,
        video_dir: str,
        exts: List[str] = ('.mp4',),
        n_frames: int = 32,
        fps: int = None,
        frame_shape: tuple[int, int] = (None, None),
        transform: Optional[Callable] = None,
        video_backend: str = 'decord',
        debug=False
    ):
        assert n_frames is None or fps is None, "n_frames and fps should be exclusive"
        assert video_backend in AVAILABLE_BACKENDS, f"Invalid video backend {video_backend}"
        # TODO: Frame resizing is currently not done by dataset. May need perf comparison.
        self.video_dir = video_dir
        self.video_paths = self._get_video_paths(exts)
        self.fps = fps
        self.n_frames = n_frames
        self.frame_shape = frame_shape
        self.transform = transform if transform else lambda x: x
        self.video_backend = video_backend
        self.debug = debug

    def _get_video_paths(self, exts):
        return [os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir) if
                any(f.endswith(ext) for ext in exts)]

    def __len__(self):
        return len(self.video_paths)

    def _load_video(self, video_path, start=None, end=None):
        return load_video_preserve_ratio(
            video_path,
            fps=self.fps, n_frames=self.n_frames,
            start=start, end=end,
            width=self.frame_shape[0], height=self.frame_shape[1],
            video_backend=self.video_backend,
            debug=self.debug)

    def __getitem__(self, idx: int):
        video_path = self.video_paths[idx]
        frames = self._load_video(video_path)
        return {
            'video_id': os.path.splitext(os.path.basename(video_path))[0],
            'video_path': video_path,
            'frames': self.transform(frames)
        }


class VideoStreamDataset(IterableDataset, ABC):
    """
    Base class for video stream dataset.
    Made for compatibility with any other dataset provides url.
    You may implement the dataset-specific logic by inheriting this class.

    Some arguments should be considered befor using:
    - transform:
        A callable that takes a video frame tensor(C x H x W) and returns a transformed tensor.
    - sample_strategy:
        'fixed' or 'dynamic'. Strategy to match the frame dimension for batching.
        'fixed':
            Sample video with fps.
            Longer videos are truncated to match the n_frames.
        'dynamic':
            Ignores fps, truncate, options.
            Dynamically sample frames to match the n_frames.
    - truncate:
        'last' or 'edge'. Truncation strategy for longer videos.
        'last':
            Truncate the last frames.
        'edge':
            Truncate the start and end frames to get center frames.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        video_id_key: str,
        url_key: str,
        caption_key: str,
        extra_keys: Optional[List] = None,
        fps: int = 1,
        n_frames: int = 20,
        frame_size: int = 336,
        preserve_ratio: bool = True,
        transform: Optional[Callable] = None,
        sample_strategy: Literal['fixed', 'dynamic'] = 'fixed',
        truncate: Literal['last', 'edge'] = 'last',
        debug=False
    ):
        self.root_dir = root_dir
        self.split = split
        self.metadata = self._init_metadata(root_dir)
        self.video_id_key = video_id_key
        self.url_key = url_key
        self.caption_key = caption_key
        self.extra_keys = extra_keys
        self.fps = fps
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.transform = transform
        self.sample_strategy = sample_strategy
        self.truncate = truncate
        self.debug = debug

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    @abstractmethod
    def _init_metadata(self, root_dir: str):
        raise NotImplementedError

    def fetch_video(self,
                    url: str,
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None):
        self.logger.debug(f'Fetching video from {url}...')
        with av.open(url) as container:
            self.logger.debug(f'Container info: {container}')
            stream = container.streams.video[0]
            if self.fps == 1:
                stream.codec_context.skip_frame = 'NONKEY'
            else:
                stream.codec_context.skip_frame = 'NONE'
            # Get total number of frames and duration
            container.seek(0)
            num_frames = stream.frames
            duration = float(stream.duration * stream.time_base)

            # Sample frame indices
            self.logger.debug(f'Extracting timestamps with {self.sample_strategy} strategy')
            if self.sample_strategy == 'fixed':
                target_timestamps = self._sample_fixed()
            else:  # 'dynamic'
                target_timestamps = self._sample_dynamic(num_frames, duration)
            self.logger.debug(f'Number of target timestamps: {len(target_timestamps)}')
            self.logger.debug(f'Target timestamps: {target_timestamps}')

            # Extract only required frames
            frames = []
            prev_frame = None
            prev_closest_idx = None
            for frame in container.decode(stream):
                if start_time is not None and frame.time < start_time:
                    continue
                if end_time is not None and frame.time > end_time:
                    break
                # Find the closest timestamp
                closest_idx = min(range(len(target_timestamps)), key=lambda x: abs(target_timestamps[x] - frame.time))
                closest = target_timestamps[closest_idx]
                # If closest changes or cursor going apart from target
                if closest_idx != prev_closest_idx or abs(closest - frame.time) >= abs(closest - prev_frame.time):
                    append_frame = prev_frame.to_image() if prev_frame else frame.to_image()
                    frames.append(append_frame)
                    target_timestamps.pop(closest_idx)
                prev_frame, prev_closest_idx = frame, closest_idx
                if len(frames) == self.n_frames:
                    break

        return frames

    def _sample_fixed(self):
        # Calculate the interval between frames based on fps
        interval = 1 / self.fps
        # Generate timestamps for the maximum number of frames
        return [i * interval for i in range(self.n_frames)]

    def _sample_dynamic(self, num_frames, duration):
        if num_frames <= self.n_frames:
            # If we have fewer frames than max_frames, sample all frames
            return [i * (duration / num_frames) for i in range(num_frames)]
        else:
            # If we have more frames than max_frames, sample evenly
            return [i * (duration / (self.n_frames - 1)) for i in range(self.n_frames)]

    def _process_base_keys(self, row: Dict) -> Dict:
        return_dict = {}
        for key in BASE_KEYS:
            if key == 'video_id':
                return_dict[key] = row[self.video_id_key]
            elif key == 'url':
                return_dict[key] = row[self.url_key]
            elif key == 'caption':
                return_dict[key] = row[self.caption_key]
            else:
                raise ValueError(f'Unknown key: {key}')
        return return_dict

    @abstractmethod
    def _process_extra_keys(self, row: Dict) -> Dict:
        return dict()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self)
        else:
            per_worker = int(np.ceil(len(self) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self))

        for idx in range(iter_start, iter_end):
            row = self._get_metadata(idx)
            return_dict = self._process_base_keys(row)
            return_dict.update(self._process_extra_keys(row))
            frames = self.fetch_video(return_dict['url'])
            if self.transform is not None:
                frames = [self.transform(frame) for frame in frames]
            return_dict.update({'video_frames': frames})
            yield return_dict

    @abstractmethod
    def _get_metadata(self, idx: int) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {len(self)} items.")
        row = self._get_metadata(idx)
        return_dict = self._process_base_keys(row)
        return_dict.update(self._process_extra_keys(row))
        frames = self.fetch_video(return_dict['url'])
        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]
        return_dict['video_frames'] = frames
        return return_dict
