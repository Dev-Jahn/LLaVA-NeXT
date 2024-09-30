import os.path
from typing import Optional, Dict, List, Callable, Literal
from abc import ABC, abstractmethod
import logging

import decord
from PIL import Image
import av
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, IterableDataset
from decord import VideoReader, cpu

BASE_KEYS = ['video_id', 'url', 'caption']


def fast_resize_opencv(image_array, new_shape):
    return cv2.resize(image_array, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)


def resize_loop(video_array, new_shape):
    return [fast_resize_opencv(frame, new_shape) for frame in video_array]


def resize_hack(video_array, new_shape):
    """ news_shape: (W, H) """
    N, H, W, C = video_array.shape
    instack = video_array.transpose((1, 2, 3, 0)).reshape((H, W, C * N))
    outstack = cv2.resize(instack, new_shape, interpolation=cv2.INTER_LINEAR)
    return outstack.reshape((new_shape[1], new_shape[0], C, N)).transpose((3, 0, 1, 2))


class LocalVideoDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            exts: List[str] = ('.mp4',),
            fps: int = 1,
            n_frames: int = 20,
            frame_size: int = 336,
            transform: Optional[Callable] = None,
            sample_strategy: Literal['fixed', 'dynamic'] = 'fixed',
    ):
        # TODO: Frame resizing is currently not done by dataset. May need perf comparison.
        self.root_dir = root_dir
        self.video_paths = self._get_video_paths(exts)
        self.fps = fps
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.transform = transform
        self.sample_strategy = sample_strategy

    def _get_video_paths(self, exts):
        return [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if
                any(f.endswith(ext) for ext in exts)]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        video_path = self.video_paths[idx]
        frames_array = self._load_video(video_path)
        frames = frames_array if self.transform is None else self.transform(frames_array)

        return {
            'video_id': os.path.splitext(os.path.basename(video_path))[0],
            'video_path': video_path,
            'frames': frames
        }

    def _load_video(self, video_path: str) -> np.array:
        vr = VideoReader(video_path, ctx=cpu(0))
        if self.sample_strategy == 'fixed':
            timestamps = self._sample_fixed()
            indices = (timestamps * vr.get_avg_fps()).round().astype(int)
            # if video is shorter than n_frames, only sample the available frames
            indices = indices[indices < len(vr)]
        else:  # 'dynamic'
            indices = np.linspace(0, len(vr), self.n_frames, endpoint=False).astype(int)
        # batched resizing
        if len(indices) <= 512:
            return resize_hack(vr.get_batch(indices).asnumpy(), (self.frame_size, self.frame_size))
        else:
            return resize_loop(vr.get_batch(indices).asnumpy(), (self.frame_size, self.frame_size))

    def _sample_fixed(self):
        interval = 1 / self.fps
        return np.array([i * interval for i in range(self.n_frames)])

    def _sample_dynamic(self, num_frames, duration):
        if num_frames <= self.n_frames:
            return [i * (duration / num_frames) for i in range(num_frames)]
        else:
            return [i * (duration / (self.n_frames - 1)) for i in range(self.n_frames)]


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
