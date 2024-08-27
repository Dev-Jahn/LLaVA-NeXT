import math
import random
from typing import Optional, Dict, Callable
from functools import lru_cache
from contextlib import contextmanager
import multiprocessing as mp

from datasets import load_from_disk, load_dataset
from torch.utils.data import get_worker_info
from yt_dlp.YoutubeDL import DownloadError
from yt_dlp.utils import ExtractorError

from .video import VideoStreamDataset
from .utils import yt2pil


class InternVidDataset(VideoStreamDataset):
    """
    metadata example
    {
        'YoutubeID': 'HdYoyzCSWyw',
        'Start_timestamp': '00:03:10.567',
        'End_timestamp': '00:03:11.200',
        'Caption': 'woman using a computer mouse and keyboard',
        'Aesthetic_Score': 4.58984375,
        'UMT_Score': 0.39794921875
    }

    feature dict
    {
        'video_id': str,
        'url': str,
        'caption': str,
        'frames': List[PIL.Image], # May be altered with video_key
        'duration': float,
        'aes_score': float,
        'umt_score': float
    }
    """

    def __init__(
            self,
            cache_dir: str = None,
            split='FLT',
            fps: int = 1,
            max_frames: int = None,
            frame_size: int = 336,
            transform: Optional[Callable] = None,
            text_preprocess: Optional[Callable] = None,
            video_key: str = 'frames',
            hwaccel=None,
            cookie_path=None,
            debug=False
    ):
        if fps is None and max_frames is not None:
            sample_strategy = 'dynamic'
        elif fps is not None:
            sample_strategy = 'fixed'
        else:
            raise ValueError('One of fps or max_frames must be provided')
        super().__init__(root_dir=cache_dir, split=split,
                         video_id_key='YoutubeID',
                         url_key='YoutubeID',
                         caption_key='Caption',
                         extra_keys=['url', 'Start_timestamp', 'End_timestamp', 'Aesthetic_Score', 'UMT_Score'],
                         fps=fps, n_frames=max_frames,
                         frame_size=frame_size, transform=transform,
                         sample_strategy=sample_strategy,
                         truncate='last',
                         debug=debug)
        self.text_preprocess = text_preprocess
        self.invalid_index_map = {}
        self.video_key = video_key
        self.hwaccel = hwaccel
        self.cookie_path = cookie_path

    def _init_metadata(self, root_dir: str):
        return load_dataset(
            "OpenGVLab/InternVid",
            "InternVid-10M",
            cache_dir=self.root_dir,
            num_proc=min(16, mp.cpu_count())
        )[self.split]

    def _process_extra_keys(self, row: Dict) -> Dict:
        extra = {}
        for key in self.extra_keys:
            if key in ['Start_timestamp', 'End_timestamp']:
                extra[key] = _parse_timestamp(row[key])
            elif key in ['Aesthetic_Score', 'UMT_Score']:
                extra[key] = float(row[key])
            elif key == 'url':
                extra[key] = _get_video_url(row['YoutubeID'])
            else:
                extra[key] = row[key]
        return extra

    def _get_metadata(self, idx: int) -> Dict:
        return self.metadata[idx]

    def fetch_video(self, url: str, start_time: Optional[float] = None, end_time: Optional[float] = None):
        return yt2pil(
            url,
            (self.frame_size, self.frame_size),
            start=start_time, end=end_time,
            fps=self.fps, max_frames=self.n_frames,
            hwaccel=self.hwaccel,
            cookie_path=self.cookie_path,
            debug=self.debug,
        )

    @lru_cache(maxsize=128)
    def __getitem__(self, idx: int, fail=0) -> Dict:
        if idx in self.invalid_index_map:
            idx = self.invalid_index_map[idx]
        try:
            row = self._get_metadata(idx)
            return_dict = self._process_base_keys(row)
            return_dict.update(self._process_extra_keys(row))
            frames = self.fetch_video(return_dict['url'],
                                      return_dict['Start_timestamp'],
                                      return_dict['End_timestamp'])
            if len(frames) == 0:
                raise RuntimeError("FFmpeg failed. Retry with debug=True")
            if self.transform:
                frames = self.transform(frames)
            if self.text_preprocess:
                return_dict.update(**self.text_preprocess(return_dict['caption']))
            return_dict.update({
                f'{self.video_key}': frames,
                'duration': return_dict['End_timestamp'] - return_dict['Start_timestamp'],
            })
            return return_dict
        except (DownloadError, ExtractorError):
            if fail >= 5:
                raise RuntimeError(f"Failed to fetch video at index {idx} after 5 retries")
            with self._seed_context(idx):
                new_idx = random.randint(0, len(self.metadata) - 1)
            self.invalid_index_map[idx] = new_idx
            if self.debug:
                print(f"Failed to fetch video at index {idx}. Replace with index {new_idx}")
            return self.__getitem__(new_idx, fail + 1)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading
            indices = range(len(self))
        else:  # in a worker process
            per_worker = int(math.ceil(len(self) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            indices = range(worker_id * per_worker, min((worker_id + 1) * per_worker, len(self)))
        return (self[i] for i in indices)

    def __repr__(self):
        return (
            f"InternVidDataset(\n"
            f"  root_dir={self.root_dir},\n"
            f"  num_videos={len(self.metadata)},\n"
            f"  fps={self.fps},\n"
            f"  max_frames={self.n_frames},\n"
            f"  frame_size={self.frame_size},\n"
            f"  sample_strategy={self.sample_strategy},\n"
            f")"
        )

    def __len__(self):
        return len(self.metadata)

    @contextmanager
    def _seed_context(self, idx: int):
        state = random.getstate()
        random.seed(42 + idx)
        try:
            yield
        finally:
            random.setstate(state)


def _parse_timestamp(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _get_video_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"
