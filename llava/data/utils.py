import json
import os
import logging
import subprocess

import yt_dlp
from PIL import Image
import cv2
import ffmpeg
from torch.utils.data import DataLoader, BatchSampler
from accelerate.data_loader import DataLoaderShard, BatchSamplerShard


class ParallelLoaderWrapper:
    """
    Change the default behavior of pytorch DataLoader
    Pytorch:
        Each worker prefetches entire batch.
        num_worker batches are queued when 1 batch is ready.
        Prefetches num_worker * batch_size * prefetch_factor samples.
    This:
        Each worker prefetches single sample.
        Collate after 1 batch is ready.
        Prefetches only batchsize * prefetch_factor samples.
    NOTE: Wrap the DataLoader after prepare if using Accelerate
    """

    def __init__(self, loader_like):
        # Accelerate compatibility
        if isinstance(loader_like, DataLoaderShard):
            self.loader = loader_like.base_dataloader
        elif isinstance(loader_like, DataLoader):
            self.loader = loader_like
        else:
            raise TypeError('Unsupported loader type')

        # Disable batching and intercept data collation
        # Reassigning some attributes to cancel DataLoader initialization
        self.loader._DataLoader__initialized = False
        if hasattr(self.loader, 'batch_sampler'):
            if isinstance(self.loader.batch_sampler, BatchSamplerShard):
                self.batch_size = self.loader.batch_sampler.batch_size
                self.loader.batch_sampler.batch_size = 1
                self.loader.batch_sampler.batch_sampler.batch_size = 1
            elif isinstance(self.loader.batch_sampler, BatchSampler):
                self.batch_size = self.loader.batch_sampler.batch_size
                self.loader.batch_sampler.batch_size = 1
        else:
            self.batch_size = getattr(self.loader, 'batch_size')
            self.loader.batch_size = 1
        self.collate_fn = getattr(self.loader, 'collate_fn')
        self.loader.collate_fn = lambda x: x
        self.loader._DataLoader__initialized = True

    def __getattr__(self, item):
        """redirect attributes"""
        if item == 'loader':
            return self.loader
        else:
            return getattr(self.loader, item)

    def __iter__(self):
        batch = []
        for d in self.loader:
            batch.extend(d)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch.clear()
        if len(batch) > 0:
            yield self.collate_fn(batch)

    def __len__(self):
        return len(self.loader) // self.batch_size + (1 if len(self.loader) % self.batch_size > 0 else 0)


class DummyLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        pass


def get_dummy_logger(name='dummy'):
    logging.setLoggerClass(DummyLogger)
    return logging.getLogger(name)


def get_video_info(path):
    probe = ffmpeg.probe(path)
    vstream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    astream = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    format_ = probe['format']
    return vstream, astream, format_


def get_video_packet_count(path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0',
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")
    return json.loads(result.stdout)


def get_video_frame_count_slow(path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=nb_read_frames',
        '-of', 'csv=p=0',
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")
    return json.loads(result.stdout)


def get_video_frame_count(path):
    cap = cv2.VideoCapture(path)
    cnt_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # Some VP9-MKV, cv2: 4563, ffmpeg: 4562, packets: 4734, have no fucking idea
    if cnt_cv2 == get_video_packet_count(path):
        return cnt_cv2
    else:
        return get_video_frame_count_slow(path)


def yt2pil(
    video_url, resolution, fps=None, max_frames=None, start=None, end=None,
    cookie_path=None, hwaccel=None, debug=False
):
    if cookie_path:
        cookie_path = os.path.abspath(os.path.expanduser(cookie_path))
        assert os.path.exists(cookie_path), f'cookie file not found: {cookie_path}'
    target_width, target_height = resolution
    # suppress yt-dlp logging
    ydl_opts = {
        'format': 'best',
        # 'format': f'bestvideo[height<={target_height*2}]', # for faster loading
        'quiet': False if debug else True,
        'no_warnings': False if debug else True,
        'logger': None if debug else get_dummy_logger(),
        'user_agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36',
        ),
        'cookiefile': cookie_path,
    }
    with suppress_system(not debug):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if 'entries' in info:
                info = info['entries'][0]
            url = info['url']
            width = info['width']
            height = info['height']
            duration = info['duration']

    # set start and end
    start = start if start is not None else 0
    end = end if end is not None else duration
    clip_duration = end - start
    if debug:
        print(f'resolution: {width}x{height} -> {target_width}x{target_height}')
        print(f'duration: {start} ~ {end} ({clip_duration:.2f}s)')

    # Letterbox pad
    aspect_ratio = min(target_width / width, target_height / height)
    new_width = int(width * aspect_ratio)
    new_height = int(height * aspect_ratio)
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2

    # fixed or dynamic sampling
    if fps is None and max_frames is not None:
        interval = clip_duration / max_frames
        fps_filter = f'fps=1/{interval}'
    elif fps is not None:
        fps_filter = f'fps={fps}'
    else:
        raise ValueError('One of fps or max_frames must be provided')

    dev_args = {
        # 'vaapi': ('-init_hw_device', 'vaapi=va:/dev/dri/renderD128',  # NOTE: for Intel GPU
        #           '-hwaccel', 'vaapi',
        #           '-hwaccel_device', 'va'),
        'vaapi': ('-hwaccel', 'vaapi'),
        'cuda': ('-hwaccel', 'cuda'),

    }.get(hwaccel, ())
    vf_filters = []
    if hwaccel == 'cuda':
        vf_filters.extend([
            'hwupload_cuda',
            fps_filter,
            f'scale_npp=w={target_width}:h={target_height}:force_original_aspect_ratio=increase:format=yuv444p',
            'hwdownload',
            'format=yuv444p',  # reconvert before rgb conversion
            # f'pad={target_width}:{target_height}:{pad_x}:{pad_y}:black'
        ])
    else:
        vf_filters.extend([
            fps_filter,
            f'scale={new_width}:{new_height}',
            f'pad={target_width}:{target_height}:{pad_x}:{pad_y}:black'
        ])
    # ffmpeg commands
    ffmpeg_cmd = [
        'ffmpeg',
        *dev_args,
        '-ss', str(start),
        '-i', url,
        '-to', str(end),
        '-vf', ','.join(vf_filters),
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-loglevel', 'error' if not debug else 'info',  # avoid logs
        '-'
    ]
    if debug:
        print(' '.join(ffmpeg_cmd))

    # redirect stderr to /dev/null
    with open(os.devnull, 'wb') as devnull:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE, stderr=devnull if not debug else None,
            bufsize=10 ** 8
        )

    frames = []
    frame_count = 0
    while True:
        raw_frame = process.stdout.read(target_width * target_height * 3)
        if not raw_frame:
            break
        if max_frames and frame_count >= max_frames:
            break
        pil_image = Image.frombytes('RGB', (target_width, target_height), raw_frame)
        frames.append(pil_image)
        frame_count += 1

    process.terminate()
    return frames


class suppress_system:
    def __init__(self, enabled=True):
        self.enabled = enabled
        if enabled:
            self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
            self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        if self.enabled:
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)
            return self
        else:
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)

            for fd in self.null_fds + self.save_fds:
                os.close(fd)
