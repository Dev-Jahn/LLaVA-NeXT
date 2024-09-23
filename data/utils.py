import os
import sys
import logging
from contextlib import contextmanager
import subprocess

import yt_dlp
from PIL import Image


class DummyLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        pass


def get_dummy_logger(name='dummy'):
    logging.setLoggerClass(DummyLogger)
    return logging.getLogger(name)


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
    with suppress_stderr(not debug):
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


@contextmanager
def suppress_stderr(enabled=True):
    if enabled:
        save = sys.stderr
        devnull = os.devnull if sys.platform != 'win32' else 'nul'
        try:
            with open(devnull, 'w') as fnull:
                sys.stderr = fnull
                yield
        finally:
            sys.stderr = save
    else:
        yield
