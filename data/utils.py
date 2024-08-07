import yt_dlp
import subprocess
import io
from PIL import Image
import os


def yt2pil(video_url, resolution, fps=None, max_frames=None, start=None, end=None, debug=False):
    target_width, target_height = resolution
    # suppress yt-dlp logging
    ydl_opts = {
        'format': 'best',
        # 'format': f'bestvideo[height<={target_height*2}]', # for faster loading
        'quiet': False if debug else True,
        'no_warnings': False if debug else True,
        'logger': None
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        # raise Runtime error if video is private
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
        print(f'duration: {start} ~ {end} ({clip_duration}s)')

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

    # ffmpeg commands
    ffmpeg_cmd = [
        'ffmpeg',
        '-ss', str(start),
        '-i', url,
        '-to', str(end),
        '-vf',
        (
            f'{fps_filter},'
            f'scale={new_width}:{new_height},'
            f'pad={target_width}:{target_height}:{pad_x}:{pad_y}:black'
        ),
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-loglevel', 'error' if not debug else 'info',  # avoid logs
        '-'
    ]

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
