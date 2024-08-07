import yt_dlp
import subprocess
import io
from PIL import Image
import os


def extract_frames(video_url, target_width, target_height, start=None, end=None, fps=None, max_frames=None):
    # yt-dlp 로그 억제
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
        'logger': None
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        url = info['url']
        width = info['width']
        height = info['height']
        duration = info['duration']  # 영상 총 길이(초)

    # 시작 및 종료 시간 설정
    start = start if start is not None else 0
    end = end if end is not None else duration
    clip_duration = end - start

    # Letterbox 패딩 계산
    aspect_ratio = min(target_width / width, target_height / height)
    new_width = int(width * aspect_ratio)
    new_height = int(height * aspect_ratio)
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2

    # fps 또는 간격 계산
    if fps is None and max_frames is not None:
        interval = clip_duration / max_frames
        fps_filter = f'fps=1/{interval}'
    elif fps is not None:
        fps_filter = f'fps={fps}'
    else:
        fps_filter = 'fps=1'  # 기본값: 1 FPS

    # ffmpeg 명령어 구성
    ffmpeg_cmd = [
        'ffmpeg',
        '-ss', str(start),
        '-i', url,
        '-to', str(end),
        '-vf', f'{fps_filter},scale={new_width}:{new_height},pad={target_width}:{target_height}:{pad_x}:{pad_y}:black',
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-loglevel', 'error',
        '-'
    ]

    # ffmpeg 실행 및 프레임 추출
    with open(os.devnull, 'wb') as devnull:
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=devnull, bufsize=10 ** 8)

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
