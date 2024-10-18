import os
import json
from typing import List, Optional, Callable, Literal, Tuple

import datasets

from .video import LocalVideoDataset
from ..model import AVAILABLE_MODELS

AVAILABLE_CONFIGS = [
]


class LlavaVideo178k(LocalVideoDataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        labeltype: Literal['none', 'captions', 'qa', 'instruct'],
        exts: List[str] = ('.mp4', '.mkv', '.webm'),
        n_frames: int = 32,
        fps: int = None,
        frame_shape: Tuple[int] = (336, 336),
        transform: Optional[Callable] = lambda x: x,
        text_preprocess: Optional[Callable] = lambda x: x,
    ):
        assert split in ['train', 'val', 'test']
        assert labeltype in ['none', 'captions', 'qa', 'instruct']
        root_dir = os.path.expanduser(root_dir)
        match split:
            case 'train' | 'val':
                video_dir = os.path.join(root_dir, 'videos', 'trainval')
            case 'test':
                video_dir = os.path.join(root_dir, 'videos', 'test')
            case _:
                raise ValueError(f"Invalid split {split}")
        super().__init__(video_dir, exts, n_frames=n_frames, fps=fps, frame_shape=frame_shape, transform=transform)
        self.root_dir = root_dir
        self.split = split
        self.id_path_map = {os.path.splitext(os.path.split(path)[-1])[0]: path for path in self.video_paths}
        self.labeltype = labeltype
        self.labels = self._load_labels()
        self.video_used = list((set([label['video_id'] for label in self.labels])))
        self.text_preprocess = text_preprocess

    def _load_labels(self) -> list:
        labels = []
        if self.labeltype == 'none':
            return
        elif self.labeltype == 'captions':
            with open(os.path.join(self.root_dir, 'captions', f'{self.split}.json'), 'r') as f:
                captions = json.load(f)
            for video_id, data in captions.items():
                for i, timestamp in enumerate(data['timestamps']):
                    labels.append({
                        'video_id': video_id,
                        'timestamp': timestamp,
                        'caption': data['sentences'][i].strip(),
                    })
        elif self.labeltype == 'qa':
            with open(os.path.join(self.root_dir, 'qa', f'{self.split}_q.json'), 'r') as f:
                questions = json.load(f)
            with open(os.path.join(self.root_dir, 'qa', f'{self.split}_a.json'), 'r') as f:
                answers = json.load(f)
                answers = {a['question_id']: a for a in answers}
            for q in questions:
                labels.append({
                    'video_id': 'v_' + q['video_name'],
                    'q': q['question'].capitalize().strip() + '?',
                    'a': answers[q['question_id']]['answer'].capitalize().strip(),
                })
        elif self.labeltype == 'instruct':
            with open(os.path.join(self.root_dir, 'VideoInstruct100K.json'), 'r') as f:
                instructions = json.load(f)
            for inst in instructions:
                labels.append({
                    'video_id': inst['video_id'],
                    'q': inst['q'],
                    'a': inst['a'],
                })
        return labels

    def __len__(self):
        if self.labeltype == 'none':
            return len(self.video_paths)
        else:
            return len(self.labels)

    def __getitem__(self, idx: int):
        if self.labeltype == 'none':
            frames = self._load_video(self.video_paths[idx])
            return {'video_id': self.id_path_map[idx], 'video': frames}
        else:
            label = self.labels[idx]
            start, end = label.get('timestamp', (None, None))
            frames = self._load_video(
                self.id_path_map[label['video_id']],
                start=start, end=end
            )
            return dict({
                'video': self.transform(frames),
                **self.text_preprocess(label),
            })

    def __repr__(self):
        return f"""ActivityNet {self.labeltype.capitalize()} Dataset
        Root dir: {self.root_dir}
        Split: {self.split}
        Number of samples: {len(self)}
        Video used: {len(self.video_used)}
        Frame shape: {self.frame_shape}"""
