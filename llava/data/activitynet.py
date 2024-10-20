import os
import json
from typing import List, Optional, Callable, Literal, Tuple

import datasets

from .video import LocalVideoDataset


class ActivityNet(LocalVideoDataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        labeltype: Literal['none', 'captions', 'qa', 'instruct'],
        exts: List[str] = ('.mp4', '.mkv', '.webm'),
        n_frames: int = 32,
        fps: int = None,
        frame_shape: Tuple[int] = (None, None),
        transform: Optional[Callable] = lambda x: x,
        text_preprocess: Optional[Callable] = lambda x: x,
        no_video: bool = False,
        **kwargs,
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
        super().__init__(video_dir, exts,
                         n_frames=n_frames, fps=fps, frame_shape=frame_shape, transform=transform, **kwargs)
        self.root_dir = root_dir
        self.split = split
        self.id_path_map = {os.path.splitext(os.path.split(path)[-1])[0]: path for path in self.video_paths}
        self.path_id_map = {v: k for k, v in self.id_path_map.items()}
        self.labeltype = labeltype
        self.labels = self._load_labels()
        if self.labeltype != 'none':
            self.video_used = list((set([label['video_id'] for label in self.labels])))
        else:
            self.video_used = list(self.id_path_map.keys())
        self.text_preprocess = text_preprocess
        self.no_video = no_video

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
        return_dict = {}
        if self.labeltype == 'none':
            path = self.video_paths[idx]
            return_dict.update({'video_id': self.path_id_map[path]})
        else:
            path = self.id_path_map[self.labels[idx]['video_id']]
            return_dict.update(self.text_preprocess(self.labels[idx]))
        if not self.no_video:
            start, end = self.labels[idx].get('timestamp', (None, None)) if self.labeltype != 'none' else (None, None)
            return_dict.update(
                {'video': self.transform(self._load_video(path, start=start, end=end))})
        return return_dict

    def __repr__(self):
        return f"""ActivityNet {self.labeltype.capitalize()} Dataset
        Root dir: {self.root_dir}
        Split: {self.split}
        Number of samples: {len(self)}
        Video used: {len(self.video_used)}
        Frame shape: {self.frame_shape}"""


_DESCRIPTION = """\
"""
_URLS = {
    "none": "https://example.com",
    "captions": "https://example.com",
    "qa": "https://example.com",
    "instruct": "https://example.com",
}

logger = datasets.logging.get_logger(__name__)


class ActivityNetConfig(datasets.BuilderConfig):
    def __init__(self,
                 label_type: str = 'none',
                 n_frames: Optional[int] = 32,
                 fps: Optional[int] = None,
                 **kwargs):
        assert label_type in ['none', 'captions', 'qa', 'instruct']
        assert (n_frames is None) != (fps is None), "n_frames and fps should be exclusive"
        super().__init__(**kwargs)
        self.label_type = label_type
        self.n_frames = n_frames
        self.fps = fps


class ActivityNetHF(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        ActivityNetConfig(name="none", label_type="none", description='No labels'),
        ActivityNetConfig(name="captions", label_type="captions", description='ActivityNet Captions'),
        ActivityNetConfig(name="qa", label_type="qa", description='ActivityNet QA'),
        ActivityNetConfig(name='instruct', label_type='instruct', description='VideoInstruct-100K')
    ]
    DEFAULT_CONFIG_NAME = 'captions'

    def _info(self):
        if self.config.label_type == "none":
            features = datasets.Features({
                "video_id": datasets.Value("string"),
                "video": datasets.Value("string"),
            })
        elif self.config.label_type == "captions":
            features = datasets.Features({
                "video_id": datasets.Value("string"),
                "video": datasets.Value("string"),
                "duration": datasets.Value("float"),
                "timestamps": datasets.Sequence(datasets.Sequence(datasets.Value("float"))),
                "sentences": datasets.Sequence(datasets.Value("string")),
            })
        elif self.config.label_type == "qa":
            features = datasets.Features({
                "video_id": datasets.Value("string"),
                "video": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
            })
        elif self.config.label_type == "instruct":
            features = datasets.Features({
                "video_id": datasets.Value("string"),
                "video": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
            })
        else:
            raise ValueError(f"Invalid label type {self.config.label_type}")

        return datasets.DatasetInfo(
            description="ActivityNet dataset",
            features=features,
            supervised_keys=None,
            homepage="http://activity-net.org/",
        )

    def _split_generators(self, dl_manager):
        url = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "data_dir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "val",
                    "data_dir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "data_dir": data_dir,
                },
            ),
        ]

    def _generate_examples(self, split: str, data_dir: str):
        if self.config.label_type == "none":
            video_dir = os.path.join(data_dir, "videos", split)
            for video_file in os.listdir(video_dir):
                video_id = os.path.splitext(video_file)[0]
                yield video_id, {
                    "video_id": video_id,
                    "video": os.path.join(video_dir, video_file),
                }
        elif self.config.label_type == "captions":
            caption_file = os.path.join(data_dir, "captions", f"{split}.json")
            with open(caption_file, "r") as f:
                captions = json.load(f)
            for video_id, data in captions.items():
                yield video_id, {
                    "video_id": video_id,
                    "video": os.path.join(data_dir, "videos", split, f"{video_id}.mp4"),
                    "duration": data["duration"],
                    "timestamps": data["timestamps"],
                    "sentences": data["sentences"],
                }
        elif self.config.label_type == "qa":
            q_file = os.path.join(data_dir, "qa", f"{split}_q.json")
            a_file = os.path.join(data_dir, "qa", f"{split}_a.json")
            with open(q_file, "r") as f:
                questions = json.load(f)
            with open(a_file, "r") as f:
                answers = json.load(f)
            for q, a in zip(questions, answers):
                yield q["question_id"], {
                    "video_id": q["video_name"],
                    "video": os.path.join(data_dir, "videos", split, f"{q['video_name']}.mp4"),
                    "question": q["question"],
                    "answer": a["answer"],
                    "question_id": q["question_id"],
                }
        elif self.config.label_type == "instruct":
            instruct_file = os.path.join(data_dir, "VideoInstruct100K.json")
            with open(instruct_file, "r") as f:
                instructions = json.load(f)
            for idx, inst in enumerate(instructions):
                if inst["video_id"].split("_")[1] in self._get_split_ids(split, data_dir):
                    yield idx, {
                        "video_id": inst["video_id"],
                        "video": os.path.join(data_dir, "videos", split, f"{inst['video_id']}.mp4"),
                        "question": inst["q"],
                        "answer": inst["a"],
                    }

    def _get_split_ids(self, split: str, data_dir: str) -> List[str]:
        split_file = os.path.join(data_dir, "captions", f"{split}_ids.json")
        with open(split_file, "r") as f:
            return json.load(f)
