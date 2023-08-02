from torchvision import get_image_backend
import json
from pathlib import Path

import torch
import torch.utils.data as data
from PIL import Image

class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader
    # Determine how images are loaded, ImageLoaderPIL() is defaulted

    # 当一个类型实现了特殊方法__call__,该类的实例就变成了可调用的类型
    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            if image_path.exists():
                video.append(self.image_loader(image_path))

        return video
