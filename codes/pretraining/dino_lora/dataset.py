# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import io
from typing import Any
import h5py
from typing import Callable, Optional, Tuple
from PIL import Image
import json
from torchvision.datasets import VisionDataset


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = io.BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target


class ExtendedVisionDataset(VisionDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        raise NotImplementedError


class PathologyDataset(ExtendedVisionDataset):
    def __init__(self, root: str = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, None, transform, target_transform)
        self.images = json.load(open(root, "r", encoding="utf-8"))
        self.transformers = transform
        self.file_handle = open("./failed_path.txt", "a")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, h5 = self.images[index]
        with h5py.File(h5, "r") as hf:
            try:
                img_bytes = hf[image][:].tobytes()
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                img = Image.new("RGB", (224, 224), (255, 255, 255))
                self.file_handle.write(f"Error loading image {image} from {h5}: {e}\n")
            if self.transformers is not None:
                img = self.transformers(img)
        return img, 0
