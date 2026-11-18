import numpy as np

from .antique import Antique
from .base import ImageFilter
from .utils import to_uint8


class OldPhoto(ImageFilter):
    name = "oldphoto"

    def __init__(self, noise=20, scratches=8):
        self.noise = float(noise)
        self.scratches = int(scratches)

    def apply_filter(self, image):
        img = Antique().apply_filter(image).astype(np.float32)
        h, w = img.shape[:2]

        grain = np.random.normal(0, self.noise, (h, w))
        img = img + grain[:, :, np.newaxis]

        for _ in range(self.scratches):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h // 2)
            length = np.random.randint(h // 10, h // 2)
            img[y:y + length, x] += np.random.randint(60, 120)

        return to_uint8(img)