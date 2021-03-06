from typing import List

import gdown
import numpy as np

from .retinaface import RetinaFace


class FaceDetector(object):
    def __init__(self, weights: str = "https://drive.google.com/uc?id=11Ksdq9sQLp9E6WFuGzOZ-3jxTpx1nFCI",
                 threshold: float = 0.8,
                 gpuid: int = -1, padding: float = 0.2, flip: bool = False, scales=None, output="crop"):

        if scales is None:
            scales = [1024, 1980]
        self.threshold = threshold
        self.padding = padding
        self.flip = flip
        self.scales = scales
        self.target_size = scales[0]
        self.max_size = scales[1]
        self.output = output

        if 'http' in weights:
            weights = self.download_weights(weights)
        self.detector = RetinaFace(weights, 0, gpuid, 'net3')

    @staticmethod
    def download_weights(url):
        import zipfile
        gdown.download(url, 'weights.zip', quiet=False)
        with zipfile.ZipFile('weights.zip', 'r') as zip_ref:
            zip_ref.extractall("./weights")
        return "./weights/R50"

    def get_faces(self, image: np.ndarray) -> List[np.ndarray]:
        im_shape = image.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.target_size) / float(im_size_min)

        if np.round(im_scale * im_size_max) > self.max_size:
            im_scale = float(self.max_size) / float(im_size_max)

        scales = [im_scale]
        faces, landmarks = self.detector.detect(image, self.threshold, scales=scales, do_flip=self.flip)

        outputs = []
        if faces is not None:
            for i in range(faces.shape[0]):
                b = faces[i].astype(np.int)
                padding_height = int((b[2] - b[0]) * self.padding)
                padding_width = int((b[3] - b[1]) * self.padding)
                b[1] = max(int(b[1] - padding_height), 0)
                b[3] = int(b[3]) + padding_height
                b[0] = max(int(b[0]) - padding_width, 0)
                b[2] = int(b[2]) + padding_width

                if self.output == 'crop':
                    outputs.append(image[b[1]:b[3], b[0]:b[2], :])
                elif self.output == 'box':
                    outputs.append(b)
        return outputs