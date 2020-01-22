from typing import List

import gdown
import numpy as np

from .retinaface import RetinaFace


class FaceDetector(object):
    def __init__(self, weights: str = "https://drive.google.com/uc?id=11Ksdq9sQLp9E6WFuGzOZ-3jxTpx1nFCI",
                 threshold: float = 0.8,
                 gpuid: int = -1, padding: float = 0.2, flip: bool = False, scales=None):

        if scales is None:
            scales = [1.0]
        self.threshold = threshold
        self.padding = padding
        self.flip = flip
        self.scales = scales

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
        faces, landmarks = self.detector.detect(image, self.threshold, scales=self.scales, do_flip=self.flip)
        crops = []
        if faces is not None:
            for i in range(faces.shape[0]):
                b = faces[i].astype(np.int)
                padding_height = int((b[2] - b[0]) * self.padding)
                padding_width = int((b[3] - b[1]) * self.padding)

                crops.append(image[max(int(b[1] - padding_height), 0):int(b[3]) + padding_height,
                             max(int(b[0]) - padding_width, 0):int(b[2]) + padding_width, :])
        return crops
