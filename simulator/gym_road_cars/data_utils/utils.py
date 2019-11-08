from typing import List, NamedTuple, Type, Any, Optional
import cv2
import numpy as np
from skimage.measure import label, regionprops

from .cvat import CvatDataset


class CarImage(NamedTuple):
    image: Type[np.ndarray]
    size: List[float]
    mask: Type[np.ndarray]
    real_image: Type[np.ndarray]


class DataSupporter:
    def __init__(self, cars_path, cvat_path, image_path):
        self._background_image = cv2.imread(image_path)
        self._data = CvatDataset()
        self._data.load(cvat_path)
        self._cars: List[CarImage] = []
        self._load_car_images(cars_path)
        self._tracks: List[Any] = []

    @property
    def data(self):
        return self._data

    def _load_car_images(self, cars_path):
        import os
        for folder in os.listdir(cars_path):
            try:
                mask = cv2.imread(os.path.join(cars_path, folder, 'mask.bmp'))
                real_image = cv2.imread(os.path.join(cars_path, folder, 'image.jpg'))

                label_image = label(mask[:, :, 0])
                region = regionprops(label_image)[0]
                minr, minc, maxr, maxc = region.bbox

                region_width = (maxc - minc)
                region_height = (maxr - minr)

                car = CarImage(
                    mask=mask,
                    real_image=real_image,
                    image=None,
                    size=[region_height, region_width],
                )
                self._cars.append(car)
            except:
                print(f'error while parsing car image source: {os.path.join(cars_path, folder)}')

    def _extract_tracks(self):
        for item in self._data.get_polylines(0):
            if item['label'] == 'car_track':
                self._tracks.append(np.array(item['points']))

    @staticmethod
    def _dist(pointA, pointB) -> float:
        return np.sqrt(np.sum((pointA - pointB)**2))

    @staticmethod
    def _expand_track(track: np.ndarray, max_dist: float = 10.0) -> np.ndarray:
        """
        Insert point in existing polyline, while dist between point more then max_dist
        """
        track = np.array(track)
        first_point = track[0].copy()
        expanded_track = [first_point.copy()]

        for next_point in track[1:]:
            vector = next_point - first_point
            vector_len = np.sqrt(np.sum((vector ** 2)))
            vector = vector / vector_len * max_dist

            expanded_point = first_point + vector
            while DataSupporter._dist(next_point, expanded_point) > max_dist:
                expanded_track.append(expanded_point.copy())
                expanded_point += vector

            expanded_track.append(expanded_point.copy())
            expanded_track.append(next_point.copy())
            first_point = next_point.copy()

        return np.array(expanded_track)

    def peek_car_image(self):
        index = np.random.choice(np.arange(len(self._cars)))
        return self._cars[index]

    def peek_track(self, expand_points: Optional[float] = 50):
        index = np.random.choice(np.arange(len(self._tracks)))
        if expand_points is not None:
            return DataSupporter._expand_track(self._tracks[index], expand_points)
        return self._tracks[index]