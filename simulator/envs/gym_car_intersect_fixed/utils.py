from typing import List, NamedTuple, Type, Any, Optional, Tuple, Union, Dict
import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.measure import label, regionprops

from envs.gym_car_intersect_fixed.cvat import CvatDataset


class CarImage(NamedTuple):
    image: np.ndarray
    mask: np.ndarray
    real_image: np.ndarray
    real_size: np.array
    car_image_center_displacement: np.ndarray
    size: np.array
    center: np.array


class DataSupporter:
    """
    Class with bunch of static function for processing, loading ect of tracks, background image, cars image.
    Also all convertations from normal world XY coordinate system to image -YX system should be
       provided via this class functions.

    """
    def __init__(self, cars_path, cvat_path, image_path):
        self._background_image = cv2.imread(image_path)

        # in XY coordinates, not in a IMAGE coordinates
        self._image_size = np.array([self._background_image.shape[1], self._background_image.shape[0]])
        # just two numbers of field in pyBox2D coordinate system
        self._playfield_size = np.array([35 * self._background_image.shape[1] / self._background_image.shape[0], 35])
        # technical field
        self._data = CvatDataset()
        self._data.load(cvat_path)
        # list of car images
        self._cars: List[CarImage] = []
        self._load_car_images(cars_path)
        # list of tracks
        self._tracks: List[Dict[str, Union[np.array, Polygon]]] = []
        self._extract_tracks()

    @property
    def track_count(self):
        return len(self._tracks)

    @property
    def car_image_count(self):
        return len(self._cars)

    @property
    def playfield_size(self) -> np.array:
        return self._playfield_size

    def set_playfield_size(self, size: np.array):
        if size.shape != (2,):
            raise ValueError
        self._playfield_size = size

    @staticmethod
    def convert_XY2YX(points: np.array):
        if len(points.shape) == 2:
            return np.array([points[:, 1], points[:, 0]])
        if points.shape == (2, ):
            return np.array([points[1], points[0]])
        raise ValueError

    @staticmethod
    def do_with_points(track_obj, func):
        """
        Perform given functions under 'line' array.
        :param track_obj: dict with two keys:
            'polygon' - shapely.geometry.Polygon object
            'line' - np.array with line points coordinates
        :param func: function to be permorm under track_obj['line']
        :return: track_obj
        """
        track_obj['line'] = func(track_obj['line'])
        return track_obj

    def convertIMG2PLAY(self, points: Union[np.array, Tuple[float, float]]) -> np.array:
        """
        Convert points from IMG pixel coordinate to pyBox2D coordinate.
        NOTE! This function doesn't flip Y to -Y, just scale coordinates.
        :param points: np.array
        :return: np.array
        """
        points = np.array(points)
        if len(points.shape) == 1:
            return self._convertXY_IMG2PLAY(points)
        else:
            return np.array([self._convertXY_IMG2PLAY(coords) for coords in points])

    def convertPLAY2IMG(self, points: Union[np.array, Tuple[float, float]]) -> np.array:
        """
        Convert points from pyBox2D coordinate to IMG pixel coordinate.
        NOTE! This function doesn't flip Y to -Y, just scale coordinates.
        :param points: np.array
        :return: np.array
        """
        points = np.array(points)
        if len(points.shape) == 1:
            return self._convertXY_PLAY2IMG(points)
        else:
            return np.array([self._convertXY_PLAY2IMG(coords) for coords in points])

    def _convertXY_IMG2PLAY(self, coords: np.array):
        """
        Technical function for IMG to pyBox2D coordinates convertation.
        """
        if coords.shape != (2, ):
            raise ValueError
        return coords * self._playfield_size / self._image_size

    def _convertXY_PLAY2IMG(self, coords: np.array):
        """
        Technical function for pyBox2D to IMG coordinates convertation.
        """
        if coords.shape != (2, ):
            raise ValueError
        return coords * self._image_size / self._playfield_size

    @property
    def data(self):
        return self._data

    def get_background(self):
        return self._background_image.copy()

    def _load_car_images(self, cars_path):
        """
        Technical function for car image loading.
        """
        import os
        for folder in os.listdir(cars_path):
            try:
                mask = cv2.imread(os.path.join(cars_path, folder, 'mask.bmp'))
                real_image = cv2.imread(os.path.join(cars_path, folder, 'image.jpg'))

                label_image = label(mask[:, :, 0])
                region = regionprops(label_image)[0]
                min_y, min_x, max_y, max_x = region.bbox

                region_size_y = (max_y - min_y)
                region_size_x = (max_x - min_x)

                car = CarImage(
                    mask=mask,
                    real_image=real_image,
                    real_size=np.array([real_image.shape[1], real_image.shape[0]]),
                    center=np.array([real_image.shape[1], real_image.shape[0]]),
                    car_image_center_displacement=region.centroid - np.array([real_image.shape[0], real_image.shape[1]]) / 2,
                    image=cv2.bitwise_and(real_image, mask),
                    size=np.array([region_size_x, region_size_y]),
                )

                if car.size[0] < 10 or car.size[1] < 10:
                    raise ValueError()

                self._cars.append(car)
            except:
                pass
                # print(f'error while parsing car image source: {os.path.join(cars_path, folder)}')

    def _extract_tracks(self):
        """
        Technical function for track loading.
        """
        track_lines = {}
        for item in self._data.get_polylines(0):
            if item['label'] == 'track_line':
                track_lines[item['attributes']['index']] = np.array(item['points'])
        track_polygons = {}
        for item in self._data.get_polygons(0):
            if item['label'] == 'track':
                track_polygons[item['attributes']['index']] = np.array(item['points'])
        for index, track_line in track_lines.items():
            if index not in track_polygons.keys():
                print(f'skip track index index {index}')
                continue
            self._tracks.append({
                'polygon': Polygon(self.convertIMG2PLAY(track_polygons[index])),
                'line': track_line,
            })

    @staticmethod
    def _dist(pointA, pointB) -> float:
        """
        Just Euclidean distance.
        """
        return np.sqrt(np.sum((pointA - pointB)**2))

    @staticmethod
    def _expand_track(track_obj, max_dist: float = 10.0) -> Dict[str, Any]:
        """
        Insert point in existing polyline, while dist between point more then max_dist.
        As a result track_obj['line'] will contain more points.
        """
        track = np.array(track_obj['line'])
        first_point = track[0].copy()
        expanded_track = [first_point.copy()]

        for index in range(1, len(track)):
            next_point = track[index]
            vector = next_point - first_point
            vector_len = np.sqrt(np.sum((vector ** 2)))
            if vector_len < 1e-8:
                print(vector)
                raise ValueError('oops, something went wrong, ask Michael Martinson, tg @MichaelMD')
            vector = vector / vector_len * max_dist

            if index == len(track) - 1 or DataSupporter._dist(next_point, first_point) > 1.2 * max_dist:
                expanded_point = first_point + vector
                while DataSupporter._dist(next_point, expanded_point) > 1.3 * max_dist:
                    expanded_track.append(expanded_point.copy())
                    expanded_point += vector
                expanded_track.append(expanded_point.copy())
            expanded_track.append(next_point.copy())
            first_point = next_point.copy()

        expanded_track.append(track[-1])
        return {
            'polygon': track_obj['polygon'],
            'line': np.array(expanded_track),
        }

    def peek_car_image(self, index: Optional[int] = None):
        """
        Return random car image.
        :param index: integer, if provided function return index'th car image
        :return: car image, named tuple
        """
        if index is None:
            index = np.random.choice(np.arange(len(self._cars)))
        return self._cars[index]

    def peek_track(self, expand_points: Optional[float] = 50, index: Optional[int] = None):
        """
        Return random track object.
        :param expand_points: if provided increase number of points in 'line' part of track object
        :param index: if provided, function return index'th track.
        :return:
        """
        if index is None:
            index = np.random.choice(np.arange(len(self._tracks)))
        if expand_points is not None:
            return DataSupporter._expand_track(self._tracks[index], expand_points)
        return self._tracks[index]

    @staticmethod
    def dist(pointA: np.array, pointB: np.array) -> float:
        """
        Just another Euclidean distance.
        """
        if pointA.shape != (2, ) or pointB.shape != (2, ):
            raise ValueError('incorrect points shape')
        return np.sqrt(np.sum((pointA - pointB)**2))

    @staticmethod
    def get_track_angle(track_obj: np.array, index=0) -> float:
        """
        Return angle between OX and track_obj['line'][index] -> track_obj['line'][index + 1] points
        """
        track = track_obj
        if isinstance(track_obj, dict):
            track = track_obj['line']
        if index == len(track):
            index -= 1
        angle = DataSupporter.angle_by_2_points(
            track[index],
            track[index + 1]
        )
        return angle

    @staticmethod
    def get_track_initial_position(track: Union[np.array, Dict[str, Any]]) -> np.array:
        """
        Just return starting position for track object.
        """
        if isinstance(track, dict):
            return track['line'][0]
        return track[0]

    @staticmethod
    def angle_by_2_points(
            pointA: np.array,
            pointB: np.array,
    ) -> float:
        return DataSupporter.angle_by_3_points(
            np.array(pointA) + np.array([1.0, 0.0]),
            np.array(pointA),
            np.array(pointB),
        )

    @staticmethod
    def angle_by_3_points(
            pointA: np.array,
            pointB: np.array,
            pointC: np.array) -> float:
        """
        compute angle
        :param pointA: np.array of shape (2, )
        :param pointB: np.array of shape (2, )
        :param pointC: np.array of shape (2, )
        :return: angle in radians between AB and BC
        """
        if pointA.shape != (2,) or pointB.shape != (2,) or pointC.shape != (2,):
            raise ValueError('incorrect points shape')

        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arctan2(np.cross(v1_u, v2_u), np.dot(v1_u, v2_u))

        return angle_between(pointA - pointB, pointC - pointB)
