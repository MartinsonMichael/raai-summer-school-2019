from .utils import *
from .cvat import *


if __name__ == 'main':
    test = DataLoader(
        None,
        '/data/Hack/CDS_Lab/sac_branch/simulator/gym_road_cars/env_data_test/tracks/background_image.jpg',
        '/data/Hack/CDS_Lab/sac_branch/simulator/gym_road_cars/env_data_test/tracks/1_background_segmentation.xml',
    )
    print(test.data)
