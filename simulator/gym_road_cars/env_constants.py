# constants fot car environment

REWARD_TILES = 1
REWARD_COLLISION = -10
REWARD_PENALTY = -10
REWARD_FINISH = 100
REWARD_OUT = -10
REWARD_STUCK = -15
REWARD_VELOCITY = -0
REWARD_TIME = 0


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 600  # Changes: 400
WINDOW_W = 600  # Changes: 1200
WINDOW_H = 600  # Changes: 1000

SCALE = 1  # Track scale
PLAYFIELD = 40 / SCALE  # Game over boundary # Changes: 600
FPS = 50
ZOOM = 7.5  # Changes: 2.7        # Camera zoom
ZOOM_FOLLOW = False  # Set to False for fixed view (don't use zoom)

ROAD_WIDTH = 8 / SCALE
SIDE_WALK = 4 / SCALE

ROAD_COLOR = [0.44, 0.44, 0.44]

# Creating all possible tragectories:
SMALL_TURN = ROAD_WIDTH * 0.5
BIG_TURN = ROAD_WIDTH * 1.5
START_1, START_2 = (-ROAD_WIDTH, ROAD_WIDTH), (-ROAD_WIDTH, -ROAD_WIDTH)
START_3, START_4 = (ROAD_WIDTH, -ROAD_WIDTH), (ROAD_WIDTH, ROAD_WIDTH)
OUT_DIST = 10  # how far from the view screen to restart new car
TARGET_2, TARGET_4 = (-PLAYFIELD - OUT_DIST, ROAD_WIDTH / 2), (-ROAD_WIDTH / 2, -PLAYFIELD - OUT_DIST)
TARGET_6, TARGET_8 = (PLAYFIELD + OUT_DIST, -ROAD_WIDTH / 2), (ROAD_WIDTH / 2, PLAYFIELD + OUT_DIST)
PATH = {
    '34': [(START_2[0] + math.cos(rad) * SMALL_TURN, START_2[1] + math.sin(rad) * SMALL_TURN)
           for rad in np.linspace(np.pi / 2, 0, 10)] + [TARGET_4],
    '36': [TARGET_6],
    '38': [(START_1[0] + math.cos(rad) * BIG_TURN, START_1[1] + math.sin(rad) * BIG_TURN)
           for rad in np.linspace(-np.pi / 2, 0, 10)] + [TARGET_8],

    '56': [(START_3[0] + math.cos(rad) * SMALL_TURN, START_3[1] + math.sin(rad) * SMALL_TURN)
           for rad in np.linspace(np.pi, np.pi / 2, 10)] + [TARGET_6],
    '58': [TARGET_8],
    '52': [(START_2[0] + math.cos(rad) * BIG_TURN, START_2[1] + math.sin(rad) * BIG_TURN)
           for rad in np.linspace(0, np.pi / 2, 10)] + [TARGET_2],

    '78': [(START_4[0] + math.cos(rad) * SMALL_TURN, START_4[1] + math.sin(rad) * SMALL_TURN)
           for rad in np.linspace(-np.pi / 2, -np.pi, 10)] + [TARGET_8],
    '72': [TARGET_2],
    '74': [(START_3[0] + math.cos(rad) * BIG_TURN, START_3[1] + math.sin(rad) * BIG_TURN)
           for rad in np.linspace(np.pi / 2, np.pi, 10)] + [TARGET_4],

    '92': [(START_1[0] + math.cos(rad) * SMALL_TURN, START_1[1] + math.sin(rad) * SMALL_TURN)
           for rad in np.linspace(0, -np.pi / 2, 10)] + [TARGET_2],
    '94': [TARGET_4],
    '96': [(START_4[0] + math.cos(rad) * BIG_TURN, START_4[1] + math.sin(rad) * BIG_TURN)
           for rad in np.linspace(-np.pi, -np.pi / 2, 10)] + [TARGET_6],
}

ALL_SECTIONS = set(list(PATH.keys()))
INTERSECT = {
    '34': {'94', '74'},
    '36': ALL_SECTIONS - {'72', '78', '92'},
    '38': ALL_SECTIONS - {'56', '92'},

    '56': {'36', '96'},
    '58': ALL_SECTIONS - {'34', '94', '92'},
    '52': ALL_SECTIONS - {'34', '78'},

    '78': {'58', '38'},
    '72': ALL_SECTIONS - {'36', '34', '56'},
    '74': ALL_SECTIONS - {'56', '92'},

    '92': {'72', '52'},
    '94': ALL_SECTIONS - {'58', '56', '78'},
    '96': ALL_SECTIONS - {'78', '34'},
}

PATH_cKDTree = dict()
for key, value in PATH.items():
    PATH_cKDTree[key] = cKDTree(value)

# Road mark crossings:
CROSS_WIDTH = 4 / SCALE
template_v = np.array([(0.8, 0), (0.2, 0), (0.2, CROSS_WIDTH), (0.8, CROSS_WIDTH)])
template_h = np.array([(CROSS_WIDTH, 0.2), (0, 0.2), (0, 0.8), (CROSS_WIDTH, 0.8)])

eps = 0 / SCALE
crossing_w = [-template_h + np.array([-ROAD_WIDTH - eps, y]) for y in np.arange(-ROAD_WIDTH + 1, ROAD_WIDTH + 1)]
crossing_e = [template_h + np.array([ROAD_WIDTH + eps, y]) for y in np.arange(-ROAD_WIDTH, ROAD_WIDTH - 0)]
crossing_n = [template_v + np.array([y, ROAD_WIDTH + eps]) for y in np.arange(-ROAD_WIDTH, ROAD_WIDTH - 0)]
crossing_s = [-template_v + np.array([y, -ROAD_WIDTH - eps]) for y in np.arange(-ROAD_WIDTH + 1, ROAD_WIDTH + 1)]
crossings = [crossing_w, crossing_e, crossing_n, crossing_s]

cross_line_w = [(-PLAYFIELD, 0), (-ROAD_WIDTH - CROSS_WIDTH - eps * 2, 0)]
cross_line_e = [(ROAD_WIDTH + CROSS_WIDTH + eps * 2, 0), (PLAYFIELD, 0)]
cross_line_n = [(0, PLAYFIELD), (0, ROAD_WIDTH + CROSS_WIDTH + eps * 2)]
cross_line_s = [(0, -PLAYFIELD), (0, -ROAD_WIDTH - CROSS_WIDTH - eps * 2)]

cross_line = [cross_line_w, cross_line_e, cross_line_n, cross_line_s]