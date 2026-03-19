WINDOW_RES = (700, 700)
WINDOW_TITLE = "Experiment 2: Rotation and Transformation"

EYE_POS = (0.0, 0.0, 5.0)
EYE_FOV = 45.0
ASPECT_RATIO = WINDOW_RES[0] / WINDOW_RES[1]
Z_NEAR = 0.1
Z_FAR = 50.0

ROTATE_STEP_DEG = 10.0

TRIANGLE_VERTICES = (
    (2.0, 0.0, -2.0),
    (0.0, 2.0, -2.0),
    (-2.0, 0.0, -2.0),
)

TRIANGLE_EDGES = (
    (0, 1),
    (1, 2),
    (2, 0),
)

EDGE_COLORS = (
    0xFF4D4F,
    0x40A9FF,
    0x73D13D,
)

LINE_RADIUS = 2.0
BACKGROUND_COLOR = 0x111111
