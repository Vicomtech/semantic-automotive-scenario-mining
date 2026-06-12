import numpy as np
from functools import partial
import math
from os import listdir

def point_in_line_shadow(p1: np.ndarray, p2: np.ndarray, q: np.ndarray) -> bool:
    segment_length = np.linalg.norm(p2 - p1)
    segment_dir = (p2 - p1) / segment_length
    projection = np.dot(q - p1, segment_dir)

    return 0 < projection < segment_length

def get_min_distance_to_segment(p1: np.ndarray, p2: np.ndarray, q: np.ndarray) -> float:
    return np.linalg.norm(np.cross(
        (p2 - p1) / np.linalg.norm(p2 - p1),
        q - p1
    )) if point_in_line_shadow(p1, p2, q) else min(
        np.linalg.norm(q - p1), np.linalg.norm(q - p2)
    )

def get_rectangle_sides(vertices: list) -> list:
    return list(map(
        lambda i: (vertices[i], vertices[(i + 1) % 4]),
        range(4)
    ))

def get_min_distance_point_rectangle(rect_sides: list, q: np.ndarray) -> float:
    return min(map(
        lambda side: get_min_distance_to_segment(*side, q),
        rect_sides
    ))

def get_min_distance_rectangles(r1: list, r2: list) -> float:
    r1 = list(map(np.asarray, r1))
    r2 = list(map(np.asarray, r2))

    min_r1_to_r2 = min(map(
        partial(
            get_min_distance_point_rectangle,
            get_rectangle_sides(r2)
        ),
        r1
    ))

    min_r2_to_r1 = min(map(
        partial(
            get_min_distance_point_rectangle,
            get_rectangle_sides(r1)
        ),
        r2
    ))

    return min(min_r1_to_r2, min_r2_to_r1)

def distance_between_cuboids(c1, c2):
    p = (c1[0], c1[1])
    yaw = c1[5]
    length = c1[6]
    width = c1[7]
    px = []
    py = []
    px_rot = []
    py_rot = []
    p_rot = []
    points = []

    # Add all "vertexes" of the bbox
    points.append((p[0] + width / 2, p[1] + length / 2))
    points.append((p[0] - width / 2, p[1] + length / 2))
    points.append((p[0] + width / 2, p[1] - length / 2))
    points.append((p[0] - width / 2, p[1] - length / 2))

    # angle -> rotation of the bbox, in this case we will use the yaw ->  cuboid = (x,y,z,roll,pitch,yaw,l,w,h)
    s = math.sin(yaw)
    c = math.cos(yaw)

    for point in points:
        px.append(point[0])
        py.append(point[1])
        # Rotation of a point with respect to another, first we move the point to the (0,0) and then we multiply
        x = (point[0] - p[0]) * c - (point[1] - p[1]) * s
        y = (point[0] - p[0]) * s + (point[1] - p[1]) * c
        # These values are the rotated values around 0,0, we need to add the transform of the original point back -> x+p[0], y + p[1]
        px_rot.append(x + p[0])
        py_rot.append(y + p[1])
        p_rot.append((x + p[0], y + p[1]))

    # Ego vehicle transformation

    ego_p = (c2[0], c2[1])
    ego_yaw = c2[5]
    ego_length = c2[6]
    ego_width = c2[7]

    ego_px = []
    ego_py = []
    ego_px_rot = []
    ego_py_rot = []
    ego_p_rot = []
    ego_points = []

    # Add all "vertexes" of the bbox
    ego_points.append((ego_p[0] + ego_width / 2, ego_p[1] + ego_length / 2))
    ego_points.append((ego_p[0] - ego_width / 2, ego_p[1] + ego_length / 2))
    ego_points.append((ego_p[0] + ego_width / 2, ego_p[1] - ego_length / 2))
    ego_points.append((ego_p[0] - ego_width / 2, ego_p[1] - ego_length / 2))

    # angle -> rotation of the bbox, in this case we will use the yaw ->  cuboid = (x,y,z,roll,pitch,yaw,w,l,h)
    s = math.sin(ego_yaw)
    c = math.cos(ego_yaw)

    for point in ego_points:
        ego_px.append(point[0])
        ego_py.append(point[1])
        # Rotation of a point with respect to another, first we move the point to the (0,0) and then we multiply
        ego_x = (point[0] - ego_p[0]) * c - (point[1] - ego_p[1]) * s
        ego_y = (point[0] - ego_p[0]) * s + (point[1] - ego_p[1]) * c
        # These values are the rotated values around 0,0, we need to add the transform of the original point back -> x+p[0], y + p[1]
        ego_px_rot.append(ego_x + ego_p[0])
        ego_py_rot.append(ego_y + ego_p[1])
        ego_p_rot.append((ego_x + ego_p[0], ego_y + ego_p[1]))

    distance = get_min_distance_rectangles(ego_p_rot, p_rot)

    return distance

def regions_between_cuboids(ego_cuboid, cuboid):
    ego_p = (ego_cuboid[0], ego_cuboid[1])
    ego_l = ego_cuboid[6]
    ego_w = ego_cuboid[7]

    obj_p = (cuboid[0], cuboid[1])
    regions = []
    # FrontOf
    if obj_p[0] > ego_p[0]+ego_l/2:
        regions.append("front_of_ego")

    if obj_p[0] < ego_p[0]-ego_l/2:
        regions.append("rear_of_ego")

    if obj_p[1] > ego_p[1]+ego_w/2:
        regions.append("left_of_ego")

    if obj_p[1] < ego_p[1]-ego_w/2:
        regions.append("right_of_ego")

    return regions



def pcdbin_to_pcd(in_dir, out_dir):
    for file in listdir(in_dir):
        scan = np.fromfile(in_dir + file, dtype=np.float32)
        points = scan.reshape((-1, 5))

        out = f'''# .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z intensity ring_index
    SIZE 4 4 4 4 4
    TYPE F F F F F
    COUNT 1 1 1 1 1
    WIDTH {len(points)}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS {len(points)}
    DATA ascii
    '''
        for p in points:
            out += ' '.join([str(x) for x in p]) + '\n'

        with open(out_dir + file.replace('.bin', ''), "w") as f:
            f.write(out)
