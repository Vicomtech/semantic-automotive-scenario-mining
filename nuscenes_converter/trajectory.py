import numpy as np
from shapely.geometry import Polygon
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import geopandas as gpd
import math

class trajectory:
    def __init__(self, x, y, yaw, length, width, vx, vy, ts, yaw_rate=None, TTC=5, sampling_fq=5, frame_start=0):
        self.x = np.array(x)
        self.y = np.array(y)
        self.yaw = np.array(yaw)
        self.length = length
        self.width = width
        self.vx = np.array(vx)
        self.vy = np.array(vy)
        self.ts = np.array(ts)
        if yaw_rate is None:
            yaw_rate = 0
        self.yaw_rate = np.array(yaw_rate)
        self.TTC = TTC
        self.sampling_fq = sampling_fq
        self.trajectory = []
        self.trajectory_polygon = []
        self.frame_start = frame_start
        self.frame_end = frame_start

    def add_state(self, x, y, yaw, vx, vy, ts, yaw_rate=None):
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.yaw = np.append(self.yaw, yaw)
        self.vx = np.append(self.vx, vx)
        self.vy = np.append(self.vy, vy)
        self.ts = np.append(self.ts, ts)
        if yaw_rate is None:
            yaw_rate = (yaw - self.yaw[-2]) / (ts - self.ts[-2])
        self.yaw_rate = np.append(self.yaw_rate, yaw_rate)
        self.frame_end += 1
    def coordinate_rotate(self, x: float, y: float, theta: float):
        # Rotate the coordinates by theta
        x_ = np.cos(theta) * x + np.sin(theta) * y
        y_ = np.sin(theta) * (-x) + np.cos(theta) * y
        return x_, y_

    def normalize_angle(self, angle: float):
        # Normalize the angle to [-pi, pi]
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    def project_angle(self, angle: float):
        # Project the angle to [-pi, pi]
        return np.arctan2(np.sin(angle), np.cos(angle))

    def create_polygon(self, x: float, y: float, yaw_angle: float, length: float, width: float):
        # Create a polygon with the given parameters
        polygon_coordinates = [
            (length / 2, - width / 2),
            (length / 2, width / 2),
            (-length / 2, width / 2),
            (-length / 2, -width / 2)
        ]
        for i in range(4):
            new_x, new_y = self.coordinate_rotate(polygon_coordinates[i][0], polygon_coordinates[i][1], -yaw_angle)
            polygon_coordinates[i] = (new_x + x, new_y + y)
        return Polygon(polygon_coordinates)

    def calculate_trajectories(self):
        if self.x.size == 1:
            return
        for i in range(len(self.x)):
                trajectory_polygons = []
                trajectory = []
                for j in range(1, int(self.TTC * self.sampling_fq)):
                    new_theta_ = self.yaw[i] + j / self.sampling_fq * self.yaw_rate[i]
                    if self.yaw_rate[i] != 0:
                        radius = np.sqrt(self.vx[i] ** 2 + self.vy[i] ** 2) / self.yaw_rate[i]
                        new_x_ = self.x[i] + radius * (np.sin(new_theta_) - np.sin(self.yaw[i]))
                        new_y_ = self.y[i] + radius * (- np.cos(new_theta_) + np.cos(self.yaw[i]))
                    else:
                        # for the case of yaw_rate = 0, constant velocity model is used
                        new_x_ = self.x[i] + j / self.sampling_fq * self.vx[i]
                        new_y_ = self.y[i] + j / self.sampling_fq * self.vy[i]
                    # project new_theta_ to (-pi,pi)
                    new_theta_ = self.project_angle(new_theta_)
                    trajectory_polygons.append(self.create_polygon(new_x_, new_y_, new_theta_, self.length, self.width))
                    trajectory.append([new_x_, new_y_, new_theta_])
                self.trajectory_polygon.append(trajectory_polygons)
                self.trajectory.append(trajectory)

    def plot_trajectory(self, index=None):
        mod_size = len(self.trajectory_polygon[0])
        all_polygon = []
        if mod_size == 0:
            print("trajectories not calculated, use self.calculate_trajectories() first")
            return

        if index == None:
            for poly_list in self.trajectory_polygon:
                for poly in poly_list:
                    all_polygon.append(poly)
        else:
            all_polygon = self.trajectory_polygon[index]

        p = gpd.GeoSeries(all_polygon)
        colors = []
        alpha = []
        step = 0.3 / mod_size
        for i in range(len(all_polygon)):
            if i % mod_size == 0:
                colors.append('red')
                alpha.append(1)
            else:
                colors.append('blue')
                alpha.append(0.4 - (i % mod_size) * step)

        p.plot(color=colors, alpha=alpha)
        plt.gca().set_aspect('equal')
        plt.show(block=True)


    def getDifference(self, b1, b2):
        r = (b2 - b1) % math.pi
        # Python modulus has same sign as divisor, which is positive here,
        # so no need to consider negative case
        if r >= math.pi/2:
            r -= math.pi
        return r

    def get_trajectory_polygons_at_frame(self, frame=0):
        if frame >= self.frame_start and frame <= self.frame_end and True:
            return self.trajectory_polygon[frame - self.frame_start]
        return None

    def calculate_ttc(self, other_traj, draw=False):
        frame_ttc = {}
        collision = False
        for frame_num in range(other_traj.frame_start, other_traj.frame_end):
            ego_polys = self.get_trajectory_polygons_at_frame(frame_num)
            obj_polys = other_traj.get_trajectory_polygons_at_frame(frame_num)
            for i in range(len(ego_polys)):
                if ego_polys[i].intersects(obj_polys[i]):
                    ts = (i + 1) / other_traj.sampling_fq
                    frame_ttc[frame_num] = ts
                    collision = True
                    if draw:
                        self.plot_trajs(ego_polys, obj_polys, i)

        return collision, frame_ttc

    def plot_trajs(self, poly_list1, poly_list2, match):
        merge_trajs = []
        colors = []
        alpha = []
        poly_len = len(poly_list1)
        for i in range(poly_len):
            if i == match:
                colors.append('red')
                alpha.append(1)
                colors.append('blue')
                alpha.append(1)
            elif i == 0:
                colors.append('orange')
                alpha.append(1)
                colors.append('green')
                alpha.append(1)
            else:
                colors.append('orange')
                alpha.append(0.2)
                colors.append('green')
                alpha.append(0.2)

            merge_trajs.append(poly_list1[i])
            merge_trajs.append(poly_list2[i])

        p = gpd.GeoSeries(merge_trajs)
        p.plot(color=colors, alpha=alpha)
        plt.gca().set_aspect('equal')
        plt.show(block=True)