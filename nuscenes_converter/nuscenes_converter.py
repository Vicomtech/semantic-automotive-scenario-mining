# install c++ build tools if its gonna be used in Windows: https://www.youtube.com/watch?v=rcI1_e38BWs
# conda install git
# conda create --name nuscenes python=3.7
# pip install nuscenes-devkit
# conda install -n nuscenes nb_conda_kernels
# If it still fails ensure that c++ build tools are properly installed or try the following commands
# pip install cython
# pip install numpy
# pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
# jupyter notebook .

import vcd.core as core
import vcd.types as types
import vcd.utils as utils
import vcd.scl as scl
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
import math
from pyquaternion import Quaternion
import json
from shapely.geometry import Polygon, Point
import itertools
from tqdm import tqdm
'''
Author: Mikel García Fonseca

Parser from nuScenes Dataset format to VCD (Video Content Descriptor)
The nuScenes format has 13 main building blocks:
    1. scene - 20 second snippet of a car's journey.
    2. sample - An annotated snapshot of a scene at a particular timestamp.
    3. sample_data - Data collected from a particular sensor.
    4. sample_annotation - An annotated instance of an object within our interest.
    5. instance - Enumeration of all object instance we observed.
    6. category - Taxonomy of object categories (e.g. vehicle, human).
    7. attribute - Property of an instance that can change while the category remains the same.
    8. visibility - Fraction of pixels visible in all the images collected from 6 different cameras..
    9. sensor - A specific sensor type.
    10. calibrated sensor - Definition of a particular sensor as calibrated on a particular vehicle.
    11. ego_pose - Ego vehicle poses at a particular timestamp.
    12. log - Log information from which the data was extracted.
    13. map - Map data that is stored as binary semantic masks from a top-down view. 

    More information of this format at: https://www.nuscenes.org/data-format

We parse all the data stored in this files into a single VCD file, one file for each Scene.

The folder with the nuScenes data should have the following structure

+-- nuscenes
|   +-- maps (optional)
|   +-- samples (optional)
|   +-- sweeps (optional)
|   +-- v1.0-mini
|   |    +-- attribute.json
|   |    +-- calibrated_sensor.json
|   |    +-- category.json
|   |    +-- ego_pose.json
|   |    +-- instance.json
|   |    +-- log.json
|   |    +-- map.json
|   |    +-- sample.json
|   |    +-- sample_annotation.json
|   |    +-- sample_data.json
|   |    +-- scene.json
|   |    +-- sensor.json
|   |    +-- visibility.json

The 13 json files from the v1.x-... folder is from where we retrieve the data to parse as VCD.
'''
# NuScenes does not provide ego vehicle dimensions, a Renault Zoe is used, the values are obtained from wikipedia.
EGO_LENGTH = 4.084
EGO_HEIGHT = 1.562
EGO_WIDTH = 1.730
EGO_REAR_OVERHANG = 0.657


class Converter:
    def __init__(self, openlabel_path, dataset_path='v1.0-mini', nuscenes_folder='v1.0-mini'):
        self.path = openlabel_path
        self.nusc = NuScenes(version=nuscenes_folder, dataroot=dataset_path, verbose=True)
        self.scenes = self.nusc.scene
        self.samples = self.nusc.sample
        self.sample_data = self.nusc.sample_data
        self.sample_annotation = self.nusc.sample_annotation
        self.instance = self.nusc.instance
        self.category = self.nusc.category
        self.attribute = self.nusc.attribute
        self.visibility = self.nusc.visibility
        self.sensors = self.nusc.sensor
        self.calibrated_sensor = self.nusc.calibrated_sensor
        self.ego_pose = self.nusc.ego_pose
        self.log = self.nusc.log
        self.map = self.nusc.map
        self.scene_n = 1
        self.total_scenes = len(self.scenes)
        self.vcd = core.OpenLABEL()
        self.error_log = []

    def generate_vcds(self, can_expansion_path=None, map_expansion_path=None):
        for my_scene in self.scenes:
            self.vcd = core.OpenLABEL()
            self.add_coordinate_systems()
            print("Converting Scene: ", str(self.scene_n) + "/" + str(self.total_scenes))

            # Add sensors to VCD
            self.add_sensors()

            # Add scene metadata, log info and map info to VCD
            self.add_metadata(my_scene)

            # Create dictionary to map between instance_token and VCD object_uids
            objects_uids = self.add_objects(my_scene)

            # Get first sample of the scene
            token_sample = my_scene['first_sample_token']
            current_sample = self.nusc.get('sample', token_sample)

            # Iterate over samples
            for frame_num in tqdm(range(0, my_scene['nbr_samples'])):
                # Check sensors for the given sample
                for sensor in current_sample['data']:
                    # Get sample_data for each sensor
                    data = self.nusc.get('sample_data', current_sample['data'][sensor])
                    # If the sensor has not been added to OpenLABEL, add it.
                    if not self.vcd.has_coordinate_system(data['channel']):
                        calibrated_sensor = self.nusc.get('calibrated_sensor', data['calibrated_sensor_token'])
                        self.add_calibrated_sensor(calibrated_sensor, data)
                    # Add stream properties
                    self.vcd.add_stream_properties(stream_name=data['channel'],
                                              stream_sync=types.StreamSync(frame_vcd=frame_num),
                                              properties={"uri": data['filename']})

                    # The ego_pose associated to the lidar_top timestamp is used as the ego_reference transform
                    if data['channel'] == 'LIDAR_TOP':
                        self.add_ego_transform(data, frame_num, objects_uids["ego_vehicle"])

                # Iterate over all the annotation in the keyframe
                for annotation_token in current_sample['anns']:
                    self.add_annotation(annotation_token, objects_uids, frame_num)

                # Add frame_properties
                self.vcd.add_frame_properties(frame_num, timestamp=str(current_sample['timestamp']),
                                         properties={"sample_token": current_sample['token']})

                if current_sample['next'] != '':
                    current_sample = self.nusc.get('sample', current_sample['next'])

            # Write vcd files
            if can_expansion_path is not None:
                try:
                    self.add_can_expansion(can_expansion_path)
                except:
                    print("Can expansion could not be added")
                    self.error_log.append("CAN not existing for: " + my_scene['token'])

            if map_expansion_path is not None:
                self.add_map_expansion(map_expansion_path)

            print("Saving Scene: ", self.path+'vcd_nuscenes_' + my_scene['token'] + '.json')

            nusc_scene_n = int(my_scene['name'].split("-")[-1])
            self.vcd.save(self.path+'annotation_task1_' + str(nusc_scene_n) + '.json', True)
            self.scene_n += 1
        print(self.error_log)

    def add_coordinate_systems(self):
        self.vcd.add_coordinate_system("odom", cs_type=types.CoordinateSystemType.scene_cs)
        self.vcd.add_coordinate_system(name="vehicle-iso8855",
                                  cs_type=types.CoordinateSystemType.local_cs,
                                  parent_name="odom")

        C = np.array([0, 0, 0]).reshape(3, 1)
        R = utils.euler2R([0, 0, 0])
        scs_wrt_lcs = utils.create_pose(R, C)
        self.vcd.add_coordinate_system(name="vehicle-iso8855",
                                  cs_type=types.CoordinateSystemType.local_cs,
                                  parent_name="odom", pose_wrt_parent=types.PoseData(val=scs_wrt_lcs.flatten().tolist(),
                                                                                     t_type=types.TransformDataType.matrix_4x4))

    def add_sensors(self):
        for sensor in self.sensors:
            if sensor['modality'] == 'camera':
                self.vcd.add_stream(stream_name=sensor['channel'], uri='', description=sensor['token'],
                               stream_type=core.StreamType.camera)
            elif sensor['modality'] == 'lidar':
                self.vcd.add_stream(stream_name=sensor['channel'], uri='', description=sensor['token'],
                               stream_type=core.StreamType.lidar)
            else:
                self.vcd.add_stream(stream_name=sensor['channel'], uri='', description=sensor['token'],
                               stream_type=core.StreamType.other)
        return self.vcd

    def add_metadata(self, scene):
        metadata = {'scene_token': scene['token'], 'scene_name': scene['name'],
                    'scene_description': scene['description']}
        self.vcd.add_metadata_properties(metadata)
        log = self.nusc.get('log', scene['log_token'])
        log_info = {'log_token': log['token'], 'log_file': log['logfile'], 'vehicle': log['vehicle'],
                    'date': log['date_captured'], 'location': log['location']}
        self.vcd.add_metadata_properties(log_info)
        # Add map metadata to VCD
        map = self.nusc.get('map', log['map_token'])
        map_info = {'map_token': map['token'], 'map_category': map['category'], 'map_filename': map['filename']}
        self.vcd.add_metadata_properties(map_info)

    def add_objects(self, scene):
        objects_uids = {}
        objects_uids["ego_vehicle"] = self.vcd.add_object(name="ego_vehicle", semantic_type="car")
        for instance in self.instance:
            first_annotation = self.nusc.get('sample_annotation', instance['first_annotation_token'])
            last_annotation = self.nusc.get('sample_annotation', instance['last_annotation_token'])
            category_token = instance['category_token']
            category = self.nusc.get('category', category_token)
            sample = self.nusc.get('sample', first_annotation['sample_token'])
            my_scene = self.nusc.get('scene', sample['scene_token'])

            if my_scene['token'] == scene['token']:
                obj_type = category["name"].split(".")[-1]
                objects_uids[instance['token']] = self.vcd.add_object(name=instance['token'], semantic_type=obj_type)
        return objects_uids

    def add_calibrated_sensor(self, calibrated_sensor, sample_data):
        channel = sample_data['channel']
        C = np.array(calibrated_sensor['translation']).reshape(3, 1)
        quaternion = calibrated_sensor['rotation']
        R = utils.q2R(quaternion[1], quaternion[2], quaternion[3], quaternion[0])
        scs_wrt_lcs = utils.create_pose(R, C)
        self.vcd.add_coordinate_system(name=channel, cs_type=types.CoordinateSystemType.sensor_cs,
                                  parent_name="vehicle-iso8855",
                                  pose_wrt_parent=types.PoseData(val=scs_wrt_lcs.flatten().tolist(),
                                                                 t_type=types.TransformDataType.matrix_4x4))
        if "CAM" in channel:
            fx = calibrated_sensor["camera_intrinsic"][0][0]
            fy = calibrated_sensor["camera_intrinsic"][1][1]
            cx = calibrated_sensor["camera_intrinsic"][0][2]
            cy = calibrated_sensor["camera_intrinsic"][1][2]
            K_3x4 = utils.fromPinholeParamsToCameraMatrix3x4(fx, fy, cx, cy)
            flatK3x4 = [item for sublist in K_3x4 for item in sublist]
            self.vcd.add_stream_properties(stream_name=channel,
                                      intrinsics=types.IntrinsicsPinhole(width_px=sample_data['width'],
                                                                         height_px=sample_data['height'],
                                                                         camera_matrix_3x4=flatK3x4,
                                                                         distortion_coeffs_1xN=None))

    def add_ego_transform(self, sample_data, frame_num, ego_uid):
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        ego_t = ego_pose['translation']
        C = np.array(ego_t).reshape(3, 1)
        ego_q = ego_pose['rotation']
        ypr = Quaternion(ego_q).yaw_pitch_roll
        #R = utils.euler2R([yaw, 0, 0])
        R = utils.q2R(ego_q[1], ego_q[2], ego_q[3], ego_q[0])
        R = Quaternion(ego_q).rotation_matrix
        odometry_pose = utils.create_pose(R, C)
        xyzypr = [C[0][0], C[1][0], C[2][0], ypr[0], ypr[1], ypr[2]]
        self.vcd.add_transform(frame_num, transform=types.Transform(
            src_name="vehicle-iso8855",
            dst_name="odom",
            transform_src_to_dst=types.TransformData(
                val=list(odometry_pose.flatten()),
                t_type=types.TransformDataType.matrix_4x4), odometry_xyzypr=xyzypr)
                          )
        # https://github.com/nutonomy/nuscenes-devkit/issues/56#issuecomment-558555994
        # RPY -> 0 because we are adding the cuboid info in the vehicle coordinate system, we are adding
        # the position information to transform from the rear coordinate to the center of the bbox, but
        # still need to check that is properly done
        cuboid_vals = [(EGO_LENGTH / 2) - EGO_REAR_OVERHANG, 0, EGO_HEIGHT / 2, 0.0, 0.0,
                       0, EGO_LENGTH, EGO_WIDTH, EGO_HEIGHT]

        self.vcd.add_object_data(uid=ego_uid,
                            object_data=types.cuboid(name='bbox3D', val=cuboid_vals,
                                                     coordinate_system='vehicle-iso8855'),
                            frame_value=frame_num)

    def add_annotation(self, annotation_token, objects_uids, frame_num):
        annotation = self.nusc.get('sample_annotation', annotation_token)
        instance_token = annotation['instance_token']
        instance = self.nusc.get('instance', instance_token)
        category_token = instance['category_token']
        category = self.nusc.get('category', category_token)
        attribute_token = annotation['attribute_tokens']
        # box = self.nusc.get_box(sample_annotation_token=annotation_token)
        speed = self.nusc.box_velocity(annotation_token)
        speed = np.nan_to_num(speed)

        if len(attribute_token) > 0:
            attribute = self.nusc.get('attribute', attribute_token[0])
            self.vcd.add_object_data(uid=objects_uids[instance['token']],
                                object_data=types.text(name='attr', val=attribute['name']),
                                frame_value=frame_num)

        visibility_token = annotation['visibility_token']
        visibility = self.nusc.get('visibility', visibility_token)
        t = annotation['translation']
        q = annotation['rotation']
        size = annotation['size']
        rvec = Quaternion(q).yaw_pitch_roll
        #R = utils.q2R(q[1], q[2], q[3], q[0])
        #rvec2 = utils.R2rvec(R)
        #print(rvec, rvec2.flatten())
        #cuboid_vals = [float(t[0]), float(t[1]), float(t[2]), float(rvec[0][0]), float(rvec[1][0]), float(rvec[2][0]),
        #               float(size[1]), float(size[0]), float(size[2])]
        cuboid_vals = [float(t[0]), float(t[1]), float(t[2]), float(rvec[2]), float(rvec[1]), float(rvec[0]),
                       float(size[1]), float(size[0]), float(size[2])]
        #print(cuboid_vals)
        self.vcd.add_object_data(uid=objects_uids[instance['token']],
                            object_data=types.cuboid(name='bbox3D', val=cuboid_vals, coordinate_system='odom'),
                            frame_value=frame_num)
        self.vcd.add_object_data(uid=objects_uids[instance['token']],
                            object_data=types.num('num_lidar_pts', annotation['num_lidar_pts']),
                            frame_value=frame_num)
        self.vcd.add_object_data(uid=objects_uids[instance['token']],
                            object_data=types.num('num_radar_pts', annotation['num_radar_pts']),
                            frame_value=frame_num)
        self.vcd.add_object_data(uid=objects_uids[instance['token']],
                            object_data=types.text('visibility', visibility['level']),
                            frame_value=frame_num)
        self.vcd.add_object_data(uid=objects_uids[instance['token']],
                            object_data=types.text('sample_annotation_token', annotation['token']),
                            frame_value=frame_num)
        self.vcd.add_object_data(uid=objects_uids[instance['token']],
                            object_data=types.vec('velocity_vector', tuple(speed)), frame_value=frame_num)
        self.vcd.add_object_data(uid=objects_uids[instance['token']],
                            object_data=types.num('velocity', np.linalg.norm(speed)), frame_value=frame_num)

    def add_can_expansion(self, can_expansion_path):
        scene_name = self.vcd.get_metadata()['scene_name']
        # Add IMU data from CAN expansion
        with open(can_expansion_path + scene_name + "_ms_imu.json", 'r') as f:
            sync_data = self.get_can_sync_data(f)
            self.add_ms_imu(sync_data)
        # Pose, steerangle, veh_monitor, zoesensors, veh_info
        with open(can_expansion_path + scene_name + "_pose.json", 'r') as f:
            sync_data = self.get_can_sync_data(f)
            self.add_pose(sync_data)
        with open(can_expansion_path + scene_name + "_steeranglefeedback.json", 'r') as f:
            sync_data = self.get_can_sync_data(f)
            self.add_steerangle_feedback(sync_data)
        with open(can_expansion_path + scene_name + "_vehicle_monitor.json", 'r') as f:
            sync_data = self.get_can_sync_data(f)
            self.add_vehicle_monitor(sync_data)
        with open(can_expansion_path + scene_name + "_zoesensors.json", 'r') as f:
            sync_data = self.get_can_sync_data(f)
            self.add_zoe_sensors(sync_data)
        with open(can_expansion_path + scene_name + "_zoe_veh_info.json", 'r') as f:
            sync_data = self.get_can_sync_data(f)
            self.add_vehicle_monitor(sync_data)

    # These functions can be generalized, but as in "add_vehicle_monitor", but are we using all the values in all jsons?
    def add_zoe_sensors(self, sync_data):
        if not self.vcd.has_stream("ZOE_SENSORS"):
            self.vcd.add_stream(stream_name="ZOE_SENSORS", uri='', description="Sensor data from the Renault ZOE",
                           stream_type=core.StreamType.other)
        for frame, value in sync_data.items():
            self.vcd.add_stream_properties(stream_name="ZOE_SENSORS",
                                           stream_sync=types.StreamSync(frame_vcd=frame),
                                           properties=value)

    def add_vehicle_monitor(self, sync_data):
        ego_uid = self.vcd.get_object_uid_by_name("ego_vehicle")
        for frame, data in sync_data.items():
            for key, value in data.items():
                self.vcd.add_object_data(uid=ego_uid, object_data=types.num(name=key, val=value), frame_value=frame)

    def add_steerangle_feedback(self, sync_data):
        ego_uid = self.vcd.get_object_uid_by_name("ego_vehicle")
        for frame, value in sync_data.items():
            self.vcd.add_object_data(uid=ego_uid, object_data=types.num(name="steer_angle", val=value['value']),
                                  frame_value=frame)

    def add_pose(self, sync_data):
        ego_uid = self.vcd.get_object_uid_by_name("ego_vehicle")
        for frame, value in sync_data.items():
            self.vcd.add_object_data(uid=ego_uid, object_data=types.vec(name="accel", val=tuple(value['accel'])),
                                  frame_value=frame)
            self.vcd.add_object_data(uid=ego_uid,
                                  object_data=types.vec(name="rotation_rate", val=tuple(value['rotation_rate'])),
                                  frame_value=frame)
            self.vcd.add_object_data(uid=ego_uid, object_data=types.vec(name="vel", val=tuple(value['vel'])),
                                  frame_value=frame)
            self.vcd.add_object_data(uid=ego_uid, object_data=types.point3d(name="pos", val=tuple(value['pos'])),
                                  frame_value=frame)

    def add_ms_imu(self, sync_data):
        if not self.vcd.has_stream("IMU"):
            self.vcd.add_stream(stream_name="IMU", uri='', description="GPS-IMU",
                           stream_type=core.StreamType.gps_imu)
        for frame, value in sync_data.items():
            self.vcd.add_stream_properties(stream_name="IMU",
                                           stream_sync=types.StreamSync(frame_vcd=frame),
                                           properties=value)

    def get_can_sync_data(self, file, fill_missing=False):
        sync_data = {}
        data = json.load(file)
        total_frames = self.vcd.get_frame_intervals().get_length()
        frame_n = 0
        frame = self.vcd.get_frame(frame_n)
        time_stamp = int(frame['frame_properties']['timestamp'])
        prev_diff = 999999999999
        prev_o = None
        for o in data:
            diff = time_stamp - o['utime']
            if frame_n < total_frames:
                if diff < 0:
                    if math.fabs(diff) < math.fabs(prev_diff):
                        sync_data[frame_n] = o
                    else:
                        sync_data[frame_n] = prev_o

                    frame = self.vcd.get_frame(frame_n)
                    time_stamp = int(frame['frame_properties']['timestamp'])
                    frame_n += 1

                prev_diff = diff
                prev_o = o
        if fill_missing:
            for i in range(frame_n, total_frames):
                sync_data[i] = prev_o

        return sync_data

    def add_map_expansion(self, map_expansion_path):
        basemap = self.vcd.get_metadata()["location"]
        print("Adding map expansion for map: " + basemap)
        nusc_map = NuScenesMap(dataroot=map_expansion_path, map_name=basemap)
        static_objs = self.get_static_objects_in_scene(nusc_map)
        token_uid_map, lane_divider_data, polygons = self.add_static_objects(nusc_map, static_objs)
        self.add_map_relations(nusc_map, static_objs, token_uid_map, polygons)

    def get_static_objects_in_scene(self, nusc_map):
        ego_uid = self.vcd.get_object_uid_by_name("ego_vehicle")
        x_coord = (0, 0)
        y_coord = (0, 0)
        scene = scl.Scene(self.vcd)
        for f, frame in enumerate(self.vcd.data['openlabel']['frames']):
            cuboid_ego = self.vcd.get_object_data(uid=ego_uid, data_name='bbox3D', frame_num=f)
            cuboid_ego = scene.transform_cuboid(cuboid_ego['val'], cs_src="vehicle-iso8855", cs_dst="odom",
                                                frame_num=f)
            if f == 0:
                x_coord = (cuboid_ego[0], cuboid_ego[0])
                y_coord = (cuboid_ego[1], cuboid_ego[1])
            else:
                x_coord = (min(x_coord[0], cuboid_ego[0]), max(x_coord[1], cuboid_ego[0]))
                y_coord = (min(y_coord[0], cuboid_ego[1]), max(y_coord[1], cuboid_ego[1]))

        x_coord = (x_coord[0] - 50, x_coord[1] + 50)
        y_coord = (y_coord[0] - 50, y_coord[1] + 50)
        print(x_coord, y_coord)
        static_objects = nusc_map.get_records_in_patch([x_coord[0], y_coord[0], x_coord[1], y_coord[1]],
                                                       layer_names=["drivable_area", "lane", "road_segment",
                                                                    "ped_crossing", "walkway", "stop_line",
                                                                    "carpark_area"])

        return static_objects

    def add_static_objects(self, nusc_map, static_objects):
        token_uid_map = {}
        lane_divider_data = {}
        polygons = {}
        for key, token_list in static_objects.items():
            lane_divider_poses = []
            for token in token_list:
                if token != '':
                    lane_divider = {}
                    uid = self.vcd.add_object(name=token, type=key)
                    token_uid_map[token] = uid
                    static_object_info = nusc_map.get(key, token)
                    if "polygon_token" in static_object_info:
                        if key == 'drivable_area':
                            polygons[uid] = [nusc_map.extract_polygon(polygon_token) for polygon_token in
                                             static_object_info['polygon_tokens']]
                        else:
                            polygon = nusc_map.extract_polygon(static_object_info['polygon_token'])
                            polygons[uid] = [polygon]
                            xy = np.array(polygon.exterior.coords)
                            # add a z = zero to each point
                            xyz = np.pad(xy, ((0, 0), (0, 1)), 'constant')
                            self.vcd.add_object_data(uid=uid, object_data=types.poly3d(name="polygon",
                                                                                       val=xyz.flatten().tolist(),
                                                                                       closed=polygon.is_closed))

        return token_uid_map, lane_divider_data, polygons

    def add_map_relations(self, nusc_map, static_objects, token_uid_map, polygons):
        scene = scl.Scene(self.vcd)
        ego_uid = self.vcd.get_object_uid_by_name("ego_vehicle")
        # Relation -> Located in
        for o in self.vcd.get_objects():
            relations = {}
            fi = None
            try:
                fi = self.vcd.get_object_data_frame_intervals(uid=o, data_name="bbox3D")
            except:
                print("No bbox3D data for object: " + o)
            if fi is not None:
                for fis_num in fi.fis_num:
                    for f in range(fis_num[0], fis_num[1]+1):
                        cuboid = self.vcd.get_object_data(uid=o, data_name="bbox3D", frame_num=f)['val']
                        if o == ego_uid:
                            cuboid = scene.transform_cuboid(cuboid, cs_src="vehicle-iso8855", cs_dst="odom",
                                                                frame_num=f)
                        point = Point(cuboid[0], cuboid[1])
                        #layers = nusc_map.layers_on_point(cuboid[0], cuboid[1])
                        for uid, polygon_list in polygons.items():
                            for polygon in polygon_list:
                                if point.within(polygon):
                                    if uid not in relations:
                                        # If the relation is not in the dictionary, add it
                                        relations[uid] = [(f, f)]
                                    else:
                                        # If the relation existed in the last frame, update frame interval
                                        if f - relations[uid][-1][1] == 1:
                                            relations[uid][-1] = (relations[uid][-1][0], f)
                                        # If the relation existed but NOT in the last frame, create new frame interval for that obj
                                        else:
                                            relations[uid].append((f, f))

                if relations == {}:
                    obj = self.vcd.get_object(o)
                    print(f"Object {obj['name']} of type {obj['type']} is not located in any polygon")
                else:
                    for key, value in relations.items():
                        self.vcd.add_relation_object_object('Relation' + str(self.vcd.get_num_relations()),
                                                                  'isLocatedIn', o, key, frame_value=value)

        # Relation -> Incoming/Outgoing lane
        for token in static_objects['lane']:
            incoming_lanes = nusc_map.get_incoming_lane_ids(token)
            outgoing_lanes = nusc_map.get_outgoing_lane_ids(token)

            if incoming_lanes != []:
                for lane in incoming_lanes:
                    if lane not in token_uid_map:
                        uid = self.vcd.add_object(name=lane, type="lane")
                        token_uid_map[lane] = uid

                    self.vcd.add_relation_object_object('Relation' + str(self.vcd.get_num_relations()),
                                                   'hasIncomingLane', token_uid_map[token], token_uid_map[lane])

            if outgoing_lanes != []:
                for lane in outgoing_lanes:
                    if lane not in token_uid_map:
                        uid = self.vcd.add_object(name=lane, type="lane")
                        token_uid_map[lane] = uid

                    self.vcd.add_relation_object_object('Relation' + str(self.vcd.get_num_relations()),
                                                   'hasOutgoingLane', token_uid_map[token], token_uid_map[lane])

        # Relation -> Lane Next To
        polygons = {}
        for i, token in enumerate(static_objects['lane']):
            lane_info = nusc_map.get('lane', token)
            nodes = [nusc_map.get('node', node_token) for node_token in lane_info['exterior_node_tokens']]
            node_coords = [(node['x'], node['y']) for node in nodes]
            p = Polygon(node_coords)
            polygons[token] = p

        combinations = list(map(dict, itertools.combinations(polygons.items(), 2)))
        for i, j in combinations:
            if polygons[i].intersects(polygons[j]):
                #print(f"Intersects {token_uid_map[i]} and {token_uid_map[j]}")
                self.vcd.add_relation_object_object('Relation' + str(self.vcd.get_num_relations()),
                                               'isNextTo', token_uid_map[i], token_uid_map[j])
                self.vcd.add_relation_object_object('Relation' + str(self.vcd.get_num_relations()),
                                               'isNextTo', token_uid_map[j], token_uid_map[i])


