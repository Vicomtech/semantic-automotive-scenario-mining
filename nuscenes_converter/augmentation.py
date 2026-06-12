import utils
import vcd.core as core
import vcd.types as types
import vcd.scl as scl
import os
import time
import numpy as np
from trajectory import trajectory

def add_distances(vcd, add_spatial_regions=True, add_following=False):
    scene = scl.Scene(vcd)
    ego_uid = vcd.get_object_uid_by_name("ego_vehicle")
    print(f"Adding ego2obj distances to VCD: {vcd.data['openlabel']['metadata']['scene_name']}")
    for f in vcd.data['openlabel']['frames']:
        frame_data = vcd.get_frame(f)
        frame_data_objs = frame_data['objects']
        cuboid_ego = vcd.get_object_data(uid=ego_uid, data_name='bbox3D', frame_num=f)
        for o in list(frame_data_objs):
            if o != ego_uid:
                try:
                    cuboid = vcd.get_object_data(uid=o, data_name='bbox3D', frame_num=f)
                except:
                    cuboid = None
                if type(cuboid) == dict:
                    cuboid_o_ego = scene.transform_cuboid(cuboid['val'], cs_src="odom", cs_dst="vehicle-iso8855",
                                                         frame_num=f)
                    distance = utils.distance_between_cuboids(cuboid_o_ego, cuboid_ego["val"])
                    vcd.add_object_data(uid=o, object_data=types.num(name="distance_to_ego", val=distance),
                                          frame_value=f)
                    if add_spatial_regions:
                        regions = utils.regions_between_cuboids(ego_cuboid=cuboid_ego["val"], cuboid=cuboid_o_ego)
                        for region in regions:
                            vcd.add_object_data(uid=o, object_data=types.boolean(name=region, val=True),
                                                frame_value=f)
    if add_following:
        if add_spatial_regions:
            return add_following_tag(vcd)

        else:
            print("Cannot add following tag without spatial regions.")
    return vcd

def add_ego_velocity_vec(vcd, max_time_diff=1.5):
    scene = scl.Scene(vcd)
    ego_uid = vcd.get_object_uid_by_name("ego_vehicle")
    frames = len(vcd.data['openlabel']['frames'])
    print(f"Adding ego velocity vec to VCD: {vcd.data['openlabel']['metadata']['scene_name']}")
    for f in range(frames):
        max_diff = max_time_diff
        has_next = True
        has_prev = True
        cuboid_ego = vcd.get_object_data(uid=ego_uid, data_name='bbox3D', frame_num=f)
        if f == 0:
            has_prev = False
        elif f == frames - 1:
            has_next = False

        if has_prev:
            first_frame = f-1
        else:
            first_frame = f
        if has_next:
            last_frame = f+1
        else:
            last_frame = f
        first = scene.transform_cuboid(cuboid_ego['val'], cs_src="vehicle-iso8855", cs_dst="odom",
                                      frame_num=first_frame)
        last = scene.transform_cuboid(cuboid_ego['val'], cs_src="vehicle-iso8855", cs_dst="odom",
                                      frame_num=last_frame)

        pos_last = np.array(last[0:3])
        pos_first = np.array(first[0:3])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * float(vcd.data['openlabel']['frames'][last_frame]['frame_properties']['timestamp']) # last
        time_first = 1e-6 * float(vcd.data['openlabel']['frames'][first_frame]['frame_properties']['timestamp']) # first
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_diff *= 2

        if time_diff > max_diff:
            # If time_diff is too big, don't return an estimate.
            pass # Should I add a NaN value in VCD files?
        else:
            vel_vec = pos_diff / time_diff
            vcd.add_object_data(uid=ego_uid,object_data=types.vec('velocity_vector', tuple(vel_vec)), frame_value=f)

    return vcd

def add_ttc(vcd, TTC=3, sampling_fq=2):
    scene = scl.Scene(vcd)
    ego_uid = vcd.get_object_uid_by_name("ego_vehicle")
    obj_traj_dict = {}
    dynamic_objs = vcd.get_objects_with_object_data_name(data_name='bbox3D')
    print(f"Adding ego2obj TTC to VCD: {vcd.data['openlabel']['metadata']['scene_name']}")
    for obj_uid in dynamic_objs:
        obj_traj_dict[obj_uid] = {}

    # Iterate over frames to add all vehicle trajectories
    for frame_num in range(len(vcd.data["openlabel"]["frames"])):
        frame = vcd.get_frame(frame_num)
        ts = int(frame["frame_properties"]["timestamp"]) / 1e6 # timestamp in seconds
        for obj_uid in dynamic_objs:
            bbox3d = vcd.get_object_data(obj_uid, "bbox3D", frame_num)
            if bbox3d is None:
                continue
            if obj_uid == ego_uid:
                cuboid = scene.transform_cuboid(bbox3d["val"], cs_src="vehicle-iso8855", cs_dst="odom", frame_num=frame_num)
                try:
                    yaw_r = vcd.get_object_data(obj_uid, "rotation_rate", frame_num)["val"][2]
                except:
                    yaw_r = None
            else:
                cuboid = bbox3d["val"]
                yaw_r = None

            vel = vcd.get_object_data(obj_uid, "velocity_vector", frame_num)["val"]

            if obj_traj_dict[obj_uid] == {}:
                veh_traj = trajectory(cuboid[0], cuboid[1], cuboid[5], cuboid[6], cuboid[7], vel[0], vel[1], ts, yaw_r, TTC, sampling_fq, frame_start=frame_num)
                obj_traj_dict[obj_uid] = veh_traj

            else:
                obj_traj_dict[obj_uid].add_state(cuboid[0], cuboid[1], cuboid[5], vel[0], vel[1], ts, yaw_r)

    # Calculate trajectory estimation for each vehicle
    for obj_uid in obj_traj_dict.keys():
        obj_traj_dict[obj_uid].calculate_trajectories()

    # calculate TTC of ego vs all other vehicles
    ego_traj = obj_traj_dict[ego_uid]
    for obj_uid in dynamic_objs:
        if obj_uid == ego_uid:
            continue
        obj_traj = obj_traj_dict[obj_uid]
        collision, frame_ttc = ego_traj.calculate_ttc(obj_traj, draw=False)
        if collision:
            print(f"TTC between ego and {obj_uid} in scene {vcd.data['openlabel']['metadata']['scene_name']}")
            for f in frame_ttc.keys():
                vcd.add_object_data(uid=obj_uid, object_data=types.num(name="TTC", val=frame_ttc[f]), frame_value=f)

    return vcd

def add_following_tag(vcd):
    # Adds the following tag to the vehicle that is closest to the ego  in each frame, only if it is in the same lane
    # Add distance must be executed first
    located_in_relations = vcd.get_elements_of_type(core.ElementType.relation, "isLocatedIn")
    ego_uid = vcd.get_object_uid_by_name("ego_vehicle")
    # Ego frame to lane uid map
    ego_location_dict = {}
    # Ego frame to closest vehicle uid map
    frame_following_dict = {}
    # Ego frame to closest vehicle distance map
    min_dist_dict = {}

    for uid in located_in_relations:
        relation = vcd.get_relation(uid)
        rdf_subject = relation["rdf_subjects"][0]["uid"]
        rdf_object = relation["rdf_objects"][0]["uid"]
        if rdf_subject == ego_uid:
            if vcd.get_object(rdf_object)["type"] == "lane":
                for fi in relation["frame_intervals"]:
                    for f in range(fi["frame_start"], fi["frame_end"]+1):
                        ego_location_dict[f] = rdf_object

    for uid in located_in_relations:
        relation = vcd.get_relation(uid)
        rdf_subject = relation["rdf_subjects"][0]["uid"]
        rdf_object = relation["rdf_objects"][0]["uid"]
        if rdf_subject != ego_uid:
            if vcd.get_object(rdf_object)["type"] == "lane":
                for fi in relation["frame_intervals"]:
                    for f in range(fi["frame_start"], fi["frame_end"]+1):
                        if f in ego_location_dict.keys():
                            # If ego is located in the same lane as the other actor
                            if ego_location_dict[f] == rdf_object:
                                # Try to get data
                                try:
                                    dist = vcd.get_object_data(rdf_subject, "distance_to_ego", f)
                                    obj_type = vcd.get_object_data(rdf_subject, "attr", f)
                                    front_of_ego = vcd.get_object_data(rdf_subject, "front_of_ego", f)
                                except:
                                    continue
                                # Init dict
                                if front_of_ego and obj_type["val"] != "vehicle.parked":
                                    if f not in frame_following_dict.keys():
                                        min_dist_dict[f] = dist["val"]
                                        frame_following_dict[f] = rdf_subject
                                    else:
                                        # If the vehicle is closer than the current closest vehicle update following tag
                                        if dist["val"] < min_dist_dict[f]:
                                            min_dist_dict[f] = dist["val"]
                                            frame_following_dict[f] = rdf_subject
    # add following tags
    if frame_following_dict != {}:
        print("Adding following tags")
    for f in frame_following_dict.keys():
        vcd.add_object_data(uid=frame_following_dict[f], object_data=types.boolean(name="ego_following", val=True),
                            frame_value=f)

    return vcd

if __name__ == '__main__':
    openlabel_path = "vcd_nuscenes"
    t1 = time.time()
    for path, subdirs, files in os.walk(openlabel_path):
        for name in files:
            vcd = core.OpenLABEL(os.path.join(path, name))
            new_vcd = add_distances(vcd, add_following=True)
            new_vcd = add_ego_velocity_vec(new_vcd)
            new_vcd = add_ttc(new_vcd)
            new_vcd.save(os.path.join(path, name), pretty=True)
    print("Time: ", time.time()-t1)