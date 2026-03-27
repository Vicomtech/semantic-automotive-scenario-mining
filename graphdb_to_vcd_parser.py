import os
import shutil
from pathlib import Path

import yaml
import requests
import vcd.core as core
import vcd.types as types
from vcd.core import VCD, ElementType, SetMode

##############################################################################
#                           SPARQL CLIENT
##############################################################################
class SparqlClient:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def query(self, sparql_query: str) -> dict | None:
        """
        Execute the SPARQL query (GET) and return results in JSON.
        """
        params = {"query": sparql_query}
        headers = {"Accept": "application/sparql-results+json"}

        try:
            response = requests.get(self.endpoint_url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[ERROR] Código HTTP: {response.status_code}")
                print(f"[ERROR] Respuesta: {response.text}")
                return None
        except Exception as e:
            print(f"[EXCEPCIÓN] Al ejecutar SPARQL: {e}")
            return None


##############################################################################
# ACTION AND EVENT PROCESSORS
##############################################################################
class ActionsProcessor:
    def __init__(self, sparql_results: dict):
        self.sparql_results = sparql_results

    def get_data_by_action(self) -> dict:
        data_by_action = {}
        if not self.sparql_results:
            return data_by_action

        for b in self.sparql_results["results"]["bindings"]:
            action_uri = b["action"]["value"]
            scene_uri  = b["scene"]["value"]
            type_uri   = b.get("type", {}).get("value", "")
            start_str  = b.get("start", {}).get("value", None)
            end_str    = b.get("end",   {}).get("value", None)

            action_id  = action_uri.split("#")[-1]
            scene_name = scene_uri.split("#")[-1]
            sem_type   = type_uri.split("#")[-1] if "#" in type_uri else "UnknownAction"

            start_val = int(start_str) if start_str else 0
            end_val   = int(end_str)   if end_str   else start_val

            if action_id not in data_by_action:
                data_by_action[action_id] = {
                    "scene_name": scene_name,
                    "semantic_type": sem_type,
                    "start_framestamp": start_val,
                    "end_framestamp": end_val
                }
            else:
                if start_val < data_by_action[action_id]["start_framestamp"]:
                    data_by_action[action_id]["start_framestamp"] = start_val
                if end_val > data_by_action[action_id]["end_framestamp"]:
                    data_by_action[action_id]["end_framestamp"] = end_val

            data_by_action[action_id]["scene_name"] = scene_name

        return data_by_action


class EventsProcessor:
    def __init__(self, sparql_results: dict):
        self.sparql_results = sparql_results

    def get_data_by_event(self) -> dict:
        data_by_event = {}
        if not self.sparql_results:
            return data_by_event

        for b in self.sparql_results["results"]["bindings"]:
            event_uri = b["event"]["value"]
            scene_uri = b["scene"]["value"]
            type_uri  = b.get("type", {}).get("value", "")
            framestamp_str = b.get("framestamp", {}).get("value", None)

            event_id   = event_uri.split("#")[-1]
            scene_name = scene_uri.split("#")[-1]
            sem_type   = type_uri.split("#")[-1] if "#" in type_uri else "UnknownEvent"
            f_val      = int(framestamp_str) if framestamp_str else 0

            if event_id not in data_by_event:
                data_by_event[event_id] = {
                    "scene_name": scene_name,
                    "semantic_type": sem_type,
                    "framestamp": f_val
                }
            else:
                data_by_event[event_id]["scene_name"]   = scene_name
                data_by_event[event_id]["framestamp"]   = f_val
                data_by_event[event_id]["semantic_type"] = sem_type

        return data_by_event


##############################################################################
#  RELATIONSHIP PROCESSOR
##############################################################################
class RelationshipsProcessor:
    def __init__(self, sparql_results: dict):
        self.sparql_results = sparql_results

    def get_data_by_relation(self) -> dict:
        data_by_rel = {}
        if not self.sparql_results:
            return data_by_rel

        for b in self.sparql_results["results"]["bindings"]:
            scene_uri = b.get("scene", {}).get("value")
            if not scene_uri:
                continue

            rel_type_uri = b["relType"]["value"]
            # Skip hasAction / hasEvent because we're already inserting them in the same scene
            if rel_type_uri.endswith("hasAction") or rel_type_uri.endswith("hasEvent"):
                continue

            subject_uri = b["subject"]["value"]
            object_uri  = b["object"]["value"]

            sub_id = subject_uri.split("#")[-1]
            obj_id = object_uri.split("#")[-1]
            if sub_id.startswith("scene-") or obj_id.startswith("scene-"): # skip relations where subject or object are scenes for the same reason; we're in the same scene
                continue

            rel_type = rel_type_uri.split("#")[-1]
            scene_name = scene_uri.split("#")[-1]

            sub_type_uri = b.get("subjectType", {}).get("value", "")
            obj_type_uri = b.get("objectType",  {}).get("value", "")

            rel_id = f"{sub_id}{rel_type}{obj_id}"
            if rel_id not in data_by_rel:
                data_by_rel[rel_id] = {
                    "scene_name": scene_name,
                    "semantic_type": rel_type,
                    "subject": {
                        "name": sub_id,
                        "type_uri": sub_type_uri
                    },
                    "object": {
                        "name": obj_id,
                        "type_uri": obj_type_uri
                    }
                }

        return data_by_rel


##############################################################################
#  HYPOTENUSE PROCESSOR (NEW FORMAT)
##############################################################################
class HypotenuseProcessor:
    """
    Interprets the query that returns: ?vehiculo ?frame ?hypotenuse
    Example row:
      adas:ego_vehicle, adas:scene-0048_ego_frame_25, 2.6655808606663594
    And groups it into: scene_name -> [ { object_name, framestamp, hypotenuse }, ...]
    """
    def __init__(self, sparql_results: dict):
        self.sparql_results = sparql_results

    def get_data_by_scene(self) -> dict:
        """
        Final structure:
        {
          "scene-0048": [
            {"object_name":"ego_vehicle", "framestamp":25, "hypotenuse":2.66},
            ...
          ],
          ...
        }
        """
        data_by_scene = {}
        if not self.sparql_results:
            return data_by_scene

        for b in self.sparql_results["results"]["bindings"]:
            veh_uri   = b["vehiculo"]["value"]      # e.g. "adas:ego_vehicle"
            frame_uri = b["frame"]["value"]         # e.g. "adas:scene-0048_ego_frame_25"
            hyp_str   = b["hypotenuse"]["value"]

            # 1) object_name = "ego_vehicle"
            obj_name  = veh_uri.split("#")[-1]

            # 2) frame_uri => "scene-0048_ego_frame_25"
            short_name = frame_uri.split("#")[-1]  # "scene-0048_ego_frame_25"
            parted = short_name.split("_")         # ["scene-0048","ego","frame","25"]
            if len(parted) < 4:
                continue
            scene_name = parted[0]  # "scene-0048"
            try:
                frame_int = int(parted[-1])  # 25
            except ValueError:
                continue

            # 3) float value
            hyp_val = float(hyp_str)

            if scene_name not in data_by_scene:
                data_by_scene[scene_name] = []

            data_by_scene[scene_name].append({
                "object_name": obj_name,
                "framestamp": frame_int,
                "hypotenuse": hyp_val
            })

        return data_by_scene


##############################################################################
#  UNIFIED INSERTER
##############################################################################
import os
from vcd.core import VCD, ElementType, SetMode
import vcd.types as types

class VcdEventsActionsInserter:
    """
    Inserts actions, events, relations, and additional data into the VCD
    """
    def __init__(self,
                 data_by_action: dict,
                 data_by_event: dict,
                 data_by_relation: dict,
                 data_by_hypotenuse: dict,
                 thresholds: dict,
                 input_dir: str,
                 output_dir: str):
        self.data_by_action = data_by_action
        self.data_by_event = data_by_event
        self.data_by_relation = data_by_relation
        self.data_by_hypotenuse = data_by_hypotenuse
        self.thresholds = thresholds
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.scenes_with_data = {
            info["scene_name"] for info in self.data_by_action.values()
        }
        self.scenes_with_data.update(info["scene_name"] for info in self.data_by_event.values())
        self.scenes_with_data.update(info["scene_name"] for info in self.data_by_relation.values())
        self.scenes_with_data.update(self.data_by_hypotenuse.keys())

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.json')]
        for file_name in files:
            input_path = os.path.join(self.input_dir, file_name)
            out_path = os.path.join(self.output_dir, file_name)
            vcd_obj = VCD()
            try:
                vcd_obj.load_from_file(input_path)
            except Exception as e:
                print(f"[ERROR] Loading '{input_path}': {e}")
                continue

            scene_name = self._get_scene_name_from_metadata(vcd_obj)
            if not scene_name:
                print(f"[INFO] '{file_name}' without scene_name, copying.")
                shutil.copy2(input_path, out_path)
                continue
            if scene_name not in self.scenes_with_data:
                shutil.copy2(input_path, out_path)
                continue

            a_cnt = self._insert_actions_for_scene(vcd_obj, scene_name)
            e_cnt = self._insert_events_for_scene(vcd_obj, scene_name)
            r_cnt = self._insert_relations_for_scene(vcd_obj, scene_name)
            h_cnt = self._insert_hypotenuse_for_scene(vcd_obj, scene_name)

            total = a_cnt + e_cnt + r_cnt + h_cnt
            if total > 0:
                try:
                    vcd_obj.save(out_path)
                    print(f"[OK] '{file_name}' -> '{out_path}' | A={a_cnt}, E={e_cnt}, R={r_cnt}, H={h_cnt}")
                except Exception as e:
                    print(f"[ERROR] Saving '{out_path}': {e}")
            else:
                shutil.copy2(input_path, out_path)

    def _get_scene_name_from_metadata(self, vcd_obj: VCD) -> str:
        return (vcd_obj.get_metadata() or {}).get('scene_name', '')

    def _find_action_uid(self, vcd_obj: VCD, name: str) -> str | None:
        # get_actions() returns a list of UID strings
        for uid in vcd_obj.get_actions() or []:
            action = vcd_obj.get_action(uid)
            if action and action.get('name') == name:
                return uid
        return None
    
    def _find_event_uid(self, vcd_obj: VCD, name: str) -> str | None:
        # get_events() returns a list of UID strings
        for uid in vcd_obj.get_events() or []:
            event = vcd_obj.get_event(uid)
            if event and event.get('name') == name:
                return uid
        return None
    

    def _insert_relations_for_scene(self, vcd_obj: VCD, scene_name: str) -> int:
        inserted = 0
        existing = vcd_obj.get_relations() or {}
        rel_counter = len(existing)

        for rel_id, info in self.data_by_relation.items():
            if info['scene_name'] != scene_name:
                continue
            sem_type = info['semantic_type']
            sub_name = info['subject']['name']
            obj_name = info['object']['name']

            # resolve numeric UIDs
            if info['subject']['type_uri'].lower().endswith('action'):
                sub_uid = self._find_action_uid(vcd_obj, sub_name)
            elif info['subject']['type_uri'].lower().endswith('event'):
                sub_uid = self._find_event_uid(vcd_obj, sub_name)
            else:
                sub_uid = vcd_obj.get_object_uid_by_name(sub_name)

            if info['object']['type_uri'].lower().endswith('action'):
                obj_uid = self._find_action_uid(vcd_obj, obj_name)
            elif info['object']['type_uri'].lower().endswith('event'):
                obj_uid = self._find_event_uid(vcd_obj, obj_name)
            else:
                obj_uid = vcd_obj.get_object_uid_by_name(obj_name)

            if sub_uid is None or obj_uid is None:
                print(f"[WARN] Relation '{rel_id}' missing UID: sub={sub_name}, obj={obj_name}")
                continue

            f_start, f_end = self._get_relation_frames(
                vcd_obj, sub_uid, info['subject']['type_uri'], obj_uid, info['object']['type_uri']
            )
            name = f"Relation{rel_counter}"
            rel_counter += 1
            vcd_obj.add_relation_subject_object(
                name=name,
                semantic_type=sem_type,
                subject_type=self._map_type_to_element_type(info['subject']['type_uri']),
                subject_uid=sub_uid,
                object_type=self._map_type_to_element_type(info['object']['type_uri']),
                object_uid=obj_uid,
                frame_value=(f_start, f_end),
                set_mode=SetMode.union
            )
            inserted += 1
        return inserted


    ###########################################################################
    #  HYPOTENUSE: LINE-BY-LINE FROM THE QUERY
    ###########################################################################

    def _insert_hypotenuse_for_scene(self, vcd_obj: VCD, scene_name: str) -> int:
        """
        Iterate over self.data_by_hypotenuse[scene_name], and for each row:
        - object_name (vehicle),
        - framestamp,
        - hypotenuse
        => add a Num(name="hypotenuse", val=...) to object_data, frame=framestamp
        """
        if scene_name not in self.data_by_hypotenuse:
            return 0

        inserted = 0
        items = self.data_by_hypotenuse[scene_name]

        for item in items:
            obj_name  = item["object_name"]     # e.g. "ego_vehicle"
            frame_val = item["framestamp"]      # e.g. 25
            hyp_val   = item["hypotenuse"]      # e.g. 2.6655...

            obj_uid = vcd_obj.get_object_uid_by_name(obj_name)
            if obj_uid is None:
                # not found => skip
                continue

            # Add a Num(name="hypotenuse", val=...) on this frame
            vcd_obj.add_object_data(
                uid=obj_uid,
                object_data=types.num("hypotenuse", hyp_val),
                frame_value=frame_val,
                set_mode=core.SetMode.union
            )
            inserted += 1

        return inserted

    def _get_uid_in_vcd(self, vcd_obj: VCD, name: str, type_uri: str):
        t = type_uri.lower()
        if t.endswith("action"):
            return vcd_obj.get_action_uid_by_name(name)
        elif t.endswith("event"):
            return vcd_obj.get_event_uid_by_name(name)
        else:
            return vcd_obj.get_object_uid_by_name(name)

    def _map_type_to_element_type(self, type_uri: str) -> ElementType:
        t = type_uri.lower()
        if t.endswith("action"):
            return ElementType.action
        elif t.endswith("event"):
            return ElementType.event
        else:
            return ElementType.object

    def _insert_actions_for_scene(self, vcd_obj: VCD, scene_name: str) -> int:
        inserted = 0
        for action_id, info in self.data_by_action.items():
            if info['scene_name'] != scene_name:
                continue
            st, en = info['start_framestamp'], info['end_framestamp']
            sem_type = info['semantic_type']

            # Insert the action and let VCD assign the numeric UID
            vcd_obj.add_action(
                name=action_id,
                semantic_type=sem_type,
                frame_value=(st, en),
                uid=None,
                set_mode=SetMode.union
            )
            inserted += 1

            parts = action_id.split('-')
            # All formats share: [<Type>, "scene", <SceneID>, ... , "frames_..."]

            # 1) BrakingHard  → ['BrakingHard','scene','0049','0','frames_27_29']
            if sem_type == 'BrakingHard':
                uid = self._find_action_uid(vcd_obj, action_id)
                if uid is not None:
                    brake_val = self.thresholds.get('brake')
                    # vehicle = parts[3]
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.num('threshold_braking_value', brake_val),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('vehicle', parts[3]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )

            # 2) AceleratingHard → ['AceleratingHard','scene','0820','0','frames_0_2']
            if sem_type == 'AceleratingHard':
                uid = self._find_action_uid(vcd_obj, action_id)
                if uid is not None:
                    accel_val = self.thresholds.get('acceleration')
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.num('threshold_acceleration_value', accel_val),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('vehicle', parts[3]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )

            # 3) PedestrianCrossingZebra → ['PedestrianCrossingZebra','scene','0978','18','41','frames_0_26']
            if sem_type == 'PedestrianCrossingZebra':
                # parts[3] = pedestrian, parts[4] = zebra_crossing
                uid = self._find_action_uid(vcd_obj, action_id)
                if uid is not None:
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('pedestrian', parts[3]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('zebra_crossing', parts[4]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )

            # 4) PedestrianCrossingRoad → ['PedestrianCrossingRoad','scene','0004','66','270','frames_7_22']
            if sem_type == 'PedestrianCrossingRoad':
                # parts[3] = pedestrian, parts[4] = road
                uid = self._find_action_uid(vcd_obj, action_id)
                if uid is not None:
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('pedestrian', parts[3]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('road', parts[4]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )

            # 5) Following → ['Following','scene','1079','1','0','frames_17_39']
            if sem_type == 'Following':
                # parts[3] = followed_car, parts[4] = following_car
                uid = self._find_action_uid(vcd_obj, action_id)
                if uid is not None:
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('followed_car', parts[3]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('following_car', parts[4]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )

            # 6) NearlyColliding → ['NearlyColliding','scene','0218','13','0','frames_4_6']
            if sem_type == 'NearlyColliding':
                # parts[3] = impacted_object, parts[4] = colliding_vehicle
                uid = self._find_action_uid(vcd_obj, action_id)
                if uid is not None:
                    threshold_val = self.thresholds.get('near_miss_collision')
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.num('threshold_near_miss_colision_value', threshold_val),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('impacted_object', parts[3]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('colliding_vehicle', parts[4]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )

            # 7) ChangingLane → ['ChangingLane','scene','0010','58','56','0','frames_33_34']
            if sem_type == 'ChangingLane':
                # parts[3] = FromLane, parts[4] = ToLane, parts[5] = vehicle
                uid = self._find_action_uid(vcd_obj, action_id)
                if uid is not None:
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('vehicle', parts[5]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('FromLane', parts[3]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('ToLane', parts[4]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )

            # 8) CuttingIn → ['CuttingIn','scene','0594','160','135','111','0','frames_15_39']
            if sem_type == 'CuttingIn':
                # parts[3]=FromLane, [4]=ToLane, [5]=cutting_in_vehicle, [6]=affected_vehicle
                uid = self._find_action_uid(vcd_obj, action_id)
                if uid is not None:
                    threshold_val = self.thresholds.get('cut_in_distance_to_ego')
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.num('threshold_cutting_in_value', threshold_val),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('affected_vehicle', parts[6]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('cutting_in_vehicle', parts[5]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('FromLane', parts[3]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('ToLane', parts[4]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )

            # 9) CuttingOut → ['CuttingOut','scene','0407','88','101','1','0','frames_0_4']
            if sem_type == 'CuttingOut':
                # parts[3]=FromLane, [4]=ToLane, [5]=cutting_out_vehicle, [6]=affected_vehicle
                uid = self._find_action_uid(vcd_obj, action_id)
                if uid is not None:
                    threshold_val = self.thresholds.get('cut_out_distance_to_ego')
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.num('threshold_cutting_out_value', threshold_val),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('affected_vehicle', parts[6]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('cutting_out_vehicle', parts[5]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('FromLane', parts[3]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_action_data(
                        uid=uid,
                        action_data=types.text('ToLane', parts[4]),
                        frame_value=(st, en),
                        set_mode=SetMode.union
                    )

        return inserted

    def _insert_events_for_scene(self, vcd_obj: VCD, scene_name: str) -> int:
        inserted = 0
        for event_id, info in self.data_by_event.items():
            if info['scene_name'] != scene_name:
                continue
            frm = info['framestamp']
            sem_type = info['semantic_type']

            # Insert the event and let VCD assign the numeric UID
            vcd_obj.add_event(
                name=event_id,
                semantic_type=sem_type,
                frame_value=(frm, frm),
                uid=None,
                set_mode=SetMode.union
            )
            inserted += 1

            parts = event_id.split('-')
            # All formats share: [<Type>, "scene", <SceneID>, ... , "frame_..."]

            # 1) HardBrake → ['HardBrake','scene','0049','0','frame_27']
            if sem_type == 'HardBrake':
                uid = self._find_event_uid(vcd_obj, event_id)
                if uid is not None:
                    brake_val = self.thresholds.get('brake')
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.num('threshold_braking_value', brake_val),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.text('vehicle', parts[3]),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )

            # 2) HardAcceleration → ['HardAcceleration','scene','0820','0','frame_2']
            if sem_type == 'HardAcceleration':
                uid = self._find_event_uid(vcd_obj, event_id)
                if uid is not None:
                    accel_val = self.thresholds.get('acceleration')
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.num('threshold_acceleration_value', accel_val),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.text('vehicle', parts[3]),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )

            # 3) PedestrianCrossesZebra → ['PedestrianCrossesZebra','scene','0358','98','194','frame_38']
            if sem_type == 'PedestrianCrossesZebra':
                # parts[3] = pedestrian, parts[4] = zebra_crossing
                uid = self._find_event_uid(vcd_obj, event_id)
                if uid is not None:
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.text('pedestrian', parts[3]),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.text('zebra_crossing', parts[4]),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )

            # 4) PedestrianCrossesRoad → ['PedestrianCrossesRoad','scene','0045','29','130','frame_0']
            if sem_type == 'PedestrianCrossesRoad':
                # parts[3] = pedestrian, parts[4] = road
                uid = self._find_event_uid(vcd_obj, event_id)
                if uid is not None:
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.text('pedestrian', parts[3]),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.text('road', parts[4]),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )

            # 5) Follows → ['Follows','scene','0564','29','0','frame_27']
            if sem_type == 'Follows':
                # parts[3] = followed_car, parts[4] = following_car
                uid = self._find_event_uid(vcd_obj, event_id)
                if uid is not None:
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.text('followed_car', parts[3]),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.text('following_car', parts[4]),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )

            # 6) NearMissCollision → ['NearMissCollision','scene','0777','10','0','frame_0']
            if sem_type == 'NearMissCollision':
                # parts[3] = impacted_object, parts[4] = colliding_vehicle
                uid = self._find_event_uid(vcd_obj, event_id)
                if uid is not None:
                    threshold_val = self.thresholds.get('near_miss_collision')
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.num('threshold_near_miss_colision_value', threshold_val),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.text('impacted_object', parts[3]),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )
                    vcd_obj.add_event_data(
                        uid=uid,
                        event_data=types.text('colliding_vehicle', parts[4]),
                        frame_value=frm,
                        set_mode=SetMode.union
                    )

        return inserted


    # I use this function so that when inserting relations it takes frame intervals from recorded events/actions (I couldn't get it from GraphDB)
    def _get_relation_frames(self, vcd_obj: VCD, sub_uid: str, sub_type_uri: str,
                             obj_uid: str, obj_type_uri: str):
        """
        If subject is action/event => read its frame_intervals
        else, if object is action/event => read its frame_intervals
        else => (0,0).
        """
        # subject
        if sub_type_uri.lower().endswith("action"):
            act = vcd_obj.get_action(sub_uid)
            fi = act.get("frame_intervals", [])
            if fi:
                return (fi[0]["frame_start"], fi[0]["frame_end"])
            return (0,0)

        if sub_type_uri.lower().endswith("event"):
            ev = vcd_obj.get_event(sub_uid)
            fi = ev.get("frame_intervals", [])
            if fi:
                return (fi[0]["frame_start"], fi[0]["frame_end"])
            return (0,0)

        # object side
        if obj_type_uri.lower().endswith("action"):
            act = vcd_obj.get_action(obj_uid)
            fi = act.get("frame_intervals", [])
            if fi:
                return (fi[0]["frame_start"], fi[0]["frame_end"])
            return (0,0)

        if obj_type_uri.lower().endswith("event"):
            ev = vcd_obj.get_event(obj_uid)
            fi = ev.get("frame_intervals", [])
            if fi:
                return (fi[0]["frame_start"], fi[0]["frame_end"])
            return (0,0)

        return (0,0)

##############################################################################
#                                   MAIN
##############################################################################
def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_path(root: Path, value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (root / p))


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "conf.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        raise SystemExit(1)

    config = load_config(config_path)

    thresholds = config.get("thresholds", {})
    endpoint_url = config.get("endpoint_url")
    if not endpoint_url:
        db_cfg = config.get("database", {})
        db_ip = db_cfg.get("db_ip")
        db_port = db_cfg.get("db_port")
        repo = db_cfg.get("repository")
        if db_ip and db_port and repo:
            endpoint_url = f"http://{db_ip}:{db_port}/repositories/{repo}"
    if not endpoint_url:
        print("[ERROR] endpoint_url not configured in conf.yaml")
        raise SystemExit(1)

    cfg_root = config_path.parent
    input_dir = resolve_path(cfg_root, config["vcd"]["vcd_path"])
    output_dir_value = config.get("paths", {}).get("output_dir", "./vcd_nuscenes_full_enriched")
    output_dir = resolve_path(cfg_root, output_dir_value)

    client = SparqlClient(endpoint_url)

    # 1) Query ACTIONS
    sparql_query_actions = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX adas: <http://www.semanticweb.org/vicomtech/ontologies/nuscenes#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>

    SELECT ?action ?scene ?type ?start ?end
    WHERE {
       ?action rdf:type ?type .
       ?scene adas:hasAction ?action .
       OPTIONAL { ?action adas:start_framestamp ?start. }
       OPTIONAL { ?action adas:end_framestamp   ?end. }
       FILTER (?type != adas:Action && ?type != owl:Thing)
    }
    ORDER BY ?scene ?action
    """

    # 2) Query EVENTS
    sparql_query_events = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX adas: <http://www.semanticweb.org/vicomtech/ontologies/nuscenes#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>

    SELECT ?event ?scene ?type ?framestamp
    WHERE {
       ?event rdf:type ?type .
       ?scene adas:hasEvent ?event .
       OPTIONAL { ?event adas:framestamp ?framestamp. }
       FILTER (?type != adas:Event && ?type != owl:Thing)
    }
    ORDER BY ?scene ?event
    """

    # 3) Query RELATIONS
    sparql_query_relations = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX adas: <http://www.semanticweb.org/vicomtech/ontologies/nuscenes#>

    SELECT ?scene ?relType ?subject ?subjectType ?object ?objectType
    WHERE {
    VALUES ?relType {
        adas:hasObject 
        adas:hasObjectFrom 
        adas:hasObjectTo 
        adas:participatesIn
    }

    ?subject ?relType ?object .

    {
        ?scene adas:hasAction ?subject .
    }
    UNION
    {
        ?scene adas:hasEvent ?subject .
    }
    UNION
    {
        ?scene adas:hasAction ?object .
    }
    UNION
    {
        ?scene adas:hasEvent ?object .
    }

    FILTER (?objectType != owl:Thing)
    OPTIONAL { ?subject rdf:type ?subjectType }
    OPTIONAL { ?object  rdf:type ?objectType }
    }
    """

    #sparql_query_relations = """
    #PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    #PREFIX adas: <http://www.semanticweb.org/vicomtech/ontologies/nuscenes#>

    #SELECT ?scene ?relType ?subject ?subjectType ?object ?objectType
    #WHERE {
      #VALUES ?relType {
        #adas:hasObject adas:hasObjectFrom adas:hasObjectTo adas:participatesIn
      #}
      #?subject ?relType ?object .
      #OPTIONAL { ?scene adas:hasAction ?subject . }
      #OPTIONAL { ?scene adas:hasEvent  ?subject . }
      #OPTIONAL { ?scene adas:hasAction ?object . }
      #OPTIONAL { ?scene adas:hasEvent  ?object . }

      #OPTIONAL { ?subject rdf:type ?subjectType . }
      #OPTIONAL { ?object  rdf:type ?objectType . }
    #}
    #ORDER BY ?scene ?subject ?relType ?object
    

    # 4) Query HYPOTENUSE in its new format:
    sparql_query_hypotenuse = """
    PREFIX adas: <http://www.semanticweb.org/vicomtech/ontologies/nuscenes#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?vehiculo ?frame ?hypotenuse
    WHERE {
      ?vehiculo adas:hasData ?frame .
      ?frame adas:hypotenuse ?hypotenuse .
    }
    ORDER BY ?vehiculo ?frame
    """

    # ---- Execute the queries
    actions_data = {}
    events_data  = {}
    relations_data = {}
    hyp_data_by_scene = {}

    # (A) Actions
    res_actions = client.query(sparql_query_actions)
    if res_actions:
        a_proc = ActionsProcessor(res_actions)
        actions_data = a_proc.get_data_by_action()

    # (B) Events
    res_events = client.query(sparql_query_events)
    if res_events:
        e_proc = EventsProcessor(res_events)
        events_data = e_proc.get_data_by_event()

    # (C) Relations
    res_rel = client.query(sparql_query_relations)
    if res_rel:
        r_proc = RelationshipsProcessor(res_rel)
        relations_data = r_proc.get_data_by_relation()

    # (D) Hypotenuse: NEW format
    res_hyp = client.query(sparql_query_hypotenuse)
    if res_hyp:
        hp_proc = HypotenuseProcessor(res_hyp)
        hyp_data_by_scene = hp_proc.get_data_by_scene() 
        # => scene-XXXX -> [ {object_name, framestamp, hypotenuse}, ...]

    # Insert everything into the VCD
    inserter = VcdEventsActionsInserter(
        data_by_action=actions_data,
        data_by_event=events_data,
        data_by_relation=relations_data,
        data_by_hypotenuse=hyp_data_by_scene,
        thresholds=thresholds,
        input_dir=input_dir,
        output_dir=output_dir
    )
    inserter.run()
    print('[FIN] Proceso completado.')
