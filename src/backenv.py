from __future__ import absolute_import

import numpy as np
import os
import sys
import math
import xml.dom.minidom
import traci
import sumolib
from gymnasium import spaces
import gymnasium as gym
import uuid
from util import copy_files, handle_exception, normal_val
import subprocess
import re
from gaodb import gaodb
import shutil


def get_zone_index(angle, angle_boundaries):
    for i, (start, end) in enumerate(angle_boundaries):
        if start <= angle < end:
            return i
    return len(angle_boundaries) - 1


def get_lane_pos(vid):
    lane_index = traci.vehicle.getLaneIndex(vid)
    eid = traci.vehicle.getRoadID(vid)
    lid = traci.vehicle.getLaneID(vid)
    lane_width = traci.lane.getWidth(lid)
    lat = traci.vehicle.getLateralLanePosition(vid)
    lane_num = traci.edge.getLaneNumber(eid)
    res = ((lane_index + 1) * lane_width + lat) / (lane_num * lane_width)
    return res


class Highway_env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 90}

    def __init__(
        self,
        env="merge",
        start_edge="E2",
        end_junc="J3",
        end_edge="E3.123",
        gui=False,
        time_limit=300 * 30,
        task_name="test",
    ):
        super(Highway_env, self).__init__()
        self.ego_id = "Auto"
        self.detect_range = 100.0
        self.end_junc = end_junc
        self.end_edge = end_edge
        self.start_edge = start_edge
        self.time_limit = time_limit
        self.start_vol = []

        self.max_acc = None
        self.max_lat_v = None
        self.maxSpeed = None
        self.max_angle = None
        self.x_goal, self.y_goal = None, None
        self.max_dis_navigation = None
        self.reset_times = 0
        self.work_id, self.work_dir = self._init_work_space(env_name=env)
        self.config_path = self.work_dir + "/highway.sumocfg"
        self.env_name = env
        self.gui = gui

        self.end_road = end_junc
        self.angle_boundaries = [
            (0.0, 60.0),  # 0~60
            (60.0, 120.0),  # 60~120
            (120.0, 180.0),  # 120~180
            (-180.0, -120.0),  # -180~-120
            (-120.0, -60.0),  # -120~-60
            (-60.0, 0.0),  # -60~0
        ]

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(29,), dtype=np.float32
        )
        self.start_sumo(gui=gui)
        self.gaodb = gaodb.get()
        self._step = 0
        self.task_level = 10
        self.consecutive_finish = 0

    def _init_work_space(self, env_name):
        """Create tmp work space for sumo"""
        task_id = uuid.uuid4()
        work_dir = f"../tmp/{task_id}"

        os.mkdir(work_dir)
        files_to_copy = [
            "background.jpg",
            "stop.xml",
            "background.xml",
            "highway.net.xml",
            "highway.sumocfg",
            "autoGenTraffic.sh",
            "bus.trips.xml",
            "bus.config.txt",
            "car.config.txt",
            "car.trips.xml",
            "vTypeDistributions.add.xml",
            "auto.trips.xml",
        ]
        copy_files(files_to_copy, f"../env/{env_name}", work_dir)
        script_path = os.path.join(work_dir, "autoGenTraffic.sh")
        with open(script_path, "r") as file:
            script_content = file.read()
            self._extract_starting_volumes(script_content)
        return task_id, work_dir

    def render(self, mode="human"):
        pass

    def raw_obs(self, veh_list):  # dimension: 24+5
        obs = []
        if self.ego_id in veh_list:
            obs_space = [[[], [], []] for _ in range(6)]  # 3*6

            ego_x, ego_y = traci.vehicle.getPosition(self.ego_id)
            dis_goal_ego = np.linalg.norm(
                np.array([self.x_goal - ego_x, self.y_goal - ego_y])
            )

            for vid in veh_list:
                veh_x, veh_y = traci.vehicle.getPosition(vid)
                dis2veh = np.linalg.norm(np.array([veh_x - ego_x, veh_y - ego_y]))

                if vid != self.ego_id and dis2veh < self.detect_range:
                    angle2veh = math.degrees(math.atan2(veh_y - ego_y, veh_x - ego_x))

                    obs_direction_index = get_zone_index(
                        angle2veh, self.angle_boundaries
                    )

                    obs_space[obs_direction_index][0].append(vid)
                    obs_space[obs_direction_index][1].append(dis2veh)
                    obs_space[obs_direction_index][2].append(angle2veh)

            for direction_space in obs_space:
                if len(direction_space[0]) == 0:
                    obs.append(self.detect_range)
                    obs.append(0.0)
                    obs.append(0.0)
                    obs.append(0.0)
                else:
                    mindis_v_index = direction_space[1].index(min(direction_space[1]))
                    obs.append(min(direction_space[1]))
                    obs.append(direction_space[2][mindis_v_index])
                    obs.append(
                        traci.vehicle.getSpeed(direction_space[0][mindis_v_index])
                    )
                    obs.append(
                        traci.vehicle.getAngle(direction_space[0][mindis_v_index])
                    )

            obs.append(traci.vehicle.getSpeed(self.ego_id))
            obs.append(traci.vehicle.getAngle(self.ego_id))
            obs.append(get_lane_pos(self.ego_id))
            obs.append(traci.vehicle.getLateralSpeed(self.ego_id))
            obs.append(dis_goal_ego)
            pos = [ego_x, ego_y]

        else:
            zeros = [0.0] * 3
            detect_range_repeat = [self.detect_range] + zeros
            obs = detect_range_repeat * 6 + [
                0.0,  # ego speed
                0.0,  # ego angle
                0.0,  # ego pos lateral
                0.0,  # ego speed lateral
                self.max_dis_navigation,
            ]
            pos = [0.0, 0.0]

        return obs, pos

    def norm_obs(self, veh_list):
        obs, pos = self.raw_obs(veh_list)
        state = []
        for i in range(6):  # 6 directions
            base_index = i * 4
            state.extend(
                [
                    obs[base_index] / self.detect_range,
                    obs[base_index + 1] / self.max_angle,
                    obs[base_index + 2] / self.maxSpeed,
                    obs[base_index + 3] / self.max_angle,
                ]
            )
        # Adding the last specific elements
        state.extend(
            [
                obs[24] / self.maxSpeed,  # ego speed
                obs[25] / self.max_angle,  # ego angle
                obs[26] / self.detect_range,  # ego pos lateral
                obs[27] / self.max_lat_v,  # ego speed lateral
                obs[28] / self.max_dis_navigation,
            ]
        )

        return state, pos

    def get_reward(self, veh_list):
        cost = 0.0
        time_step_limit = self.time_limit
        overtime_check = False
        navigation_check = False
        idle_step_limit = 600
        idle_dis_threshold = 10
        idle_check = False

        obs, _ = self.raw_obs(veh_list)

        dis_front_right = obs[0]
        dis_front = obs[4]
        dis_front_left = obs[8]
        dis_rear_left = obs[12]
        dis_rear = obs[16]
        dis_rear_right = obs[20]
        dis_sides = [
            dis_front_right,
            dis_front_left,
            dis_rear_left,
            dis_rear_right,
            dis_front,
            dis_rear,
        ]

        v_ego = obs[24]
        ego_lat_pos = obs[26]
        ego_lat_v = obs[27]
        dis_goal_ego = obs[28]

        collision_check = self.check_collision()

        min_side_dist = min(dis_sides)
        cost = (
            -normal_val(abs(min_side_dist / self.detect_range))
            + abs(ego_lat_v / self.max_lat_v) / 100
            + abs(ego_lat_pos / 5) / 100
        )

        if collision_check:
            cost = math.sqrt(ego_lat_v**2 + v_ego**2)

        if self._step % idle_step_limit == 0:
            if (
                abs(self.last_travel_dis - abs(dis_goal_ego - self.max_dis_navigation))
                < idle_dis_threshold
            ):
                idle_check = True
                cost += 1
                print(">>>>>>>>>>>>>>>>>>>>>>>>> Idle too long:", self._step, "\n")
            self.last_travel_dis = abs(dis_goal_ego - self.max_dis_navigation)

        if dis_goal_ego < 15.0:
            navigation_check = True
            navigation_precent = 1
            self.consecutive_finish += 1
            print("=========================== Finish!" + "\n")
        else:
            navigation_precent = 1 - dis_goal_ego / self.max_dis_navigation

        if self._step > time_step_limit:
            overtime_check = True
            print("+++++++++++++++++++++> over time:", navigation_precent, "\n")

        speed_reward = v_ego / self.maxSpeed
        self.mean_speed += speed_reward
        self.total_reward += speed_reward - cost + navigation_precent
        return (
            speed_reward - cost + navigation_precent,
            collision_check,
            navigation_precent,
            overtime_check,
            navigation_check,
            idle_check,
        )

    def check_collision(self):
        collision_check = False
        vlist = traci.simulation.getCollidingVehiclesIDList()
        if self.ego_id in vlist:
            collision_check = True
            print("====================> Collision!")
        return collision_check

    def close(self):
        traci.close()
        if os.path.exists(f"../tmp/{self.work_id}"):
            shutil.rmtree(f"../tmp/{self.work_id}")

    def step(self, action_a):
        try:
            acc, lane_change = action_a[0].item(), action_a[1].item()
            control_acc = self.max_acc * acc

            traci.vehicle.changeSublane(self.ego_id, lane_change)

            traci.vehicle.setAcceleration(self.ego_id, control_acc, duration=0.03)
            traci.simulationStep()
            self._step += 1
            if self._step % (100 * 30) == 0:
                print("Step time:", int(self._step) / 30)

            veh_list = traci.vehicle.getIDList()

            (
                reward,
                collision_check,
                navigation_precent,
                overtime_chceck,
                navigation_check,
                idle_check,
            ) = self.get_reward(veh_list)
            next_state, pos = self.norm_obs(veh_list)

            terminated = navigation_check or collision_check
            truncated = overtime_chceck or idle_check

            info = {
                "collision_check": collision_check,
                "mean_speed": self.mean_speed / self._step,
                "idle_check": idle_check,
                "time_step": self._step,
                "navigation": navigation_precent,
                "position": pos,
                "task_level": self.task_level,
                "reward": reward,
                "total_reward": self.total_reward / self._step,
            }
            if terminated or truncated:
                self.consecutive_finish = 0
                self.gaodb.log(info)

            return next_state, reward, terminated, truncated, info
        except Exception as e:
            handle_exception(e)

    def reset(self, seed=None, options=None):
        try:
            self._initialize_reset_vals()
            self._create_traffic_config()
            self._update_config()
            self._load_simulation()
            self._wait_for_auto_car()
            VehicleIds = traci.vehicle.getIDList()
            initial_state, info = self.norm_obs(VehicleIds)
            return initial_state, info
        except Exception as e:
            handle_exception(e)

    def _initialize_reset_vals(self):
        """Initialize variables and parse config."""
        self._step = 0
        self.last_travel_dis = 20
        self.dom = xml.dom.minidom.parse(self.config_path)
        self.root = self.dom.documentElement
        self.mean_speed = 0
        self.total_reward = 0

    def _create_traffic_config(self):
        """Handles trip XML file creation or updating."""
        config_dir = os.path.dirname(self.config_path)
        trip_path = os.path.join(config_dir, "auto.trips.xml")

        if not os.path.exists(trip_path):
            trip_dom = xml.dom.minidom.Document()
            trips_element = trip_dom.createElement("trips")
            trip_dom.appendChild(trips_element)
        else:
            trip_dom = xml.dom.minidom.parse(trip_path)
            trips_element = trip_dom.documentElement

        if not self._auto_trip_exists(trips_element):
            self._create_traffic(trip_dom, trips_element)
        else:
            self._insert_auto_trip(trips_element)
        self._save_trip_file(trip_dom, trip_path)

        if self.consecutive_finish >= 10 or self.reset_times == 0:
            os.chdir(self.work_dir)
            script_path = os.path.join(os.getcwd(), "autoGenTraffic.sh")
            if not os.access(script_path, os.X_OK):
                os.chmod(script_path, 0o755)
            if self.task_level < 1:
                return
            new_volumes = [
                float(start_vol) * (self.task_level + 1) for start_vol in self.start_vol
            ]
            self.task_level -= 1
            self.consecutive_finish = 0
            print("new vol:", new_volumes)
            with open(script_path, "r") as file:
                content = file.read()
                matches = re.findall(r"-p\s+\d+\.\d+", content)

                if len(matches) >= 2:
                    content = content.replace(matches[0], f"-p {new_volumes[0]}", 1)
                    content = content.replace(matches[1], f"-p {new_volumes[1]}", 1)
                with open(script_path, "w") as file:
                    file.write(content)

            try:
                result = subprocess.run(
                    [script_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.stderr:
                    print("Generate Traffic Error Output:")
                    print(result.stderr)
            except subprocess.CalledProcessError as e:
                handle_exception(e)
            finally:
                os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def _auto_trip_exists(self, trips_element):
        """Check if the 'Auto' trip already exists."""
        trip_elements = trips_element.getElementsByTagName("trip")
        return any(trip.getAttribute("id") == "Auto" for trip in trip_elements)

    def _create_traffic(self, trip_dom, trips_element):
        """Create a new 'Auto' trip element."""
        attributes = {
            "id": "Auto",
            "depart": "20",
            "departLane": "best",
            "departSpeed": "2.00",
            "color": "red",
            "from": self.start_edge,
            "to": self.end_edge,
        }
        new_trip_element = trip_dom.createElement("trip")
        for key, value in attributes.items():
            new_trip_element.setAttribute(key, value)
        trips_element.insertBefore(new_trip_element, trips_element.firstChild)

    def _insert_auto_trip(self, trips_element):
        """Update the existing 'Auto' trip element."""
        trip_elements = trips_element.getElementsByTagName("trip")
        auto_trip = next(
            trip for trip in trip_elements if trip.getAttribute("id") == "Auto"
        )
        attributes = {
            "id": "Auto",
            "depart": "20",
            "departLane": "best",
            "departSpeed": "2.00",
            "color": "red",
            "from": self.start_edge,
            "to": self.end_edge,
        }
        for key, value in attributes.items():
            auto_trip.setAttribute(key, value)

    def _save_trip_file(self, trip_dom, trip_path):
        """Save the updated trips XML file."""
        with open(trip_path, "w") as trip_file:
            trip_dom.writexml(trip_file)

    def _update_config(self):
        """Update and save the configuration with the new seed value."""
        seed = np.random.randint(0, 10000)
        random_seed_element = self.root.getElementsByTagName("seed")[0]
        super().reset(seed=seed)
        try:
            if self.reset_times % 2 == 0:
                random_seed_element.setAttribute("value", str(seed))

            with open(self.config_path, "w") as file:
                self.dom.writexml(file)
        except Exception as e:
            handle_exception(e)

    def _load_simulation(self):
        """Load the simulation with the new configuration."""
        traci.load(["-c", self.config_path])
        print(f"============= Resetting the env {self.reset_times}============")
        self.reset_times += 1

    def _extract_starting_volumes(self, script_content):
        # Regex to find all instances of -p followed by a number
        pattern = re.compile(r"-p (\d+\.\d+)")
        matches = pattern.findall(script_content)
        self.start_vol.extend(matches)

    def _wait_for_auto_car(self):
        """Wait for the 'Auto' car to appear in the simulation."""
        AutoCarAvailable = False
        while not AutoCarAvailable:
            traci.simulationStep()
            VehicleIds = traci.vehicle.getIDList()
            if self.ego_id in VehicleIds:
                AutoCarAvailable = True
                if self.gui:
                    traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
                    traci.gui.trackVehicle(traci.gui.DEFAULT_VIEW, self.ego_id)
                traci.vehicle.setSpeedFactor(self.ego_id, 3)
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)
                traci.vehicle.setSpeedMode(self.ego_id, 0)

                self.maxSpeed = traci.vehicle.getMaxSpeed(self.ego_id)
                self.max_angle = 360.0
                self.x_goal, self.y_goal = traci.junction.getPosition(self.end_junc)
                self.max_dis_navigation = sum(
                    traci.lane.getLength(v + "_0")
                    for v in traci.vehicle.getRoute(self.ego_id)
                )
                self.max_acc = traci.vehicle.getAccel(self.ego_id)
                self.max_lat_v = traci.vehicle.getMaxSpeedLat(self.ego_id)

    def start_sumo(
        self,
        gui=False,
    ):

        sumoBinary = "sumo-gui" if gui else "sumo"
        traci.start(
            [sumoBinary, "-c", self.config_path, "--collision.check-junctions", "true"]
        )
