from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import sys
import math
import xml.dom.minidom
import traci
import sumolib
from gymnasium import spaces
import gymnasium as gym


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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self, env="merge", start_edge="E2", end_junc="J3", end_edge="E3.123", gui=False
    ):
        super(Highway_env, self).__init__()
        self.ego_id = "Auto"
        self.detect_range = 150.0
        self.end_junc = end_junc
        self.end_edge = end_edge
        self.start_edge = start_edge

        self.max_acc = None
        self.max_lat_v = None
        self.maxSpeed = None
        self.max_angle = None
        self.x_goal, self.y_goal = None, None
        self.max_dis_navigation = None
        self.reset_times = 0
        self.config_path = f"../env/{env}/highway.sumocfg"
        self.env_name = env
        self._step = 0
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
        self.start(gui=gui)

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
        time_step_limit = 300 * 30
        idle_step_limit = 900
        idle_dis_threshold = 0.3  
        overtime_check = False
        idle_check = False
        navigation_check = False
        done = False

        raw_obs, _ = self.raw_obs(veh_list)

        
        dis_front_right = raw_obs[0]
        dis_front = raw_obs[4]
        dis_front_left = raw_obs[8]
        dis_rear_left = raw_obs[12]
        dis_rear = raw_obs[16]
        dis_rear_right = raw_obs[20]
        dis_sides = [dis_front_right, dis_front_left, dis_rear_left, dis_rear_right]

        v_ego = raw_obs[24]  # 自车速度
        ego_lat_pos = raw_obs[26]  # 自车横向位置
        ego_lat_v = raw_obs[27]  # 自车横向速度
        dis_goal_ego = raw_obs[28]  

        speed_reward = v_ego / 5.0

        collision_check = self.check_collision()

        if collision_check:
            cost = 5.0
            done = True

        
        if hasattr(self, 'position_history'):
            if self._step % idle_step_limit ==0:
                self.position_history = abs( dis_goal_ego -self.position_history)
        else:
           self.position_history = dis_goal_ego

        
        
        if self.position_history < idle_dis_threshold:
                idle_check = True
                done = True
                print(">>> Idle too long:", self._step)
        
        if dis_goal_ego < 15.0:
            navigation_check = True
            navigation_precent = 100
            done = True
            print(">>>>>> Finish!")
        else:
            navigation_precent = -np.log(1.0 + dis_goal_ego / self.max_dis_navigation) - 1.0


        if self._step > time_step_limit:
            overtime_check = True
            done = True
            print("+++> over time:", navigation_precent)

        return (
            speed_reward - cost + navigation_precent,
            collision_check,
            speed_reward,
            self._step,
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
            print("===>Checker-0: Collision!")
        return collision_check

    def step(self, action_a):
        acc, lane_change = action_a[0].item(), action_a[1].item()
        control_acc = self.max_acc * acc

        traci.vehicle.changeSublane(self.ego_id, lane_change)

        traci.vehicle.setAcceleration(self.ego_id, control_acc, duration=0.03)
        traci.simulationStep()
        self._step += 1

        veh_list = traci.vehicle.getIDList()

        (
            reward,
            collision_check,
            speed_reward,
            time_step,
            navigation_precent,
            overtime_chceck,
            navigation_check,
            idle_check,
        ) = self.get_reward(veh_list)
        next_state, pos = self.norm_obs(veh_list)

        terminated =  collision_check or navigation_check 
        truncated = overtime_chceck or idle_check
        info = {
            "collision": collision_check,
            "speed_reward": speed_reward,
            "time_step": time_step,
            "navigation": navigation_precent,
            "position": pos,
            "reward": reward
        }

        return next_state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        dom = xml.dom.minidom.parse(self.config_path)
        root = dom.documentElement

        config_dir = os.path.dirname(self.config_path)
        trip_path = os.path.join(config_dir, "auto.trips.xml")
        # Check if car.trips.xml exists and load its content
        if not os.path.exists(trip_path):
            trip_dom = xml.dom.minidom.Document()
            trips_element = trip_dom.createElement("trips")
            trip_dom.appendChild(trips_element)
        else:
            trip_dom = xml.dom.minidom.parse(trip_path)
            trips_element = trip_dom.documentElement

        # Check if the trip with id "Auto" already exists
        trip_elements = trips_element.getElementsByTagName("trip")
        auto_trip_exists = any(
            trip.getAttribute("id") == "Auto" for trip in trip_elements
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

        if not auto_trip_exists:
            # Create a new trip element
            new_trip_element = trip_dom.createElement("trip")
            for key, value in attributes.items():
                new_trip_element.setAttribute(key, value)
            trips_element.insertBefore(new_trip_element, trips_element.firstChild)
        else:
            # Update the existing "Auto" trip
            auto_trip = next(
                trip for trip in trip_elements if trip.getAttribute("id") == "Auto"
            )
            for key, value in attributes.items():
                auto_trip.setAttribute(key, value)

        # Save the updated XML file
        with open(trip_path, "w") as trip_file:
            trip_dom.writexml(trip_file)
        random_seed_element = root.getElementsByTagName("seed")[0]

        super().reset(seed=seed)
        if self.reset_times % 2 == 0:
            random_seed = "%d" % self.reset_times
            random_seed_element.setAttribute("value", seed)

        with open(self.config_path, "w") as file:
            dom.writexml(file)

        traci.load(["-c", self.config_path])
        print("Resetting the env", self.reset_times)
        self.reset_times += 1

        AutoCarAvailable = False
        while AutoCarAvailable == False:
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

        initial_state, info = self.norm_obs(VehicleIds)

        return initial_state, info

    def close(self):
        traci.close()

    def start(
        self,
        gui=False,
    ):
        sumoBinary = "sumo-gui" if gui else "sumo"
        traci.start(
            [sumoBinary, "-c", self.config_path, "--collision.check-junctions", "true"]
        )
