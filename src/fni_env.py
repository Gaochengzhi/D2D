from __future__ import absolute_import
from __future__ import print_function

import gym
import numpy as np
import os
import sys
import math
import xml.dom.minidom
import traci
import sumolib


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

    def __init__(self, env="merge", start_edge="E2", end_junc="J3", end_edge="E3.123"):
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

        self.end_road = end_junc
        self.angle_boundaries = [
            (0.0, 60.0),  # 0~60
            (60.0, 120.0),  # 60~120
            (120.0, 180.0),  # 120~180
            (-180.0, -120.0),  # -180~-120
            (-120.0, -60.0),  # -120~-60
            (-60.0, 0.0),  # -60~0
        ]

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
            info = [ego_x, ego_y]

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
            info = [0.0, 0.0]

        return obs, info

    def norm_obs(self, veh_list):
        obs, info = self.raw_obs(veh_list)
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

        return state, info

    def get_reward(self, veh_list):
        cost = 0.0
        infraction = 0.0
        infraction_check = False
        navigation_check = False
        done = False

        raw_obs, _ = self.raw_obs(veh_list)
        dis_fr = raw_obs[0]
        dis_f = raw_obs[4]
        dis_fl = raw_obs[8]
        dis_rl = raw_obs[12]
        dis_r = raw_obs[16]
        dis_rr = raw_obs[20]
        dis_sides = [dis_fr, dis_fl, dis_rl, dis_rr]
        v_ego = raw_obs[24]
        ego_lat_pos = raw_obs[26]
        ego_lat_v = raw_obs[27]
        dis_goal_ego = raw_obs[28]

        reward = v_ego / 5.0

        collision_value = self.check_collision(dis_f, dis_r, dis_sides, veh_list)

        if collision_value == True:
            cost = 10.0
            done = True


        if self._step > 600 * 30:
            navigation = 100.0
            navigation_check = True
            done = True
            print(">>>>>> Stuck")
        else:
            navigation = -np.log(1.0 + dis_goal_ego / self.max_dis_navigation) - 1.0

        return (
            reward - cost - infraction + navigation,
            collision_value,
            cost,
            False,
            infraction,
            navigation_check,
            done,
        )

    def check_collision(self, dis_f, dis_r, dis_sides, vhe_list):
        collision_value = False
        vlist = traci.simulation.getCollidingVehiclesIDList()
        if self.ego_id in vlist:
            collision_value = True
            print("===>Checker-0: Collision!\n")
        return collision_value

    def step(self, action_a):
        acc, lane_change = action_a[0].item(), action_a[1].item()
        control_acc = self.max_acc * acc

        traci.vehicle.changeSublane(self.ego_id, lane_change)

        traci.vehicle.setAcceleration(self.ego_id, control_acc, duration=0.03)
        traci.simulationStep()
        self._step += 1

        veh_list = traci.vehicle.getIDList()

        (
            reward_cost,
            collision_value,
            cost,
            infraction_check,
            infraction,
            navigation_check,
            done,
        ) = self.get_reward(veh_list)
        next_state, info = self.norm_obs(veh_list)

        return (
            reward_cost,
            next_state,
            collision_value,
            cost,
            infraction_check,
            infraction,
            navigation_check,
            done,
            info,
        )

    def reset(self):
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
            "depart": "30",
            "departLane": "best",
            "departSpeed": "5.00",
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

        if self.reset_times % 2 == 0:
            random_seed = "%d" % self.reset_times
            random_seed_element.setAttribute("value", random_seed)

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

        initial_state, _ = self.norm_obs(VehicleIds)

        return initial_state

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
