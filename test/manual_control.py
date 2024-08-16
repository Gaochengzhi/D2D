import tkinter as tk
import os
import sys
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
from run_env import Highway_env


class SteeringControlApp:
    def __init__(self, root, env):
        self.root = root
        self.env = env

        self.acc = 0.0
        self.steer = 0.0

        self.create_widgets()
        self.bind_keys()

        self.update()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()

    def bind_keys(self):
        self.root.bind("<Up>", self.increase_acc)
        self.root.bind("<Down>", self.decrease_acc)
        self.root.bind("<Left>", self.steer_left)
        self.root.bind("<Right>", self.steer_right)
        self.root.bind("<KeyRelease>", self.reset_steering)

    def increase_acc(self, event):
        self.acc = min(self.acc + 1, 1.0)

    def decrease_acc(self, event):
        self.acc = max(self.acc - 1, -1.0)

    def steer_left(self, event):
        self.steer += 0.8

    def steer_right(self, event):
        self.steer += -0.8

    def reset_steering(self, event):
        self.steer = 0.0

    def update(self):
        action = np.array([self.acc, self.steer])

        # Step environment with the current action
        (
            reward_cost,
            next_state,
            collision_value,
            cost,
            infraction_check,
            infraction,
            navigation_check,
            done,
            info,
        ) = self.env.step(action)
        # self.profiler.print()

        # Update UI or handle simulation step here
        # For now, just print the current status

        # Schedule the update method to be called again after 100 ms
        self.root.after(1, self.update)


def test_handle_control():
    env = Highway_env()
    env.start(gui=True)
    s = env.reset()

    root = tk.Tk()
    app = SteeringControlApp(root, env)

    root.mainloop()


if __name__ == "__main__":
    test_handle_control()
