# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction
from typing import List

import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
import glfw
from collections import deque
import enum
from functools import partial

xml = './mj_envs/robohive/envs/arms/ur10e/scene_chem_vel.xml' 
contacts = deque(maxlen=1000)

class ObjLabels(enum.Enum):
    LEFT_GRIP = 0
    RIGHT_GRIP = 1
    ENV = 2
    GOAL = 3

class BodyIdInfo:
    def __init__(self, model: mujoco.MjModel):
        self.manip_body_id = model.body("rbf").id

        left_bodies = [model.body(i).id for i in range(model.nbody) if model.body(i).name.startswith("Lgrip/")]
        self.left_range = (min(left_bodies), max(left_bodies))

        right_bodies = [model.body(i).id for i in range(model.nbody) if model.body(i).name.startswith("Rgrip/")]
        self.right_range = (min(right_bodies), max(right_bodies))

        self.goal_id = model.body("busbin2").id

def arm_control(model: mujoco.MjModel, data: mujoco.MjData, id_info: BodyIdInfo):
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system
    touching_objects = set(get_touching_objects(model, data, id_info))
    contacts.append(touching_objects)
    return touching_objects

def get_touching_objects(model: mujoco.MjModel, data: mujoco.MjData, id_info: BodyIdInfo):
    for con in data.contact:
        if model.geom(con.geom1).bodyid == id_info.manip_body_id:
            yield body_id_to_label(model.geom(con.geom2).bodyid, id_info)
        elif model.geom(con.geom2).bodyid == id_info.manip_body_id:
            yield body_id_to_label(model.geom(con.geom1).bodyid, id_info)

def body_id_to_label(body_id, id_info: BodyIdInfo):
    #print(body_id, id_info.left_range[0], id_info.left_range[1], id_info.right_range[0], id_info.right_range[1])
    if id_info.left_range[0] - 1 <= body_id < id_info.left_range[1]:
        return ObjLabels.LEFT_GRIP
    elif id_info.right_range[0] -1 <= body_id < id_info.right_range[1]:
        return ObjLabels.RIGHT_GRIP
    elif body_id == id_info.goal_id:
        return ObjLabels.GOAL
    else:
        return ObjLabels.ENV

def load_callback(model=None, data=None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)

    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)

    id_info = BodyIdInfo(model)

    if model is not None:
        # Can set initial state

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(partial(arm_control, id_info=id_info))

    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)