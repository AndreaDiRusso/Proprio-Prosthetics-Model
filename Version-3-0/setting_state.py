#!/usr/bin/env python3
"""
Example for how to modifying the MuJoCo qpos during execution.
"""

import os
from mujoco_py import load_model_from_xml, MjSim, MjViewer


with open('murdoc.xml', 'r') as f:
    MODEL_XML = f.read()


def print_box_xpos(sim):
    print("box xpos:", sim.data.get_body_xpos("box"))


model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)

states = [{'box:x': +0.8, 'box:y': +0.8},
          {'box:x': -0.8, 'box:y': +0.8},
          {'box:x': -0.8, 'box:y': -0.8},
          {'box:x': +0.8, 'box:y': -0.8},
          {'box:x': +0.0, 'box:y': +0.0}]

# MjModel.joint_name2id returns the index of a joint in
# MjData.qpos.
x_joint_i = sim.model.get_joint_qpos_addr("box:x")
y_joint_i = sim.model.get_joint_qpos_addr("box:y")

print_box_xpos(sim)

while True:
    for state in states:
        sim_state = sim.get_state()
        #sim_state.qpos[x_joint_i] = state["box:x"]
        #sim_state.qpos[y_joint_i] = state["box:y"]
        #sim.set_state(sim_state)
        sim.forward()
        print("updated state to", state)
        #print_box_xpos(sim)
        viewer.render()

    if os.getenv('TESTING') is not None:
        break
