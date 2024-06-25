import numpy as np
import mujoco 
import cv2
import os


def create_vid(images):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('vid_vb.mp4', fourcc, 100, (width, height))
    for image in images:
        video.write(image)
    video.release()


def check_collision(model, data):
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -0.6, 1.2
    z_min, z_max = 0.73, 1.78
    i = 1
    print(model.njnt)
    while i < model.njnt:
        joint_frame_id = model.jnt_bodyid[i]
        joint_pos = data.xpos[joint_frame_id]  # Get the x, y, z position of the joint
        # Check if the joint position is within the defined boundaries
        if not (x_min <= joint_pos[0] <= x_max and y_min <= joint_pos[1] <= y_max and z_min <= joint_pos[2] <= z_max):
            print(joint_pos)
            print(f"Joint {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)} is out of bounds!")
            # Implement additional logic here, e.g., stop simulation, adjust joint, etc.
            return True
        i += 1
    return False


def main():
    ## Setup
    images = []
    height = 880
    width = 1080

    model_path = './mj_envs/robohive/envs/arms/ur10e/scene_chem.xml' 
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    #boundary_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "boundary_geom")
    print('body id', mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'p1'))

    renderer = mujoco.Renderer(model, height=height, width=width)

    ## Move to grabbing position
    kf = model.keyframe('home_3')
    data.qpos = kf.qpos
    mujoco.mj_forward(model, data)


    renderer.update_scene(data, camera='right_cam')
    images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    ## Move to grabbing position
    ctrl = kf.qpos[:7]
    ctrl[6] = 1
    for i in range(200):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera='right_cam')
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        #check_collision(model, data)


    ## Lift up 
    ctrl = kf.qpos[:7] 
    ctrl[3] = 0
    ctrl[2] = 0
    ctrl[6] = 1
    for i in range(300):
        saved_qpos = np.copy(data.qpos)
        saved_qvel = np.copy(data.qvel)

        data.ctrl = ctrl
        mujoco.mj_step(model, data, nstep=1)
        renderer.update_scene(data, camera='right_cam')
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        if check_collision(model, data):
            data.qpos = saved_qpos
            data.qvel = saved_qvel
            mujoco.mj_forward(model, data)
            #break

    create_vid(images)


if __name__ == '__main__':
    main()


