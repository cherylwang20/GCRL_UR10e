import numpy as np
import mujoco 
import cv2
import os

from python_api import BodyIdInfo, arm_control, get_touching_objects, ObjLabels


def create_vid(images):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('vid.mp4', fourcc, 200, (width, height))
    for image in images:
        video.write(image)
    video.release()


def main():
    ## Setup
    images = []
    height = 480
    width = 680
    camera_id = 'ft_cam'

    model_path = './mj_envs/robohive/envs/arms/ur10e/scene_chem_vel.xml' 
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    ## Move to grabbing position
    kf = model.keyframe('home_1')
    data.qpos = kf.qpos
    mujoco.mj_forward(model, data)
    id_info = BodyIdInfo(model)

    renderer.update_scene(data, camera=camera_id)
    images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    
    print(set(get_touching_objects(model, data, id_info)))

    ## Move to grabbing position
    ctrl = [1.45, -2.6, -0.865, -1.19, 1.57, 0, 0]
    #ctrl[6] = 1
    for i in range(200):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    
    print(set(get_touching_objects(model, data, id_info)))

    ctrl[6] = 1
    for i in range(150):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    
    print(set(get_touching_objects(model, data, id_info)))
    

    ctrl = [1.51, -2.16, -0.825, -1.52, 1.57, 0, 1]
    for i in range(200):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    
    print('grabbing', set(get_touching_objects(model, data, id_info)))

    ctrl = [2.14, -2.14, -0.88, -1.57, 1.57, 0.754, 1]
    for i in range(200):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    
    print(set(get_touching_objects(model, data, id_info)))
    
    ctrl = [2.07, -2.51, -0.88, -1.38, 1.57, 0.503, 1]
    for i in range(200):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    
    print(set(get_touching_objects(model, data, id_info)))
    
    ctrl = [2.07, -2.51, -0.88, -1.38, 1.57, 0.503, 0]
    for i in range(200):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    print(set(get_touching_objects(model, data, id_info)))

    ctrl = [2, -1.76, -0.785, -2.15, 1.57, 0., 0]
    for i in range(200):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    print(set(get_touching_objects(model, data, id_info)))
    
    ctrl = [1.26, -2.07, -1.73, -0.318, 1.57, 0, 0]
    for i in range(200):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    print(set(get_touching_objects(model, data, id_info)))

    ctrl = [1.26, -2.26, -1.7, 0.41, 1.57, 0, 0]
    for i in range(100):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    print(set(get_touching_objects(model, data, id_info)))

    ctrl = [1.26, -2.26, -1.7, 0.41, 1.57, 0, 1]
    for i in range(200):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    print(set(get_touching_objects(model, data, id_info)))

    ctrl = [1.51, -2.06, -0.825, -1.52, 1.57, 0, 1]
    scaled_ctrl = [c / 3 for c in ctrl]
    scaled_ctrl[-1] = 1

    for i in range(300):
        data.ctrl = scaled_ctrl if i % 2 == 0 else ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    print(set(get_touching_objects(model, data, id_info)))

    ctrl = [2.23, -2.06, -0.825, -1.52, 1.57, 0, 1]

    for i in range(800):
        if i % 2:
            data.ctrl = ctrl
        else:
            data.ctrl = data.qpos[:7]
            data.ctrl[-1] = 1
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    print(set(get_touching_objects(model, data, id_info)))

    ctrl = [2.26, -2, -1.35, -1.13, 1.57, 0, 1]
    for i in range(400):
        if i % 2:
            data.ctrl = ctrl
        else:
            data.ctrl = data.qpos[:7]
            data.ctrl[-1] = 1
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    print(set(get_touching_objects(model, data, id_info)))
    
    ctrl = [2.26, -2.14, -1.32, -1.13, 1.57, 0, 0]
    for i in range(300):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    print(set(get_touching_objects(model, data, id_info)))

    ctrl = [1.26, -1.7, -1.73, -0.318, 1.57, 0, 0]
    for i in range(300):
        data.ctrl = ctrl
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    print(set(get_touching_objects(model, data, id_info)))
    
    create_vid(images)


if __name__ == '__main__':
    main()