import sys
import jax
import numpy as np
import jax.numpy as jnp
import multiprocessing as mp
 
from jsac.helpers.logger import Logger
from jsac.helpers.utils import WrappedEnv
from jsac.algo.agent import sample_actions
from jsac.algo.initializers import init_inference_actor 
from jsac.envs.robohive_envs.ur104C import UR10_4C_ENV
from jsac.envs.dmc_visual_env.dmc_env import DMCVisualEnv
from jsac.envs.mujoco_visual_env.mujoco_visual_env import MujocoVisualEnv


def start_eval_process(args, log_dir, eval_queue, num_eval_episodes):
    eval_process = mp.Process(target=eval, args=(args, 
                                                 log_dir, 
                                                 eval_queue, 
                                                 num_eval_episodes))
    eval_process.start()
    return eval_process
    

def eval(args, log_dir, eval_queue, num_eval_episodes):
    if args['env_type'] == 'MUJOCO':
        env = MujocoVisualEnv(args['env_name'], 
                            args['mode'], 
                            args['seed'] + 42, 
                            args['image_history'], 
                            args['image_width'], 
                            args['image_height'])
        env = WrappedEnv(env)
    elif args['env_type'] == 'DMC':
        env = DMCVisualEnv(args['env_name'], 
                           args['mode'], 
                           args['seed'] + 42, 
                           args['image_history'], 
                           args['image_width'], 
                           args['image_height'], 
                           args['num_cameras'], 
                           args['action_repeat'])
        env = WrappedEnv(env)
    elif args['env_type'] == 'RLC':
        env = UR10_4C_ENV(args['env_name'], 
                          args['image_history'], 
                          args['image_width'], 
                          args['image_height'], 
                          eval_mode=True,
                          video_path=args['video_dir'])
        env = WrappedEnv(env, 200)
        
    env_steps = int(args['env_steps'])
    
    logger = Logger(log_dir, eval=True) 
    rng = jax.random.PRNGKey(0)
    rng, actor = init_inference_actor(rng, 
                                      args['image_shape'],
                                      args['proprioception_shape'],
                                      args['net_params'],
                                      args['action_shape'][-1],
                                      args['resnet'],
                                      args['spatial_softmax'],
                                      args['mode'],
                                      jnp.float32)
    
    variables = None
    while True:
        data = eval_queue.get()
        if isinstance(data, str):
            if data == 'close':
                logger.close() 
                sys.exit()
        else:
            variables = data
            step = int(eval_queue.get())
        
        epi = 0
        
        vid = False
        if args['env_type'] == 'RLC':
            if step % env_steps == 0:
                vid = True
        
        state = env.reset(create_vid=vid)
        while epi < num_eval_episodes: 
            rng, action = sample_actions(rng, 
                                         actor.apply, 
                                         variables, 
                                         state, 
                                         args['mode'], 
                                         True)

            action = np.asarray(action).clip(-1, 1)
            state, reward, done, info = env.step(action)
            
            if done or 'truncated' in info:
                state = env.reset(create_vid=vid)
                info['tag'] = 'eval'
                info['dump'] = True
                info['eval_step'] = step 
                logger.push(info)
                epi += 1
                
                # if step % env_steps == 0 and epi == 5:
                #     vid = False
                
        logger.plot()