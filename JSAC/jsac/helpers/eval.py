import os
import sys
import jax
import flax
import numpy as np
import jax.numpy as jnp
import multiprocessing as mp
 
from jsac.helpers.logger import Logger
from jsac.helpers.utils import WrappedEnv
from jsac.algo.agent import sample_actions
from jsac.envs.rl_chemist.env import RLC_Env
from jsac.algo.initializers import init_inference_actor 
from jsac.envs.dmc_visual_env.dmc_env import DMCVisualEnv
from jsac.envs.mujoco_visual_env.mujoco_visual_env import MujocoVisualEnv


def start_eval_process(args, log_dir, eval_queue, num_eval_episodes, rlc_eval=True):
    eval_process = mp.Process(target=eval, args=(args, 
                                                 log_dir, 
                                                 eval_queue, 
                                                 num_eval_episodes,
                                                 rlc_eval))
    eval_process.start()
    return eval_process
    

def eval(args, log_dir, eval_queue, num_eval_episodes, rlc_eval):
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
        if rlc_eval:
            env_mode = "eval"
        else:
            env_mode = "train"
        step_time = None
        if args['step_time'] > 0:
            step_time = args['step_time']
        env = RLC_Env(args['env_name'], 
                          args['image_history'], 
                          args['image_width'], 
                          args['image_height'],  
                          mask_type=args['mask_type'],
                          env_mode=env_mode,
                          video_path=args['video_dir'],
                          step_time=step_time)
        env = WrappedEnv(env, 200)
        
    env_steps = int(args['env_steps'])
    
    if not rlc_eval:
        xtick = num_eval_episodes * 200
        logger = Logger(log_dir, xtick=xtick) 
        tag = 'train'
    else:
        logger = Logger(log_dir, eval=True) 
        tag = 'eval'
    rng = jax.random.PRNGKey(0)
    rng, actor = init_inference_actor(rng, 
                                      args['image_shape'],
                                      args['proprioception_shape'],
                                      args['net_params'],
                                      args['action_shape'][-1],
                                      args['spatial_softmax'],
                                      args['mode'],
                                      jnp.float32)
    
    best_return = -1e8
    best_actor_params_path = os.path.join(log_dir, 'best_actor_params.pkl') 
    params = None
    while True:
        data = eval_queue.get()
        if isinstance(data, str):
            if data == 'close':
                logger.close() 
                env.close()
                sys.exit()
        else:
            params = data
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
                                         params, 
                                         state, 
                                         args['mode'], 
                                         True)

            action = np.asarray(action).clip(-1, 1)
            state, reward, done, info = env.step(action)
            
            if done or 'truncated' in info:
                if info['return'] >= best_return:
                    best_return = info['return']
                    if os.path.exists(best_actor_params_path):
                        os.remove(best_actor_params_path)
                    with open(best_actor_params_path, 'wb') as f: 
                        f.write(flax.serialization.to_bytes(params))
                    # load saved params
                    # with open(best_actor_params_path, 'rb') as f:
                    #     params_loaded = flax.serialization.from_bytes(params, f.read())
                
                state = env.reset(create_vid=vid)
                info['tag'] = tag
                info['dump'] = True
                info['eval_step'] = step 
                logger.push(info)
                epi += 1
                
                # if step % env_steps == 0 and epi == 5:
                #     vid = False
        if 'sync' in args:
            eval_queue.put(1)
        logger.plot()