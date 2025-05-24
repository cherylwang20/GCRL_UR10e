import optax
import numpy as np
from jax import random
from jax import numpy as jnp
from flax.training.train_state import TrainState

from jsac.helpers.utils import MODE
from jsac.algo.models import ActorModel, CriticModel, Temperature


def get_init_data(init_image_shape, 
                  init_proprioception_shape, 
                  mode):
    init_image = None
    init_proprioception = None 

    if mode == MODE.IMG or mode == MODE.IMG_PROP:
        init_image = np.random.randint(
            0, 256, size=(1, *init_image_shape), dtype=np.uint8)
    if mode == MODE.PROP or mode == MODE.IMG_PROP:
        init_proprioception = np.random.uniform(
            size=(1, *init_proprioception_shape)).astype(np.float32)

    return init_image, init_proprioception
    

def init_critic(rng,
                learning_rate, 
                init_image_shape, 
                init_proprioception_shape, 
                net_params, 
                action_dim, 
                spatial_softmax,
                mode,
                dtype,
                num_critic_networks,
                global_norm):

    model = CriticModel(net_params, 
                        action_dim, 
                        spatial_softmax,  
                        mode,
                        dtype,
                        num_critic_networks)
    
    rng, key1, key2 = random.split(rng, 3)
    init_actions = random.uniform(key1, (1, action_dim), dtype=jnp.float32)

    init_image, init_proprioception = get_init_data(
        init_image_shape, 
        init_proprioception_shape, 
        mode)
    
    params = model.init(key2, 
                        init_image, 
                        init_proprioception, 
                        init_actions)['params']

    # tx = optax.adam(learning_rate=learning_rate, mu_dtype=dtype)
    tx=optax.chain(optax.zero_nans(), 
                   optax.clip_by_global_norm(global_norm), 
                   optax.adam(learning_rate, mu_dtype=dtype))

    return rng, TrainState.create(apply_fn=model.apply, 
                                  params=params, 
                                  tx=tx)


def init_inference_actor(rng, 
                         init_image_shape, 
                         init_proprioception_shape, 
                         net_params, 
                         action_dim, 
                         spatial_softmax, 
                         mode,
                         dtype):
    
    model = ActorModel(net_params,
                       action_dim,  
                       spatial_softmax,
                       mode, 
                       dtype)
    
    init_image, init_proprioception = get_init_data(
        init_image_shape, 
        init_proprioception_shape, 
        mode)

    rng, key1, key2 = random.split(rng, 3)
    model.init(key1, 
               key2,
               init_image, 
               init_proprioception)

    return rng, model

def init_actor(rng, 
               critic, 
               learning_rate, 
               init_image_shape, 
               init_proprioception_shape, 
               net_params,  
               action_dim, 
               spatial_softmax, 
               mode, 
               dtype, 
               global_norm):
    
    model = ActorModel(net_params,
                       action_dim,  
                       spatial_softmax,
                       mode, 
                       dtype)

    rng, key1, key2 = random.split(rng, 3)
    
    init_image, init_proprioception = get_init_data(
        init_image_shape, 
        init_proprioception_shape, 
        mode)
    
    params = model.init(key1, 
                        key2,
                        init_image,
                        init_proprioception)['params']
    
    if mode==MODE.IMG_PROP or mode==MODE.IMG:
        params['encoder'] = critic.params['encoder']
    
    tx=optax.chain(optax.zero_nans(), 
                   optax.clip_by_global_norm(global_norm), 
                   optax.adam(learning_rate, mu_dtype=dtype))
    
    return rng, TrainState.create(apply_fn=model.apply, 
                                  params=params, 
                                  tx=tx)


def init_temperature(rng, learning_rate, alpha, dtype):
    model = Temperature(initial_temperature=alpha, dtype=dtype)
    rng, key = random.split(rng)
    params = model.init(key)['params']

    tx = optax.adam(learning_rate=learning_rate)

    return rng, TrainState.create(apply_fn=model.apply, 
                                  params=params, 
                                  tx=tx)
