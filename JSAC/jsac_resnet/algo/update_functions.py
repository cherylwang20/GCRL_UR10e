import jax
import functools
from jax import random, numpy as jnp

from jsac.algo.replay_buffer import Batch

# KP = 3.0


def critic_apply(critic, images, proprioceptions, actions, resnet, params=None):
    train = True if params else False
    
    if resnet:
        if params:
            variables = {'params': params, 'batch_stats': critic.batch_stats}
        else:
            variables = {'params': critic.params, 'batch_stats': critic.batch_stats}
    else:
        if params:
            variables = {'params': params}
        else:
            variables = {'params': critic.params}

    if train: 
        return critic.apply_fn(variables, 
                               images, 
                               proprioceptions, 
                               actions, 
                               train=True,
                               mutable=['batch_stats'])
    else: 
        return critic.apply_fn(variables, 
                               images, 
                               proprioceptions, 
                               actions, 
                               train=False) 
        

def actor_apply(actor, key, images, proprioceptions, resnet, params=None):
    if resnet:
        if params:
            variables = {'params': params, 'batch_stats': actor.batch_stats}
        else:
            variables = {'params': actor.params, 'batch_stats': actor.batch_stats}
    else:
        if params:
            variables = {'params': params}
        else:
            variables = {'params': actor.params}
    
    return actor.apply_fn(variables, 
                          key, 
                          images, 
                          proprioceptions, 
                          train=False) 


def critic_update(rng, 
                  actor, 
                  critic, 
                  critic_target_params, 
                  temp, 
                  batch, 
                  discount, 
                  resnet):

    rng, key_ac = random.split(rng, 2)
    critic_target = critic.replace(params=critic_target_params)
    next_actions, next_log_probs = actor_apply(actor, 
                                               key_ac, 
                                               batch.next_images, 
                                               batch.next_proprioceptions, 
                                               resnet)
    target_Qs = critic_apply(critic_target, 
                             batch.next_images,  
                             batch.next_proprioceptions, 
                             next_actions, 
                             resnet)
    
    target_Qs = jnp.transpose(target_Qs) 
    target_Q_min = jnp.min(target_Qs, axis=1)
    target_V = target_Q_min - temp.apply_fn({"params": temp.params}) * next_log_probs
    target_Q = batch.rewards + (batch.masks * discount * target_V) 
    target_Q = jnp.expand_dims(target_Q, -1)

    def critic_loss_fn(critic_params):
        x = critic_apply(critic, 
                         batch.images, 
                         batch.proprioceptions, 
                         batch.actions, 
                         resnet, 
                         critic_params)
        if resnet:
            qs, stats = x
            batch_stats = stats['batch_stats']
        else:
            qs = x
            batch_stats = None
                 
        qs  = jnp.transpose(qs)   
        critic_loss = jnp.mean((qs - target_Q)**2)
        
        return critic_loss, {
            'critic_loss': critic_loss,
            'qs': qs.mean(),
            'batch_stats' : batch_stats
        }
   
    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    critic_new = critic.apply_gradients(grads=grads)
    batch_stats = info.pop('batch_stats')
    if batch_stats:
        critic_new = critic_new.replace(batch_stats=batch_stats)
    return rng, critic_new, info


def actor_update(rng, 
                 actor, 
                 critic, 
                 temp, 
                 batch, 
                 resnet):
    rng, key_ac = random.split(rng)
    
    # The actor's encoder parameters are not updated
    # They are copied from critic's parameters
    if 'encoder' in critic.params:
        actor_params = actor.params.copy()
        actor_params['encoder'] = critic.params['encoder']
        actor = actor.replace(params=actor_params)
    if resnet:
        actor = actor.replace(batch_stats=critic.batch_stats)

    def actor_loss_fn(actor_params):   
        actions, log_probs = actor_apply(actor, 
                                         key_ac, 
                                         batch.images, 
                                         batch.proprioceptions, 
                                         resnet, 
                                         actor_params) 

        qs = critic_apply(critic, 
                          batch.images, 
                          batch.proprioceptions, 
                          actions, 
                          resnet)                
        
        q = jnp.min(qs, axis=0) 
        
        actor_loss = (log_probs * temp.apply_fn({"params": temp.params}) - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    actor_new = actor.apply_gradients(grads=grads)
 
    return rng, actor_new, info


def temp_update(temp, entropy, target_entropy):
    
    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {
            'temperature': temperature, 
            'temp_loss': temp_loss}

    grads, info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
    temp_new = temp.apply_gradients(grads=grads)

    return temp_new, info


def target_update(critic, critic_target_params, tau):
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        critic_target_params)

    return new_target_params

## random_crop and batched_random_crop source:
## https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py

def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2, ), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1, ), dtype=jnp.int32)])
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)),
                         mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


@functools.partial(jax.jit, static_argnames=('update_actor',
                                             'update_target',
                                             'num_critic_updates', 
                                             'resnet'))
def update_jit(rng, 
               actor, 
               critic, 
               critic_target_params, 
               temp, 
               batch, 
               discount, 
               tau,
               target_entropy, 
               update_actor, 
               update_target,
               num_critic_updates,
               resnet):
    
    rng, key1, key2 = random.split(rng, 3)

    img_fl = batch.images is not None

    if img_fl:
        images = batched_random_crop(key1, batch.images)
        next_images = batched_random_crop(key2, batch.next_images)
        
        batch = batch._replace(images=images,
                            next_images=next_images)
        
    batch_size = batch.actions.shape[0] // num_critic_updates
    for i in range(num_critic_updates):
        m_batch = Batch(images=batch.images[i*batch_size: (i+1)*batch_size] if img_fl else None,
                        proprioceptions=batch.proprioceptions[i*batch_size: (i+1)*batch_size],
                        actions=batch.actions[i*batch_size: (i+1)*batch_size],
                        rewards=batch.rewards[i*batch_size: (i+1)*batch_size],
                        masks=batch.masks[i*batch_size: (i+1)*batch_size],
                        next_images=batch.next_images[i*batch_size: (i+1)*batch_size] if img_fl else None,
                        next_proprioceptions=batch.next_proprioceptions[i*batch_size: (i+1)*batch_size],)
        
        rng, critic, critic_info = critic_update(rng, 
                                                 actor, 
                                                 critic, 
                                                 critic_target_params, 
                                                 temp, 
                                                 m_batch, 
                                                 discount, 
                                                 resnet)
        
        if update_actor and i == num_critic_updates - 1:
            rng, actor, actor_info = actor_update(rng, 
                                                  actor, 
                                                  critic, 
                                                  temp, 
                                                  m_batch,
                                                  resnet)

            temp, alpha_info = temp_update(
                temp, 
                actor_info['entropy'],
                target_entropy)
        else:
            actor_info = {}
            alpha_info = {}

    if update_target:
        critic_target = target_update(
            critic, 
            critic_target_params, 
            tau)
    else:
        critic_target = critic_target_params

    return rng, actor, critic, critic_target, temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }