import jax
import functools
import numpy as np
import jax.image
from jax import random, numpy as jnp 

from jsac.algo.replay_buffer import Batch

KP = 3.0

def critic_update(rng, 
                  actor, 
                  critic, 
                  critic_target_params, 
                  temp, 
                  batch, 
                  discount):

    rng, key_ac = random.split(rng)
    
    critic_target = critic.replace(params=critic_target_params)
    
    next_actions, next_log_probs = actor.apply_fn(
        {"params": actor.params}, 
        key_ac,
        batch.next_images, 
        batch.next_proprioceptions) 

    target_Qs = critic_target.apply_fn(
        {"params": critic_target.params}, 
        batch.next_images, 
        batch.next_proprioceptions, 
        next_actions)                          
    
    target_Qs = jnp.transpose(target_Qs) 
    target_Q_min = jnp.min(target_Qs, axis=1)
    target_V = target_Q_min - temp.apply_fn({"params": temp.params}) * next_log_probs
    target_Q = batch.rewards + (batch.masks * discount * target_V) 
    target_Q = jnp.expand_dims(target_Q, -1)

    def critic_loss_fn(critic_params):
        qs = critic.apply_fn( 
            {'params': critic_params}, 
            batch.images, 
            batch.proprioceptions, 
            batch.actions)      
        qs  = jnp.transpose(qs)   
        critic_loss = jnp.mean((qs - target_Q)**2)
        
        return critic_loss, {
            'critic_loss': critic_loss,
            'qs': qs.mean()
        }
    
    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    critic_new = critic.apply_gradients(grads=grads)
    clipped_params = jax.tree_util.tree_map(lambda x: jnp.clip(x, -KP, KP), critic_new.params)
    critic_new = critic_new.replace(params=clipped_params)
    return rng, critic_new, info


def actor_update(rng, 
                 actor, 
                 critic, 
                 temp, 
                 batch):
    rng, key_ac = random.split(rng)
    
    # The actor's encoder parameters are not updated
    # They are copied from critic's parameters
    if 'encoder' in critic.params:
        actor_params = actor.params.copy()
        actor_params['encoder'] = critic.params['encoder']
        actor = actor.replace(params=actor_params)

    def actor_loss_fn(actor_params):    
        actions, log_probs = actor.apply_fn(
            {"params": actor_params}, 
            key_ac,
            batch.images, 
            batch.proprioceptions)
 
        qs = critic.apply_fn(
            {'params': critic.params}, 
            batch.images, 
            batch.proprioceptions, 
            actions)                      
        
        q = jnp.min(qs, axis=0) 
        
        actor_loss = (log_probs * temp.apply_fn({"params": temp.params}) - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    actor_new = actor.apply_gradients(grads=grads)
    clipped_params = jax.tree_util.tree_map(lambda x: jnp.clip(x, -KP, KP), actor_new.params)
    actor_new = actor_new.replace(params=clipped_params)
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

def random_contrast(key, image, lower=0.5, upper=1.5):
    """Apply random contrast to RGB channels only."""
    factor = jax.random.uniform(key, (), minval=lower, maxval=upper)
    mean = jnp.mean(image, axis=(0, 1), keepdims=True)
    return (image - mean) * factor + mean

def random_brightness(key, image, delta=0.2):
    """Apply random brightness to RGB channels only."""
    factor = jax.random.uniform(key, (), minval=-delta, maxval=delta)
    return image + factor

def random_saturation(key, image, lower=0.5, upper=1.5):
    """Apply random saturation to RGB channels only."""
    factor = jax.random.uniform(key, (), minval=lower, maxval=upper)
    gray = jnp.mean(image, axis=-1, keepdims=True)
    return (image - gray) * factor + gray

def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    x = jnp.arange(-(size // 2), size // 2 + 1)
    y = jnp.arange(-(size // 2), size // 2 + 1)
    X, Y = jnp.meshgrid(x, y)
    kernel = jnp.exp(-(X**2 + Y**2)/(2*sigma**2))
    return kernel / jnp.sum(kernel)

def random_gaussian_blur(key, image, kernel_size=5, sigma_range=(0.1, 2.0)):
    """Apply random Gaussian blur to RGB channels only."""
    sigma = jax.random.uniform(key, (), minval=sigma_range[0], maxval=sigma_range[1])
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel[:, :, jnp.newaxis, jnp.newaxis]
    
    # Prepare kernel for each input channel
    num_channels = image.shape[-1]
    kernel = jnp.repeat(kernel, num_channels, axis=-2)
    
    # Pad the image for convolution
    pad_width = kernel_size // 2
    padded = jnp.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')
    
    # Reshape for conv_general_dilated
    padded = padded.transpose(2, 0, 1)[None, ...]  # NCHW format
    kernel = kernel.transpose(2, 3, 0, 1)  # IOHW format
    
    # Apply convolution
    result = jax.lax.conv_general_dilated(
        lhs=padded,
        rhs=kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        feature_group_count=num_channels
    )
    
    # Reshape back to original format
    result = result[0].transpose(1, 2, 0)
    return result

def augment_rgb_group(key, image_group, config):
    """Apply augmentations to a group of 3 RGB channels."""
    keys = jax.random.split(key, 4)
    
    # Apply augmentations in sequence
    image_group = random_contrast(keys[0], image_group, 
                                lower=config['contrast_lower'], 
                                upper=config['contrast_upper'])
    image_group = random_brightness(keys[1], image_group, 
                                  delta=config['brightness_delta'])
    image_group = random_saturation(keys[2], image_group, 
                                  lower=config['saturation_lower'], 
                                  upper=config['saturation_upper'])
    image_group = random_gaussian_blur(keys[3], image_group, 
                                     kernel_size=config['blur_kernel_size'],
                                     sigma_range=config['blur_sigma_range'])
    
    # Clip values to valid range for uint8
    return jnp.clip(image_group, 0, 255)

def augment_single_image(key, image, config):
    """Apply augmentations to a single image with multiple RGB groups and masks."""
    keys = jax.random.split(key, 3)
    
    # Process each RGB group separately
    rgb_groups = [
        (0, 1, 2),  # First RGB group
        (4, 5, 6),  # Second RGB group
        (8, 9, 10)  # Third RGB group
    ]
    
    result = image.astype(jnp.float32)
    
    for idx, (r, g, b) in enumerate(rgb_groups):
        # Extract RGB group
        rgb_image = result[:, :, [r, g, b]]
        
        # Apply augmentations
        augmented = augment_rgb_group(keys[idx], rgb_image, config)
        
        # Put back the augmented channels
        result = result.at[:, :, [r, g, b]].set(augmented)
    
    return result.astype(jnp.uint8)

def batched_augmentations(key, images, config=None):
    """Apply augmentations to a batch of images."""
    if config is None:
        config = {
            'contrast_lower': 0.8,
            'contrast_upper': 1.3,
            
            'brightness_delta': 0.2,
            
            'saturation_lower': 0.8,
            'saturation_upper': 1.3,
            
            'blur_kernel_size': 3, 
            'blur_sigma_range': (0.1, 0.5)
        }
    
    # Generate keys for each image in the batch
    keys = jax.random.split(key, images.shape[0])
    
    # Apply augmentations to each image in the batch
    return jax.vmap(augment_single_image, (0, 0, None))(keys, images, config)

@functools.partial(jax.jit, static_argnames=('update_actor',
                                             'update_target',
                                             'num_critic_updates'))
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
               num_critic_updates):
    
    rng, key1, key2 = random.split(rng, 3)

    img_fl = batch.images is not None

    if img_fl:
        image = batched_augmentations(key1, batch.images)
        next_images = batched_augmentations(key1, batch.next_images)
                                      
        images = batched_random_crop(key2, image)
        next_images = batched_random_crop(key2, next_images) 

        batch = batch._replace(images=images, next_images=next_images)
        
    batch_size = batch.actions.shape[0] // num_critic_updates
    for i in range(num_critic_updates):
        m_batch = Batch(images=batch.images[i*batch_size: (i+1)*batch_size] if img_fl else None,
                        proprioceptions=batch.proprioceptions[i*batch_size: (i+1)*batch_size],
                        actions=batch.actions[i*batch_size: (i+1)*batch_size],
                        rewards=batch.rewards[i*batch_size: (i+1)*batch_size],
                        masks=batch.masks[i*batch_size: (i+1)*batch_size],
                        next_images=batch.next_images[i*batch_size: (i+1)*batch_size] if img_fl else None,
                        next_proprioceptions=batch.next_proprioceptions[i*batch_size: (i+1)*batch_size],)
        
        rng, critic, critic_info = critic_update(
            rng, 
            actor, 
            critic, 
            critic_target_params, 
            temp, 
            m_batch, 
            discount)
        
        if update_actor and i == num_critic_updates - 1:
            rng, actor, actor_info = actor_update(
                rng, 
                actor, 
                critic, 
                temp,
                m_batch)

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