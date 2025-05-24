import os
import time
import pickle
import threading
import numpy as np
import collections
from multiprocessing import shared_memory, Process, Queue, Lock


SB_0 = 'sb0'
SB_1 = 'sb1'
READY = 'ready'
START = 'start'
SAVE = 'save'
CLOSE = 'close'

## rb: replay buffer
## sb: sampling buffer
## sp: sampling process
## sm: shared memory
## ip: inference process

Batch = collections.namedtuple(
    'Batch', ['images', 'proprioceptions', 'actions', 'rewards',
              'masks', 'next_images', 'next_proprioceptions'])


class ReplayBuffer():
    """Buffer to store environment transitions."""

    def __init__(self, 
                 image_shape, 
                 proprioception_shape, 
                 action_shape,
                 capacity, 
                 batch_size, 
                 init_buffers=True, 
                 load_path='',
                 img_aug_path='',
                 min_episode_length=20):

        self._image_shape = image_shape
        self._proprioception_shape = proprioception_shape
        self._action_shape = action_shape
        self._capacity = capacity
        self._batch_size = batch_size
        
        self._min_episode_length = min_episode_length

        self._lock = Lock()

        self._ignore_image = True
        self._ignore_propri = True

        if image_shape is not None:
            self._ignore_image = False

        if proprioception_shape is not None:
            self._ignore_propri = False

        if load_path:
            self._load_path = load_path
        else:
            self._load_path = ''

        self._aug_imgs = None
        if len(img_aug_path) > 0:
            self._aug_imgs = np.load(img_aug_path)
            self._total_aug_imgs = self._aug_imgs.shape[0]
            self._aug_img_index = 0
            aug_intensity = [0.0, 0.05, 0.1, 0.15, 0.2]
            shape = (90, 159, 12)
            self._aug_masks = []
            active_channels = [0, 1, 2, 4, 5, 6, 8, 9, 10] 
            
            for intensity in aug_intensity:
                mask = np.zeros(shape, dtype=np.float32)  
                mask[..., active_channels] = intensity
                alt_mask = 1 - mask
                self._aug_masks .append((alt_mask, mask))
                            

        if init_buffers:
            self._init_buffers()

    def _init_buffers(self):
        total_size = 0

        if self._load_path:
            total_size = self._load()
        else:
            self._idx = 0
            self._full = False
            self._count = 0
            self._steps = 0
                
            if not self._ignore_image: 
                self._image_cap = self._capacity + ((self._capacity // self._min_episode_length) * 2) 
        
                self._images = np.empty(
                    (self._image_cap, *self._image_shape), dtype=np.uint8) 
                self._images_idxs = np.empty((self._capacity,), dtype=np.int32)
                self._next_images_idxs = np.empty((self._capacity,), dtype=np.int32)
                
                self._im_idx = 0
                self._last_im_idx = -1
                
                total_size += self._images.nbytes + self._images_idxs.nbytes + \
                    self._next_images_idxs.nbytes
                    
            else:
                self._image_cap = -1
                self._im_idx = -1
                self._last_im_idx = -1

            if not self._ignore_propri:
                self._propris = np.empty(
                    (self._capacity, *self._proprioception_shape), 
                    dtype=np.float32)
                self._next_propris = np.empty(
                    (self._capacity, *self._proprioception_shape), 
                    dtype=np.float32)
                
                total_size += self._propris.nbytes + self._next_propris.nbytes

            self._actions = np.empty(
                (self._capacity, *self._action_shape), 
                dtype=np.float32)
            
            self._rewards = np.empty((self._capacity), 
                                     dtype=np.float32)
            self._masks = np.empty((self._capacity), 
                                   dtype=np.float32)
            
            total_size += self._actions.nbytes + self._rewards.nbytes + self._masks.nbytes
        
        return total_size

    def _add_image(self, image, next_image, first_step):
        if first_step: 
            idx1 = self._im_idx
            idx2 = (self._im_idx + 1) % self._image_cap
            self._images[idx1] = image
            self._images[idx2] = next_image
            self._im_idx = (self._im_idx + 2) % self._image_cap 
        else:
            idx1, idx2 = self._last_im_idx, self._im_idx
            self._images[idx2] = next_image
            self._im_idx = (self._im_idx + 1) % self._image_cap
        
        self._last_im_idx = idx2   
        return idx1, idx2
            
    def add(self, 
            image, 
            propri, 
            action, 
            reward, 
            next_image, 
            next_propri, 
            mask,
            first_step):
        if not self._ignore_image:
            if self._aug_imgs is not None:
                aug_img = self._aug_imgs[self._aug_img_index]
                img_intensity, aug_intensity = self._aug_masks[self._aug_img_index % len(self._aug_masks)]
                self._aug_img_index = (self._aug_img_index + 1) % self._total_aug_imgs
                
                h, w, c = image.shape
                history = c//4
            
                aug_img = np.concatenate((aug_img, np.zeros((h, w, 1), dtype=np.uint8)), axis=-1) 
                aug_img = np.concatenate([aug_img] * history, axis=-1).astype(np.float32) * aug_intensity
                
                image = image.astype(np.float32) * img_intensity + aug_img
                next_image = next_image.astype(np.float32) * img_intensity + aug_img
                
                image = image.astype(np.uint8)
                next_image = next_image.astype(np.uint8) 
                
            idx1, idx2 = self._add_image(image, next_image, first_step)
            self._images_idxs[self._idx] = idx1
            self._next_images_idxs[self._idx] = idx2
            
            
        if not self._ignore_propri:
            self._propris[self._idx] = propri
            self._next_propris[self._idx] = next_propri
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._masks[self._idx] = mask
            
        self._idx = (self._idx + 1) % self._capacity
        self._full = self._full or self._idx == 0
        self._count = self._capacity if self._full else self._idx
        self._steps += 1

    def sample(self):
        idxs = np.random.randint(0, 
                                 self._count,
                                 size=min(self._count, self._batch_size))
        
        if self._ignore_image:
            images = None
            next_images = None
        else:
            idxs_1 = self._images_idxs[idxs]
            idxs_2 = self._next_images_idxs[idxs]
            images = self._images[idxs_1]
            next_images = self._images[idxs_2]

        if self._ignore_propri:
            propris = None
            next_propris = None
        else:
            propris = self._propris[idxs]
            next_propris = self._next_propris[idxs]

        actions = self._actions[idxs]
        rewards = self._rewards[idxs]
        masks = self._masks[idxs]

        return Batch(images=images, 
                     proprioceptions=propris,
                     actions=actions, 
                     rewards=rewards, 
                     masks=masks,
                     next_images=next_images, 
                     next_proprioceptions=next_propris)


    def save(self, save_path):
        tic = time.time()
        print(f'Saving the replay buffer in {save_path}..')
        with self._lock:
            data = {
                'idx': self._idx,
                'full': self._full,
                'count': self._count,
                'steps': self._steps,
                'image_cap': self._image_cap,
                'im_idx': self._im_idx,
                'last_im_idx': self._last_im_idx,
            }

            with open(os.path.join(save_path, "buffer_data.pkl"),
                      "wb") as handle:
                pickle.dump(data, handle, protocol=4)

            if not self._ignore_image:
                np.save(os.path.join(save_path, "images.npy"), self._images) 
                np.save(os.path.join(save_path, "images_idxs.npy"), self._images_idxs)
                np.save(os.path.join(save_path, "next_images_idxs.npy"), self._next_images_idxs)

            if not self._ignore_propri:
                np.save(os.path.join(save_path, "propris.npy"), self._propris)
                np.save(os.path.join(save_path, "next_propris.npy"),
                        self._next_propris)

            np.save(os.path.join(save_path, "actions.npy"), self._actions)
            np.save(os.path.join(save_path, "rewards.npy"), self._rewards)
            np.save(os.path.join(save_path, "masks.npy"), self._masks)

        print("Saved the buffer locally,", end=' ')
        print("took: {:.3f}s.".format(time.time() - tic))

    def _load(self):
        tic = time.time()
        print("Loading buffer")

        data = pickle.load(open(os.path.join(self._load_path,
                                             "buffer_data.pkl"), "rb"))
        self._idx = data['idx']
        self._full = data['full']
        self._count = data['count']
        self._steps = data['steps']
        self._image_cap = data['image_cap']
        self._im_idx = data['im_idx']
        self._last_im_idx = data['last_im_idx']

        if not self._ignore_image:
            self._images = np.load(os.path.join(self._load_path, "images.npy"))
            self._images_idxs = np.load(os.path.join(self._load_path, "images_idxs.npy"))
            self._next_images_idxs = np.load(os.path.join(self._load_path, "next_images_idxs.npy"))

        if not self._ignore_propri:
            self._propris = np.load(os.path.join(self._load_path,
                                                 "propris.npy"))
            self._next_propris = np.load(os.path.join(self._load_path,
                                                      "next_propris.npy"))

        self._actions = np.load(os.path.join(self._load_path, "actions.npy"))
        self._rewards = np.load(os.path.join(self._load_path, "rewards.npy"))
        self._masks = np.load(os.path.join(self._load_path, "masks.npy"))

        print("Loaded the buffer from: {}".format(self._load_path), end=' ')
        print("Took: {:.3f}s".format(time.time() - tic))
        
    def close(self):
        pass


class AsyncSMReplayBuffer(ReplayBuffer):
    def __init__(self, 
                 image_shape, 
                 proprioception_shape, 
                 action_shape, 
                 capacity, 
                 batch_size, 
                 obs_queue, 
                 load_path='',
                 img_aug_path='',):
        
        super().__init__(
            image_shape, 
            proprioception_shape, 
            action_shape, 
            capacity,
            batch_size, 
            False, 
            load_path, 
            img_aug_path)
        
        sizes = self._get_sb_sizes(batch_size)

        self._obs_queue = obs_queue

        self._rcv_from_sampling_process_queue = Queue()
        self._send_to_sampling_process_queue = Queue()

        self._start_batch = False

        self._producer_process = Process(target=self._produce_samples_sp)
        self._producer_process.start()

        self._sb_0, self._sb_0_sm, sb_0_sm_names = self._create_sm_sb(sizes)
        self._sb_1, self._sb_1_sm, sb_1_sm_names = self._create_sm_sb(sizes)

        self._send_to_sampling_process_queue.put(sb_0_sm_names)
        self._send_to_sampling_process_queue.put(sb_1_sm_names)

        self._last_sb = None


    def sample(self):
        if not self._start_batch:
            self._start_batch = True
            self._obs_queue.put('start')
        sb_code = self._rcv_from_sampling_process_queue.get()

        if self._last_sb is not None:
            self._send_to_sampling_process_queue.put(self._last_sb)

        self._last_sb = sb_code

        if sb_code == SB_0:
            batch = self._sb_0
        else:
            batch = self._sb_1

        return batch
    
    def _recv_obs_sp(self):
        while True:
            observation = self._obs_queue.get()
            if isinstance(observation, str):
                if observation == CLOSE:
                    return
                if observation == START:
                    self._start_batch = True
                    continue

            with self._lock:
                self.add(*observation)

    def _get_sb_sizes(self, batch_size):
        image_size = 0
        proprioception_size = 0
        if not self._ignore_image:
            image_size = np.random.randint(
                0, 256, size=(batch_size, *self._image_shape), 
                dtype=np.uint8).nbytes
        if not self._ignore_propri:
            proprioception_size = np.random.uniform(
                size=(batch_size, *self._proprioception_shape)
                ).astype(np.float32).nbytes

        action_size = np.random.uniform(
            size=(batch_size, *self._action_shape)).astype(np.float32).nbytes
        
        mask_size = np.random.uniform(
            size=(batch_size,)).astype(np.float32).nbytes
        
        reward_size = np.random.uniform(
            size=(batch_size,)).astype(np.float32).nbytes

        return {
            'img_sb_size': image_size,
            'proprioception_sb_size': proprioception_size, 
            'action_sb_size': action_size,
            'mask_sb_size': mask_size,
            'reward_sb_size': reward_size
            }

    def _create_sm_sb(self, sizes):        
        images = None
        next_images = None
        img_sm = None
        next_img_sm = None
        if not self._ignore_image:
            img_sm = shared_memory.SharedMemory(create=True, size=sizes['img_sb_size'])
            next_img_sm = shared_memory.SharedMemory(create=True, 
                                                      size=sizes['img_sb_size'])
            
            images = np.ndarray((self._batch_size, *self._image_shape), 
                                dtype=np.uint8, buffer=img_sm.buf)
            next_images = np.ndarray((self._batch_size, *self._image_shape), 
                                     dtype=np.uint8, buffer=next_img_sm.buf)

        propris = None
        next_propris = None
        proprioception_sm = None
        next_proprioception_sm = None
        if not self._ignore_propri:
            proprioception_sm = shared_memory.SharedMemory(
                create=True, size=sizes['proprioception_sb_size'])
            next_proprioception_sm = shared_memory.SharedMemory(
                create=True, size=sizes['proprioception_sb_size'])
            
            propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=proprioception_sm.buf)
            next_propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=next_proprioception_sm.buf)


        action_sm = shared_memory.SharedMemory(create=True, size=sizes['action_sb_size'])
        mask_sm = shared_memory.SharedMemory(create=True, size=sizes['mask_sb_size'])
        reward_sm = shared_memory.SharedMemory(create=True, size=sizes['reward_sb_size'])

        actions = np.ndarray((self._batch_size, *self._action_shape), 
                             dtype=np.float32, buffer=action_sm.buf)
        masks = np.ndarray((self._batch_size,), dtype=np.float32, 
                           buffer=mask_sm.buf)
        rewards = np.ndarray((self._batch_size,), dtype=np.float32, 
                             buffer=reward_sm.buf)
        
        sb = Batch(images=images, proprioceptions=propris,
                      actions=actions, rewards=rewards, masks=masks,
                      next_images=next_images, next_proprioceptions=next_propris)
        
        sm_names = {}
        if not self._ignore_image:
            sm_names['img_sm'] = img_sm.name
            sm_names['next_img_sm'] = next_img_sm.name
        if not self._ignore_propri:
            sm_names['proprioception_sm'] = proprioception_sm.name
            sm_names['next_proprioception_sm'] = next_proprioception_sm.name
        sm_names['action_sm'] = action_sm.name
        sm_names['mask_sm'] = mask_sm.name
        sm_names['reward_sm'] = reward_sm.name
        
        sms = (img_sm, next_img_sm,  proprioception_sm, 
                next_proprioception_sm, action_sm, mask_sm, reward_sm)

        return sb, sms, sm_names
    
    def _copy_batch(self, batch_src, batch_dest):        
        if not self._ignore_image:
            np.copyto(batch_dest.images, batch_src.images)
            np.copyto(batch_dest.next_images, batch_src.next_images)

        if not self._ignore_propri:
            np.copyto(batch_dest.proprioceptions, batch_src.proprioceptions)
            np.copyto(batch_dest.next_proprioceptions, 
                      batch_src.next_proprioceptions)

        np.copyto(batch_dest.actions, batch_src.actions)
        np.copyto(batch_dest.rewards, batch_src.rewards)
        np.copyto(batch_dest.masks, batch_src.masks)

    def _get_sm_sb_sp(self, sm_names):
        total_size = 0
        images = None
        next_images = None
        img_sm = None
        next_img_sm = None
        if not self._ignore_image:
            img_sm = shared_memory.SharedMemory(
                name=sm_names['img_sm'])
            next_img_sm = shared_memory.SharedMemory(
                name=sm_names['next_img_sm'])
            
            images = np.ndarray((self._batch_size, *self._image_shape), 
                                dtype=np.uint8, buffer=img_sm.buf)
            next_images = np.ndarray((self._batch_size, *self._image_shape), 
                                     dtype=np.uint8, buffer=next_img_sm.buf)
            total_size += images.nbytes + next_images.nbytes

        propris = None
        next_propris = None    
        proprioception_sm = None
        next_proprioception_sm = None
        if not self._ignore_propri:
            proprioception_sm = shared_memory.SharedMemory(
                name=sm_names['proprioception_sm'])
            next_proprioception_sm = shared_memory.SharedMemory(
                name=sm_names['next_proprioception_sm'])
            
            propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=proprioception_sm.buf)
            next_propris = np.ndarray(
                (self._batch_size, *self._proprioception_shape), 
                dtype=np.float32, buffer=next_proprioception_sm.buf)
            
            total_size += propris.nbytes + next_propris.nbytes

        action_sm = shared_memory.SharedMemory(
            name=sm_names['action_sm'])
        mask_sm = shared_memory.SharedMemory(
            name=sm_names['mask_sm'])
        reward_sm = shared_memory.SharedMemory(
            name=sm_names['reward_sm'])
        
        actions = np.ndarray((self._batch_size, *self._action_shape), 
                             dtype=np.float32, buffer=action_sm.buf)
        masks = np.ndarray((self._batch_size,), dtype=np.float32, 
                           buffer=mask_sm.buf)
        rewards = np.ndarray((self._batch_size,), dtype=np.float32, 
                             buffer=reward_sm.buf)
        
        total_size += actions.nbytes + masks.nbytes + rewards.nbytes
        
        sb = Batch(images=images, proprioceptions=propris,
                      actions=actions, rewards=rewards, masks=masks,
                      next_images=next_images, next_proprioceptions=next_propris)
              
        sms = (img_sm, next_img_sm,  proprioception_sm, 
                next_proprioception_sm, action_sm, mask_sm, reward_sm)

        return sb, sms, total_size

    def _produce_samples_sp(self): 
        rb_size = self._init_buffers()
        self._recv_obs_thread = threading.Thread(target=self._recv_obs_sp)
        self._recv_obs_thread.start()

        self._rcv_from_ip_queue = self._send_to_sampling_process_queue
        self._send_to_ip_queue = self._rcv_from_sampling_process_queue

        sb_0_sm_names = self._rcv_from_ip_queue.get()
        sb_1_sm_names = self._rcv_from_ip_queue.get()

        self._sb_0, self._sb_0_sm, sb_0_size = self._get_sm_sb_sp(sb_0_sm_names)
        self._sb_1, self._sb_1_sm, sb_1_size = self._get_sm_sb_sp(sb_1_sm_names)
        
        total_size = rb_size + sb_0_size + sb_1_size
        print(f'Total size of buffers (in GB): {total_size / 1e9}')

        while not self._start_batch:
            # Checking if the replay buffer process 
            # needs to be closed while waiting
            if not self._rcv_from_ip_queue.empty():
                code = self._rcv_from_ip_queue.get()
                if code == CLOSE:
                    self._close_sp()
                    return
                elif code == SAVE:
                    self._rcv_from_ip_queue.get()
            time.sleep(0.1)

        with self._lock:
            sb_0 = super().sample()
        self._copy_batch(sb_0, self._sb_0)

        with self._lock:
            sb_1 = super().sample()
        self._copy_batch(sb_1, self._sb_1)

        self._send_to_ip_queue.put(SB_0)
        self._send_to_ip_queue.put(SB_1)

        while True:
            with self._lock:
                batch = super().sample()
            code = self._rcv_from_ip_queue.get()
            if code == SB_0:
                self._copy_batch(batch, self._sb_0)
                self._send_to_ip_queue.put(SB_0)
            elif code == SB_1:
                self._copy_batch(batch, self._sb_1)
                self._send_to_ip_queue.put(SB_1)
            elif code == SAVE:
                save_path = self._rcv_from_ip_queue.get()
                super().save(save_path)
            elif code == CLOSE:
                break

        self._close_sp()

    def _close_sp(self):
        self._obs_queue.put('close')
        print('Closng replay buffer shared memory..')
        with self._lock:
            for mem in self._sb_0_sm:
                if mem is not None:
                    try:
                        mem.close()
                    except:
                        pass
            for mem in self._sb_1_sm:
                if mem is not None:
                    try:
                        mem.close()
                    except:
                        pass
        
    def save(self, save_path):
        self._send_to_sampling_process_queue.put(SAVE)
        self._send_to_sampling_process_queue.put(save_path)
        time.sleep(1)
        self._lock.acquire()
        self._lock.release()

    def close(self):
        self._send_to_sampling_process_queue.put(CLOSE)
        self._producer_process.join()

        for mem in self._sb_0_sm:
            if mem is not None:
                try:
                    mem.close()
                    mem.unlink()
                except:
                    pass
        for mem in self._sb_1_sm:
            if mem is not None:
                try:
                    mem.close()
                    mem.unlink()
                except:
                    pass

