from setuptools import setup, find_packages

setup(
    name='jsac',
    version='1.0.0',
    description='Jax based real-time Reinforcement Learning for Vision-Based Robotics Utilizing Local and Remote Computers',
    author='Fahim Shahriar',
    author_email='fshahri1@ualberta.ca',
    url='https://github.com/fahimfss/JSAC',
    packages=find_packages(include=['jsac', 'jsac.*']),
    install_requires=[ 
        'gymnasium==0.29.0',
        'seaborn==0.13.2',
        'termcolor==2.4.0',
        'tensorboardX==2.6.2.2',
        'flax==0.8.5',
        'pyopengl==3.1.7',
        'wandb==0.16.0',
        'tensorflow_probability==0.21.0',
        'imageio==2.34.1',
        'mujoco==3.2.3',
        'dm_control==1.0.22',
        'opencv_python==4.9.0.80',
        'numpy==1.25.2',
        'orbax-checkpoint==0.4.3',
        'moviepy==1.0.3'
    ],
)
