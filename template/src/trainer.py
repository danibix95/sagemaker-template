#! /usr/bin/env python3

import os
import sys
import gym
import stat
import boto3
import zipfile

from abc import ABC

# Unity Gym wrapper for Open AI Gym
from gym_unity.envs.unity_env import UnityEnv

# ============================================= #
#  Default configuration parameters to perform  #
#    the RL training on Unity Gym environment   # 
# ============================================= #
_default_params = {
    # - the name of the bucket from which download the Unity env
    'bucket': '',
    # - the filename of the zipped Unity environment
    'zip_env_name': '',
    # - the folder where to extract Unity environment
    'build_folder': '',
    # - the executable file of Unity environment
    'env_file': '',
    # - refers to the port to use for communication with the environment. Defaults to 0.
    'worker_id': 0,
    # - wheter to use visual observations (True) or vector observations (False). Defaults to False.
    'use_visual': False,
    # - whether to use integer [0-255] (True) or float [0.0-1.0] (False) visual representation
    'use_uint8_visual': False,
    # - whether you intent to launch an environment which contains more than one agent. Defaults to False
    'multiagent': False,
    # - will flatten a branched discrete action space into a Gym Discrete.
    #   Otherwise, it will be converted into a MultiDiscrete. Defaults to False
    'flatten_branched': False,
    # - whether to return a list of observation instead of only one. Defaults to False.
    'allow_multiple_visual_obs': False,
    # whether to not initialize the graphic drivers to run the environment
    # (set to False only in case use_visual is set to True and the graphic drivers are required)
    'no_graphics': True
}

class Trainer(ABC):
    def __init__(self, params = _default_params):
        self.params = params

    def __repr__(self):
        return self.params

    def download_unity_env(self):
        """
            Retrieve and prepare the Unity environment
            for being wrapped into an Open AI Gym
            (use selected parameters during trainer initialization)
        """
        s3 = boto3.client('s3')
        s3.download_file(self.params['bucket'], self.params['zip_env_name'],
                         self.params['zip_env_name'])

        with zipfile.ZipFile(self.params['zip_env_name'], 'r') as zip_ref:
                zip_ref.extractall(self.params['build_folder'])
        
        # full path of the executable Unity environment file
        unity_file = os.path.join(self.params['build_folder'], self.params['env_file'])
        
        # assign proper permissions to Unity environment file,
        # so that it can be executed for training
        os.chmod(unity_file, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IRGRP)

        return unity_file

    def get_gym_env(self, unity_file):
        """
            @param unity_file   the full path of Unity environment

            Returns an Open AI Gym that wraps given Unity environment
            based on selected trainer parameters.
        """
        return UnityEnv(unity_file,
                        self.params.get('worker_id', 0),
                        use_visual=self.params.get('use_visual', False),
                        uint8_visual=self.params.get('use_uint8_visual', False),
                        multiagent=self.params.get('multiagent', False),
                        flatten_branched=self.params.get('flatten_branched', False),
                        allow_multiple_visual_obs=self.params.get('allow_multiple_visual_obs', False),
                        no_graphics=self.params.get('no_graphics', True))
    
    @abstractmethod
    def run(self):
        """
            Executes the RL training algorithm.
            To be implemented by subclasses.
        """
        pass

    def close_env(environment):
        """
            Static method provided to remember that Gym environments
            should be closed after the training process is finished
        """
        environment.close()
                       
if __name__ == '__main__':
    tr = Trainer()
    print(tr)
