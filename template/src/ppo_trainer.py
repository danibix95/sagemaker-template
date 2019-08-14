#! /usr/bin/env python3

import os
import tensorflow as tf

from trainer import Trainer

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# ============================================= #
#  Example configuration parameters to perform  #
#    the RL training on Unity Gym environment   #
# ============================================= #
_params = {
    'bucket': 'fruitpunch-sagemaker-test',
    'zip_env_name': 'lx_build.zip',
    'build_folder': 'LinuxBuild',
    'env_file': 'rl_demo.x86_64',
    'worker_id': 0,
    'use_visual': False,
    'use_uint8_visual': False,
    'multiagent': False,
    'flatten_branched': False,
    'allow_multiple_visual_obs': False,
    'no_graphics': True
}

def generate_checkpoint_from_model(model_path, checkpoint_name):
    """
        https://github.com/hill-a/stable-baselines/issues/329

        To convert stable baselines models into a TensorFlow one.
    """
    model = PPO2.load(model_path)

    with model.graph.as_default():
        if os.path.exists(checkpoint_name):
            shutil.rmtree(checkpoint_name)

        tf.saved_model.simple_save(model.sess, checkpoint_name,
                                   inputs={"obs": model.act_model.obs_ph},
                                   outputs={"action": model.action_ph})

class BaselinePPOTrainer(Trainer):
    def run(self, max_episodes = 500, max_timesteps = 10000):
        """
            Run the PPO RL algorithm provided by stable baselines library
            (https://github.com/hill-a/stable-baselines) and save
            the generated model back to training job S3 bucket
        """
        unity_file = self.download_unity_env()
        env = self.get_gym_env(unity_file)
        # ========================================== #

        # RL stable baselines algorithms require a vectorized environment to run
        env = DummyVecEnv([lambda: env])

        model = PPO2(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=max_timesteps)

        obs = env.reset()
        for i in range(max_episodes):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

        sb_model_path = os.path.join('/tmp', 'ppo2_rldemo_sb')
        model.save(sb_model_path)

        # Note: the content of /opt/ml/model and /opt/ml/output is automatically uploaded
        # to previously selected bucket (by the estimator) at the end of the execution
        # os.environ['SM_MODEL_DIR'] correspongs to /opt/ml/model
        model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'ppo2_rldemo')

        # Note: this model can not be directly employed in Unity ml-agents
        #       it has to be converted into Barracuda format
        generate_checkpoint_from_model(sb_model_path, model_path)

        # ========================================== #
        BaselinePPOTrainer.close_env(env)

if __name__ == '__main__':
    ppo_tr = BaselinePPOTrainer(_params)
    ppo_tr.run()
