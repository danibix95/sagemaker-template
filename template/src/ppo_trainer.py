#! /usr/bin/env python3

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

        # Note: the content of /opt/ml/model and /opt/ml/output is automatically uploaded
        # to previously selected bucket (by the estimator) at the end of the execution
        # os.environ['SM_MODEL_DIR'] correspongs to /opt/ml/model
        model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'ppo2_rldemo')
        
        # Note: this model can not be directly employed in Unity ml-agents
        model.save(model_path)
        
        # ========================================== #
        BaselinePPOTrainer.close_env(env)
                       
if __name__ == '__main__':
    ppo_tr = BaselinePPOTrainer(_params)
    ppo_tr.run()
