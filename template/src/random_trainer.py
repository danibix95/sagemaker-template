#! /usr/bin/env python3

import argparse
from trainer import Trainer

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

class RandomTrainer(Trainer):
    def __init__(self, params, h_params):
        super().__init__(params)
        self.h_params = h_params

    def run(self, max_episodes = 50, max_steps = 200):
        """
            Run a RL algorithm that generates number of episodes
            where at each step a random action is performed by the agent.
        """
        unity_file = self.download_unity_env()
        env = self.get_gym_env(unity_file)
        # ========================================== #

        # hyperparameters test
        print('Received hyperparameters:', self.h_params)

        for episode in range(max_episodes):
            observation = env.reset()

            for step in range(max_steps):
                env.render()
                print(observation)

                # assign the agent a random action to perform
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)

                if done:
                    print("Episode {} finished after {} timesteps".format(episode, step+1))
                    break

        # ========================================== #
        RandomTrainer.close_env(env)

    def read_hyperparameters():
      args_parser = argparse.ArgumentParser()

      # more hyperparameters can be added, depending on project requirements
      # pay attention that hyperparameters types should be limited
      # to primitive data types, such as int, str, float, bool, ...
      # For further information, please refer to https://docs.python.org/3/library/argparse.html
      args_parser.add_argument('--hyper-param-example', default='',
                                 type=str,
                                 help='This is an example of passing an hyperparameters to the script')
      args_parser.add_argument('--maximum-limit-of-everything', default=7,
                                 type=int,
                                 help='This is the maximum size of everything. Just a joke :)')

      return args_parser.parse_args()

if __name__ == '__main__':
    # read hyperparameters given by the job launcher notebook
    h_params = RandomTrainer.read_hyperparameters()

    r_tr = RandomTrainer(_params, h_params)
    r_tr.run()
