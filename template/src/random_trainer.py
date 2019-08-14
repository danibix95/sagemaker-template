#! /usr/bin/env python3

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
    def run(self, max_episodes = 50, max_steps = 200):
        """
            Run a RL algorithm that generates number of episodes
            where at each step a random action is performed by the agent.
        """
        unity_file = self.download_unity_env()
        env = self.get_gym_env(unity_file)
        # ========================================== #

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

if __name__ == '__main__':
    r_tr = RandomTrainer(_params)
    r_tr.run()
