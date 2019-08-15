#! /usr/bin/env python3

import os
import onnx
import keras2onnx
import numpy as np
import tensorflow as tf

from trainer import Trainer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

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
    'no_graphics': True,
    'observation_size': 8,
    'epsilon': 0.08,
    'gamma': 0.95
}

def build_model(obs_size):
    inputs = Input(shape=(obs_size, ))
    h = Dense(16, activation='relu')(inputs)
    h = Dense(64, activation='relu')(h)
    h = Dense(64, activation='relu')(h)
    preds = Dense(2, activation='tanh')(h)

    model = Model(inputs=inputs, outputs=preds)
    model.compile(optimizer=Adam(amsgrad=True), loss='mean_squared_error')

    return model

class TfTrainer(Trainer):
    def run(self, max_episodes = 500, max_timesteps = 1000):
        """
            Run a Deep RL algorithm (inspired by https://keon.io/deep-q-learning/)
        """
        unity_file = self.download_unity_env()
        env = self.get_gym_env(unity_file)
        # ========================================== #

        model = build_model(self.params['observation_size'])

        print('Training...')
        for episode in range(max_episodes):
            print('Running episode {} of {}'.format(episode+1, max_episodes), end='\r')
            # observation == state
            observation = env.reset()
            observation = np.reshape(observation, (1, self.params['observation_size']))

            for step in range(max_timesteps):
                if np.random.rand() <= self.params['epsilon']:
                    # The agent acts randomly
                    action = [env.action_space.sample()]
                else:
                    action = model.predict(observation)

                observation, reward, done, info = env.step(action)

                action_val = action[0]
                targets = np.zeros(len(action_val))
                for i in range(len(action_val)):
                    targets[i] = reward + self.params['gamma'] * action_val[i]

                observation = np.reshape(observation, (1, self.params['observation_size']))
                model.fit(observation, np.asarray([targets]), epochs=1, verbose=0)

                # if objective reached, no need to continue
                if done:
                    break

        # Note: the content of /opt/ml/model and /opt/ml/output is automatically uploaded
        # to previously selected bucket (by the estimator) at the end of the execution
        # os.environ['SM_MODEL_DIR'] correspongs to /opt/ml/model
        model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'tf_rldemo.onnx')

        # Note: converting Keras model to ONNX one for being
        #       later converted into Barracuda format
        #       In fact, this model can not be directly employed in Unity ml-agents
        #       More info can be found here: https://github.com/onnx/keras-onnx
        onnx_model = keras2onnx.convert_keras(model, model.name)
        onnx.save_model(onnx_model, model_path)
        print('\nTraining finished!')

        # ========================================== #
        TfTrainer.close_env(env)

if __name__ == '__main__':
    tf_tr = TfTrainer(_params)
    tf_tr.run(max_episodes = 1000)
