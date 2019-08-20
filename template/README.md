
# Repository Template
In this folder is reported the basic structure that a repository should adopt to run a *reinforcement learning* algorithm on Amazon AWS Sagemaker.

The folder structure is composed by two main components:
- the `job_launcher` Jupyter notebook that is employed to launch the training jobs, according to provided configuration
- the `src` folder which contains the base Trainer class (`trainer.py`) and all the trainer RL algorithms that the user provides.

## Notebook Configuration
The job launcher Jupyter notebook provides a detailed description of how to configure and run a training job. In a nutshell, the notebook requires the configuration of the S3 output bucket, which image use for training and which source file run.

**Note**: the job launcher notebook is designed to be run within a Sagemaker Jupyter environment (kernel `conda_tensorflow_p36`). In case users need to run the notebook in a different location, it is their own duty to implement the authentication code, which could be based on [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) library and [Sagemaker SDK](https://sagemaker.readthedocs.io/en/stable/).

## Trainer Implementation
In order to create a new RL algorithm it is sufficient to create a new Python file, import the Trainer abstract class from the `trainer` file and extend it by implementing the `run` method. In addition, in the main guard is necessary to instantiate the class and call its `run` method. In this manner the Sagemaker RLEstimator will execute the proper code with provided configuration (as a dictionary within implemented user file).

Regarding configuration parameters, the trainer job expects to find the Unity environment as a compressed file stored on a S3 bucket. Therefore, it is required to provide:

- the S3 bucket name
- the compressed file name
- the folder where to extract the Unity environment
- the file name of the Unity executable within the extracted folder

**Note**: it is advised to chose a S3 bucket whose region is the same where the custom Docker image is stored on the ECR. This reduce transfer times and costs.

Moreover, other parameters are needed to be set so that the [Unity Gym wrapper](https://github.com/Unity-Technologies/ml-agents/tree/master/gym-unity) can provide a proper Open AI Gym interface for the interaction between the Unity environment and the RL algorithm. These parameters are shown below:

- `env_file` - the executable file of Unity environment
- `worker_id` - refers to the port to use for communication with the environment. Defaults to 0
- `use_visual` - wheter to use visual observations (True) or vector observations (False). Defaults to False
- `use_uint8_visual` - whether to use integer [0-255] (True) or float [0.0-1.0] (False) visual representation
- `multiagent` - whether you intent to launch an environment which contains more than one agent. Defaults to False
- `flatten_branched` - is set to True it flats a branched discrete action space into a Gym Discrete. Otherwise, it will be converted into a MultiDiscrete. Defaults to False
- `allow_multiple_visual_obs` - whether to return a list of observation instead of only one. Defaults to False
- `no_graphics` - whether to not initialize the graphic drivers to run the environment (set to False only in case use_visual is set to True and the graphic drivers are required and available). Defaults to True

In the current `src` folder, in addition to the base abstract class, three basic examples are provided (`random_trainer`, `ppo_trainer`, `tf_trainer`) to show a possible `Trainer` class extension and parameters configuration.

In addition to the main `Trainer` class, a `LocalTrainer` class has been made available so that the same RL algorithm can be trained locally on your computer by just changing from which class your code inherits and configuring the parameters in the proper manner, so that the Unity environment files can be found on your computer.

## AWS Sagemaker
The AWS Sagemaker console easily allow to connect your own repository (e.g. Github, ...) with the code provided within a Sagemaker Jupyter environment. As a result, an example of workflow that can be adopted is the following one:

- create your own repository, starting from our template folder
- implement your own training algorithm
- push the repository online
- create a notebook instance on Amazon AWS Sagemaker
- associate your repository with the notebook instance
- start the Sagemaker Jupyter environment and execute the job launcher notebook to create a training job
- monitor the execution from Sagemaker training job console and CloudWatch logs panel
- collect the results from provided S3 bucket, where they are stored in a folder named as the *training job name* followed by the timestamp of the job initialization

## Repository Integration
A well written guide on how to create or link a repository on AWS Sagemaker is reported [here](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-notebooks-now-support-git-integration-for-increased-persistence-collaboration-and-reproducibility/).

In case your repository is private and stored on Github, [this guide](https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line) on creating a *Personal Access Token* (PAT) might be useful. As a clarification, when granting permissions to a PAT, it is sufficient to check the `repo` one to allow AWS to access your private repository.

Another noteworthy point to take into considerations is the fact that the synchronization between the Github repository and AWS Sagemaker Jupyter environment works similarly to a local repository. As a results any modification on AWS Sagemaker must be reverted in order to update the code after a push to the online repository. Adoptiong Jupyter Lab might help in performing git operations on the AWS repository.

## Further Examples
In the repository provided by AWS Labs can be found more [Sagemaker RL examples](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/reinforcement_learning). However, they are more general, based on pre-existing toolkits and not tailored for managing Unity environments.
