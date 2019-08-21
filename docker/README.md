# Fruit Punch AI - Competition Docker container

The FruitPunch AI competition provide a Unity environment as simulation space to train your own agent by means of a *Reinforcement Learning* (RL) technique. Since training RL algorithms on your own computer might require a non-negligible amount of time and consume many resources, a viable alternative to speed up the process and offload the computation is to employ Amazon AWS Sagemaker service.

[Amazon Sagemaker](https://aws.amazon.com/sagemaker/) service easily allows to train machine, deep and reinforcement learning algorithms exploiting Docker containers to manage algorithms dependencies. Nonetheless many pre-built images are available, FruitPunch AI provides a custom TensorFlow image which integrates all the libraries needed to interact with an Unity environment, such as [Unity ml-agents](https://github.com/Unity-Technologies/ml-agents) and [Open AI Gym](https://gym.openai.com/).

In this document is therefore reported how to build this custom Docker image necessary to run the RL training of an Unity *ml-agent* on AWS Sagemaker.

### Requirements
The following instruction have been tailored for being execuited in a system that support `shell` scripting environment. However, it should be possible to run the same `docker` commands in Windows.

In order to build the custom image, the following packages needs to be installed and configured on your machine:

- Docker Engine and Client (https://docs.docker.com/install/)
- Amazon AWS Command Line Interface (https://aws.amazon.com/cli/)
- Amazon AWS Docker ECR helper (https://github.com/awslabs/amazon-ecr-credential-helper)

**Note**: please refer to each package documentation to read their installation process. In addition, make sure that `aws configure` command has been executed and all the required details filled in. For more information check [aws-cli documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html).

Furthermore, it is necessary to create on [Amazon AWS ECR](https://aws.amazon.com/ecr/) service a repository where to store the built image. To perform this task it is sufficient to select the ECR service from [AWS console](console.aws.amazon.com) and click on the `Create repository` button shown in the service home page. Then it is required to insert a name (e.g `fruitpunchai/tf-mlagents`) and to select *mutable* to allow later updates of the same image tag.

### Building Process
Once the requirements are fulfilled, the script `builder.sh` provides all the commands to create and upload the custom image to Amazon ECR service. Before running mentioned script, there are some parameters that has to be configured within it:

- `ACCOUNT_ID` a twelve digit numbers that represents your Amazon Root User ID. More details can be found [here](https://docs.aws.amazon.com/general/latest/gr/acct-identifiers.html).
- `REGION` the code of the region where the image should be pushed on (e.g `eu-west-1`). More region codes can be retrieved [here](https://docs.aws.amazon.com/general/latest/gr/rande.html).
- `REPOSITORY_NAME` the name of the ECR repository created in the previous step (e.g. `fruitpunchai/tf-mlagents`).

These information are employ as configuration parameters of the builder script.
Finally, after every step has been completed, it is possible to run the builder script

    ./builder.sh

which will pull the base TensorFlow image, install the proper libraries, build the Docker image and push it to AWS ECR.

### Troubleshooting
Here are reported some information and links in case of potential issues:

- **Account Permissions**: it might be possible that *current user* (the one configured for the `aws-cli`) does not own the permissions to write onto the ECR repository. Therefore it is necessary to open the *Identity and Access Management* (IAM) service with the root account and grant missing permission to that user.  
Consequently, from the list of users select the one associated to the `aws-cli` and add required permissions (ECR service access) by clicking `add-inline-policy` and fill in proper details.
- **Linux Kernel 5.2**: when pushing the custom image an error is reported. Please refer to this [issue](https://github.com/docker/for-linux/issues/711).

### Future Changes
In case more or different Python packages are required to be bundled inside the Docker image, a simple process is provided. Indeed, it is sufficient to edit the `requirements.txt` file and executing again the builder script to update the online image, which is the one used for any new Sagemaker training job.

In addition to the default image (Tensorflow 1.7), other custom images have been made available to test different frameworks and their versions. However, those later images have to be tested before being used in production.
