# Sagemaker Repository Template
This repository provides a template for running *reinforcement learning* algorithms that trains a [ml-agents](https://github.com/Unity-Technologies/ml-agents) for a Unity environment on Amazon [AWS Sagemaker](https://aws.amazon.com/sagemaker/) service.

In addition, it offers a configurable script to generate a custom Docker container on which training algorithms will run as training jobs on AWS Sagemaker.


## Get Started with Amazon Web Services (AWS)

Here we collected some link to useful resources that can help you understand AWS and how to use the services.

- What is Amazon Web Services https://aws.amazon.com/what-is-aws
- AWS Account creation https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account
- AWS User creation https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html
- AWS Educational program https://aws.amazon.com/education/awseducate/
- AWS Sagemaker documentation https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html
- AWS Sagemaker Python SDK https://github.com/aws/sagemaker-python-sdk
- AWS S3 create bucket https://docs.aws.amazon.com/AmazonS3/latest/gsg/CreatingABucket.html
- AWS Identity and Access Management (IAM) guide https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html
- AWS Sagemaker notebook instance creation https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html

In addition, we provided further description of the different components in each sub-folder. Please check also them to get a complete overview.

## Steps to be up and running with Sagemaker

- Create an AWS Account and carry out the initial security measures, such as creating a non-root user and granting them proper permissions (in our case it means full access to Sagemaker, S3 and ECR services).
- Open Sagemaker console and create a new notebook instance. During the process it might be useful to create a new *role*, which will be required to execute the training jobs.  
In addition, a S3 bucket is automatically created and associated to this notebook instance. In case of different needs, it is still possible to create another S3 bucket as container for Sagemaker input/output data, but it might require to configure the proper permissions.  
In case you already have created a repository for your project, during the creation of the notebook there is the possibility to associate it with the notebook environment. (More details are provided in the `template` folder [description](https://github.com/danibix95/sagemaker-template/tree/master/template)).
- If required, build and upload the custom Docker image tailored to allow TensorFlow interact with Unity environments (check the detailed description [here](https://github.com/danibix95/sagemaker-template/tree/master/docker)).
- Exploit the template repository to implement your own training algorithm.
- Synch your repository project with the notebook environment and execute the job launcher.
- Once the training job is finished, you can collect the model and potential output data from S3 associated bucket.

**Note**: after executing the training job, if you do not require to run further job, please stop from Sagemaker console the notebook instance. This allows yoy to save computation hours (and costs) of the AWS instance.
