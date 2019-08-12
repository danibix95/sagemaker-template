#! /bin/bash

# =================================== #
#      PARAMETERS  CONFIGURATION      #
# =================================== #
# Amazon AWS account ID (can be retrieved when logged as root account
# from https://console.aws.amazon.com/billing/home?#/account )
ACCOUNT_ID=<012345678912>
# select in which region the container will be stored
# (https://docs.aws.amazon.com/general/latest/gr/rande.html)
REGION=eu-west-1
# name of the repository on AWS ECR service, where the image will be stored
REPOSITORY_NAME=fruitpunchai/tf-mlagents
# =================================== #

# retrieve a Tensorflow image from Amazon public ECR and use it as starting point
docker pull 520713654638.dkr.ecr.$REGION.amazonaws.com/sagemaker-tensorflow-scriptmode:1.11.0-gpu-py3

# build the Docker image exploiting the dockerfile contained in this folder
docker build -t $REPOSITORY_NAME .

# tag the image so that it can be recognized in the repository
docker tag $REPOSITORY_NAME:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest

# upload created container on AWS ECR service
# (note: keep track of the image size to forecast potential billings)
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest