#! /bin/bash

# =================================== #
#      PARAMETERS  CONFIGURATION      #
# =================================== #
# Amazon AWS account ID (can be retrieved when logged as root account
# from https://console.aws.amazon.com/billing/home?#/account )
ACCOUNT_ID=<012345678912>
# select in which region the container will be stored (e.g eu-west-1 => Ireland)
# (https://docs.aws.amazon.com/general/latest/gr/rande.html)
# Note: this is indipendent from the region selected for downloading the base images in the dockerfiles
REGION=eu-west-1
# name of the repository on AWS ECR service, where the image will be stored
REPOSITORY_NAME=fruitpunchai/tf-mlagents
# =================================== #

# in case no dockerfile is selected then use the default image
# retrieve a Tensorflow image from Amazon public ECR and use it as starting point
dcf=$1
if [ -z $dcf ]; then
  dcf=dockerfile.tf-1.7
fi

# build the Docker image exploiting selected dockerfile
case $dcf in
"dockerfile.tf-1.7"|"dockerfile.tf-1.11")
  docker pull 520713654638.dkr.ecr.$REGION.amazonaws.com/sagemaker-tensorflow-scriptmode:1.11.0-gpu-py3
  docker build -t $REPOSITORY_NAME -f $dcf .
  ;;
"dockerfile.tf-1.14")
  docker pull 763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-training:1.14-gpu-py3
  docker build -t $REPOSITORY_NAME -f $dcf .
  ;;
"dockerfile.pytorch-1.1.0")
  docker pull 520713654638.dkr.ecr.us-east-1.amazonaws.com/sagemaker-pytorch:0.4.1-gpu-py3
  docker build -t $REPOSITORY_NAME -f $dcf .
  ;;
*)
  echo "Selected dockerfile does not exist!"
esac

# tag the image so that it can be recognized in the repository
docker tag $REPOSITORY_NAME:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest

# upload created container on AWS ECR service
# (note: keep track of the image size to forecast potential billings)
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest