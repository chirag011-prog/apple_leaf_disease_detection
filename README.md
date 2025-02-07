Dataset: https://www.kaggle.com/datasets/akinyemijoseph/apple-leaf-disease-dataset-6-classes-v2
The model was deployed using AWS services. Tried with EC2, Lambda (serverless) and Sagemaker AI. Since this cannot be done using aws-free tier due to the size of the model. But suceeded in creating endpoint and deployed model using AWS sagemaker AI, but couldnt load the model using ml.t2.medium which was only available for free-tier but if we can use large type we can put the model into use. 

Steps for EC2 :
launch an instance -> connect the instance using ssh -> create python virtual environment -> install necessary dependencies -> copy model path file form device into the instance -> create fastAPI script on your ec2 instance -> run fastapi server. ( note: possible issues -> storage issue of ec2 , solution: attach EBS or some other instance type with more storeage space , can make it more beautiful using Route53 to host

Steps for Lambda (serverless) :
store model in s3 -> create lambda function -> load the model from s3 (IAM role with fulls3access)  -> install fastAPI + Dependencies in Lambda -> code (lambda_function.py) downloads the model from S3 to pytorch and process requests -> deploy on AWS lambda -> expose API

Steps for Sagemaker AI :
store the model in s3 bucket -> IAM role with sagemakerfull access and s3fullaccess -> create notebook in sagemaker AI and add the IAM role -> create infernece.py file and notebook to model deployment and endpoint -> model deployed as endopint make sure its InService -> test the model in same jupyter notebook. 
