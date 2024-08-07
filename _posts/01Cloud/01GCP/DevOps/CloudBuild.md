






custom build configuration files
- can perform other actions, in parallel or in sequence, in addition to simply building containers:
  - running tests on your newly built containers,
  - pushing them to various destinations,
  - and even deploying them to Kubernetes Engine.


![Screenshot 2024-07-30 at 21.30.40](/assets/img/Screenshot%202024-07-30%20at%2021.30.40.png)


1. 2 Git repositories:
   1. app repository: contains the source code of the application itself
   2. env repository: contains the manifests for the Kubernetes Deployment

2. When you push a change to the app repository,
   1. the `Cloud Build` pipeline runs tests, builds a container image, and pushes it to `Artifact Registry`.
   2. After pushing the image, `Cloud Build` updates the Deployment manifest and pushes it to the `env repository`.
   3. This triggers another `Cloud Build` pipeline that applies the manifest to the GKE cluster and, if successful, stores the manifest in another branch of the `env repository`.

3. The app and env repositories are kept separate because they have different lifecycles and uses.
   1. The main users of the `app repository` are actual humans and this repository is dedicated to a specific application.
   2. The main users of the `env repository` are automated systems (such as Cloud Build), and this repository might be shared by several applications.
      1. The env repository can have several branches that each map to a specific environment (you only use production in this lab) and reference a specific container image, whereas the app repository does not.

![Screenshot 2024-07-30 at 21.33.44](/assets/img/Screenshot%202024-07-30%20at%2021.33.44.png)


```bash
gcloud config set project "qwiklabs-gcp-03-56f994cb15bb"
export PROJECT_ID=$(gcloud config get-value project)
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
export REGION=us-central1
gcloud config set compute/region $REGION

gcloud services enable container.googleapis.com \
    cloudbuild.googleapis.com \
    sourcerepo.googleapis.com \
    containeranalysis.googleapis.com

# Create an Artifact Registry Docker repo to store the container images:
gcloud artifacts repositories create my-repository \
  --repository-format=docker \
  --location=$REGION

# Create a GKE cluster to deploy the sample application of this lab:
gcloud container clusters create hello-cloudbuild \
  --num-nodes 1 \
  --region $REGION

git config --global user.email "jiaying_luo2@apple.com"
git config --global user.name "Grace JyLuo"



# ========= Task 2. Create the Git repositories in Cloud Source Repositories

#  create the two Git repositories:
gcloud source repos create hello-cloudbuild-app
gcloud source repos create hello-cloudbuild-env

# download code
cd ~
mkdir hello-cloudbuild-app
gcloud storage cp -r gs://spls/gsp1077/gke-gitops-tutorial-cloudbuild/* hello-cloudbuild-app


# Configure Cloud Source Repositories as a remote:
cd ~/hello-cloudbuild-app

export REGION=us-central1
sed -i "s/us-central1/$REGION/g" cloudbuild.yaml
sed -i "s/us-central1/$REGION/g" cloudbuild-delivery.yaml
sed -i "s/us-central1/$REGION/g" cloudbuild-trigger-cd.yaml
sed -i "s/us-central1/$REGION/g" kubernetes.yaml.tpl

PROJECT_ID=$(gcloud config get-value project)

git init
git add .
git remote add google "https://source.developers.google.com/p/${PROJECT_ID}/r/hello-cloudbuild-app"
git commit -m "Initial commit"


# ========= Task 3. Create a container image with Cloud Build

# from flask import Flask
# app = Flask('hello-cloudbuild')
# @app.route('/')
# def hello():
#   return "Hello World!\n"
# if __name__ == '__main__':
#   app.run(host = '0.0.0.0', port = 8080)

# FROM python:3.7-slim
# RUN pip install flask
# WORKDIR /app
# COPY app.py /app/app.py
# ENTRYPOINT ["python"]
# CMD ["/app/app.py"]

# create a Cloud Build build based on the latest commit with the following command:
cd ~/hello-cloudbuild-app
COMMIT_ID="$(git rev-parse --short=7 HEAD)"
gcloud builds submit --tag="${REGION}-docker.pkg.dev/${PROJECT_ID}/my-repository/hello-cloudbuild:${COMMIT_ID}" .


# ========= Task 4. Create the Continuous Integration (CI) pipeline
# In the Cloud console, go to Cloud Build > Triggers.
# Click Create Trigger
# In the Name field, type hello-cloudbuild.
# Under Event, select Push to a branch.
# Under Source, select hello-cloudbuild-app as your Repository and .* (any branch) as your Branch.
# Under Build configuration, select Cloud Build configuration file.
# In the Cloud Build configuration file location field, type cloudbuild.yaml after the /.
# For the Service account, use the Compute Engine default service account.
# Click Create.

# To start this trigger, run the following command:
cd ~/hello-cloudbuild-app
git add .
git commit -m "Type Any Commit Message here"
git push google master



# ========= Task 5. Create the Test Environment and CD pipeline
# Cloud Build is also used for the continuous delivery pipeline. The pipeline runs each time a commit is pushed to the candidate branch of the hello-cloudbuild-env repository. The pipeline applies the new version of the manifest to the Kubernetes cluster and, if successful, copies the manifest over to the production branch. This process has the following properties:

# The candidate branch is a history of the deployment attempts.
# The production branch is a history of the successful deployments.
# You have a view of successful and failed deployments in Cloud Build.
# You can rollback to any previous deployment by re-executing the corresponding build in Cloud Build. A rollback also updates the production branch to truthfully reflect the history of deployments.

# Grant Cloud Build access to GKE
# To deploy the application in your Kubernetes cluster, Cloud Build needs the Kubernetes Engine Developer Identity and Access Management role.
# In Cloud Shell execute the following command:
PROJECT_NUMBER="$(gcloud projects describe ${PROJECT_ID} --format='get(projectNumber)')"
gcloud projects add-iam-policy-binding ${PROJECT_NUMBER} \
  --member=serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com \
  --role=roles/container.developer

# initialize the hello-cloudbuild-env repository with two branches (production and candidate) and a Cloud Build configuration file describing the deployment process.

# clone the hello-cloudbuild-env repository and create the production branch.
# It is still empty.
# In Cloud Shell execute the following command:
cd ~
gcloud source repos clone hello-cloudbuild-env
cd ~/hello-cloudbuild-env
git checkout -b production


# The cloudbuild-delivery.yaml file describes the deployment process to be run in Cloud Build. It has two steps:
# - Cloud Build applies the manifest on the GKE cluster.
# - If successful, Cloud Build copies the manifest on the production branch.
# cloudbuild-delivery.yaml


# copy the cloudbuild-delivery.yaml file available in the hello-cloudbuild-app repository and commit the change:
cd ~/hello-cloudbuild-env
cp ~/hello-cloudbuild-app/cloudbuild-delivery.yaml ~/hello-cloudbuild-env/cloudbuild.yaml
git add .
git commit -m "Create cloudbuild.yaml for deployment"

# Create a candidate branch and push both branches for them to be available in Cloud Source Repositories:
git checkout -b candidate
git push origin production
git push origin candidate


# Grant the Source Repository Writer IAM role to the Cloud Build service account for the hello-cloudbuild-env repository:
PROJECT_NUMBER="$(gcloud projects describe ${PROJECT_ID} \
  --format='get(projectNumber)')"

cat >/tmp/hello-cloudbuild-env-policy.yaml <<EOF
bindings:
- members:
  - serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com
  role: roles/source.writer
EOF

gcloud source repos set-iam-policy \
  hello-cloudbuild-env /tmp/hello-cloudbuild-env-policy.yaml



# Create the trigger for the continuous delivery pipeline

# In the Cloud console, go to Cloud Build > Triggers.
# Click Create Trigger.
# In the Name field, type hello-cloudbuild-deploy.
# Under Event, select Push to a branch.
# Under Source, select hello-cloudbuild-env as your Repository and ^candidate$ as your Branch.
# Under Build configuration, select Cloud Build configuration file.
# In the Cloud Build configuration file location field, type cloudbuild.yaml after the /.
# For the Service account, use the Compute Engine default service account.
# Click Create.




# Modify the continuous integration pipeline to trigger the continuous delivery pipeline.
# - add some steps to the continuous integration pipeline that will generate a new version of the Kubernetes manifest and push it to the hello-cloudbuild-env repository to trigger the continuous delivery pipeline.
# Copy the extended version of the cloudbuild.yaml file for the app repository:
cd ~/hello-cloudbuild-app
cp cloudbuild-trigger-cd.yaml cloudbuild.yaml
# The cloudbuild-trigger-cd.yaml is an extended version of the cloudbuild.yaml file. It adds the steps below: they generate the new Kubernetes manifest and trigger the continuous delivery pipeline.


# Commit the modifications and push them to Cloud Source Repositories:
cd ~/hello-cloudbuild-app
git add cloudbuild.yaml
git commit -m "Trigger CD pipeline"
git push google master





# ========= Task 6. Review Cloud Build Pipeline

# In the Cloud console, go to Cloud Build > Dashboard.
# Click into the hello-cloudbuild-app trigger to follow its execution and examine its logs.
# The last step of this pipeline pushes the new manifest to the hello-cloudbuild-env repository, which triggers the continuous delivery pipeline.
# Return to the main Dashboard.
# You should see a build running or having recently finished for the hello-cloudbuild-env repository. You can click on the build to follow its execution and examine its logs.




# ========= Task 7. Test the complete pipeline

# The complete CI/CD pipeline is now configured. Test it from end to end.

# In the Cloud console, go to Kubernetes Engine > Gateways,Services & Ingress.
# There should be a single service called hello-cloudbuild in the list. It has been created by the continuous delivery build that just ran.

# Click on the endpoint for the hello-cloudbuild service. You should see "Hello World!". If there is no endpoint, or if you see a load balancer error, you may have to wait a few minutes for the load balancer to be completely initialized. Click Refresh to update the page if needed.
# App 1 Hello World!

# In Cloud Shell, replace "Hello World" with "Hello Cloud Build", both in the application and in the unit test:
# Commit and push the change to Cloud Source Repositories:
cd ~/hello-cloudbuild-app
sed -i 's/Hello World/Hello Cloud Build/g' app.py
sed -i 's/Hello World/Hello Cloud Build/g' test_app.py

git add app.py test_app.py
git commit -m "Hello Cloud Build"
git push google master


# This triggers the full CI/CD pipeline.
# After a few minutes, reload the application in your browser. You should now see "Hello Cloud Build!".


# ========= Task 8. Test the rollback

# In this task, you rollback to the version of the application that said "Hello World!".

# In the Cloud console, go to Cloud Build > Dashboard.
# Click on View all link under Build History for the hello-cloudbuild-env repository.
# Click on the second most recent build available.
# Click Rebuild.
# Rollback success screen

# When the build is finished, reload the application in your browser. You should now see "Hello World!" again.

# When the build is finished, reload the application in your browser. You should now see "Hello World!" again.

```
