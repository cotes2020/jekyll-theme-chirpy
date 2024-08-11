---
title: GCP - Artifact Registry
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, DevOps]
tags: [GCP]
toc: true
image:
---

- [GCP Artifact Registry](#gcp-artifact-registry)
  - [overview](#overview)
  - [Repository formats](#repository-formats)
  - [Repository modes](#repository-modes)
  - [Repository usage example](#repository-usage-example)
  - [create repo](#create-repo)
    - [create standard repo](#create-standard-repo)
  - [Connect to Build](#connect-to-build)
    - [Connect to Cloud Build](#connect-to-cloud-build)
  - [Deploy to Google Cloud](#deploy-to-google-cloud)
    - [Deploying to Cloud Run](#deploying-to-cloud-run)

---

# GCP Artifact Registry

```bash
gcloud artifacts repositories create quickstart-docker-repo \
    --repository-format=docker \
    --location=us-west1 \
    --description="Docker repository" \
    --project=PROJECT

gcloud artifacts repositories list \
    --project=PROJECT

# Before you can push or pull images, configure Docker to use the Google Cloud CLI to authenticate requests to Artifact Registry.
gcloud auth configure-docker us-west1-docker.pkg.dev

# Obtain an image to push
docker pull us-docker.pkg.dev/google-samples/containers/gke/hello-app:1.0

# Tag the image with a registry name
docker tag us-docker.pkg.dev/google-samples/containers/gke/hello-app:1.0 \
    us-west1-docker.pkg.dev/PROJECT/quickstart-docker-repo/quickstart-image:tag1

# Push the image to Artifact Registry
docker push us-west1-docker.pkg.dev/PROJECT/quickstart-docker-repo/quickstart-image:tag1

# Pull the image from Artifact Registry
docker pull us-west1-docker.pkg.dev/PROJECT/quickstart-docker-repo/quickstart-image:tag1

# Delete
gcloud artifacts repositories delete quickstart-docker-repo \
    --location=us-west1
```

---

## overview

Artifact Registry enables you to store different artifact types, create multiple repositories in a single project, and associate a specific region or multi-region with each repository.

---

## Repository formats

- Each repository is associated with a specific artifact format.
- For example, a Docker repository stores Docker images.
- You can create multiple repositories for each format in the same Google Cloud project.

---

## Repository modes

There are multiple repository modes.
- you cannot change the repository mode after you have created a repository.

**Standard repository**
- regular Artifact Registry repositories for private artifacts.
- upload and download artifacts directly with these repositories and use Artifact Analysis to scan for vulnerabilities and other metadata.

**Remote repository**
- A `read-only` repository that acts as a proxy to store artifacts from preset external sources such as Docker Hub, Maven Central, the Python Package Index (PyPI), Debian or CentOS as well as user-defined sources for supported formats.
- The first time you request an artifact version, the repository downloads it from the external source and caches a copy of it.
- The repository serves the cached copy when the same version is requested again.

- Remote repositories reduce latency and improve availability for builds and deployments on Google Cloud. You can also use Artifact Analysis to scan cached packages for vulnerabilities and other metadata.

**Virtual repository**
- A `read-only` repository that acts as a single access point to download, install, or deploy artifacts of the same format from one or more upstream repositories.
- An upstream repository can be a standard, remote, or virtual repository.

- Virtual repositories simplify client configuration for consumers of your artifacts.
- You can also mitigate dependency confusion attacks by configuring your upstream policy to prioritize repositories with your private artifacts over remote repositories that cache public artifacts.

---

## Repository usage example

![Screenshot 2024-07-19 at 15.59.46](/assets/img/Screenshot%202024-07-19%20at%2015.59.46.png)

1. In the development project, a Java development team uses Cloud Build to build a Java application.

    1. The build can request public Java dependencies using the virtual repository.

    2. The virtual repository serves the dependencies from the remote repository, which is a caching proxy for Maven Central.

    3. Cloud Build uploads the package to the standard Maven repository in the component project.

2. In the runtime project, Cloud Build containerizes the Java application.

   1. The build uses the Maven virtual repository to download the application.
   2. The virtual repository serves the package from the standard repository in the development project.
   3. The build can also download public Java dependencies from the same virtual repository.

3. In the runtime project, **Cloud Build** uploads the built container image to a `standard Docker repository`.

4. GKE pulls images from the Docker virtual repository.

    1. The upstream standard Docker repository provides private images, such as the containerized Java application.

    2. The upstream remote repository provides images that GKE requests from Docker Hub.


---

## create repo

### create standard repo

```bash
gcloud artifacts repositories create REPOSITORY \
    --repository-format=apt \
    --location=LOCATION \
    --description="DESCRIPTION" \
    --kms-key=KMS-KEY \
    --async
```

---

## Connect to Build

You can build your artifacts with:

- Cloud Build, which tightly integrates with Artifact Registry.

- Format-specific tools such as Maven for Java packages or Docker for container images.

- General build or continuous integration tools such as Jenkins or Tekton.

---

### Connect to Cloud Build

1. Configure a Docker build

```yaml
# build config file
steps:
images:
- '${_LOCATION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/${_IMAGE}'

- name: 'gcr.io/cloud-builders/docker'
  args: [ 'build', '-t', '${_LOCATION}-docker.pkg.dev/$PROJECT_ID/${_REPOSITORY}/${_IMAGE}', '.' ]

# pip install
- name: python
  entrypoint: pip
  args: ["install", "-r", "requirements.txt", "--user"]

# python upload to repo
- name: python
  entrypoint: python
  args:
  - '-m'
  - 'twine'
  - 'upload'
  - '--repository-url'
  - 'https://${_LOCATION}-python.pkg.dev/$PROJECT_ID/${_REPOSITORY}/'
  - 'dist/*'

# pip install the package from the Python repository
- name: python
    entrypoint: pip
    args:
    - 'install'
    - '--index-url'
    - 'https://${_LOCATION}-python.pkg.dev/$PROJECT_ID/${_REPOSITORY}/simple/'
    - '${_PACKAGE}'
    - '--verbose'
```

2. When you are ready to run the build, specify values for the user-defined substitutions. For example, this command substitutes:

```bash
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_LOCATION="us-east1",_REPOSITORY="my-repo",_IMAGE="my-image" .
```


---

## Deploy to Google Cloud


Google Cloud runtime environments are preconfigured with access to repositories in the same project.

You must configure or modify permissions if:
- You are using a service account in one project to access Artifact Registry in a different project
- You are using a service account with read-only access to storage, but you want the service account to both upload and download artifacts
- You are using a custom service account to interact with Artifact Registry.

---

### Deploying to Cloud Run

- You can deploy a container image stored in Artifact Registry to Cloud Run.
- You can also deploy directly from source to Cloud Run, which includes automatically creating a container image for your built source and storing the image in Artifact Registry.

Deploying from local source, Cloud Run can automatically:
- Containerize local source code.
- Push the container image to an Artifact Registry repository.
- Deploy the container image Cloud Run from the repository.

Cloud Run pushes and pulls images using the repository cloud-run-source-deploy in the region that you specify at deploy time. If the repository does not exist, Cloud Run creates it for you if your account has the required permissions.


Deploying container images

- You can deploy an image by tag or digest that is stored in Artifact Registry.

- Deploying to a service for the first time creates its first revision. Note that revisions are immutable. If you deploy from a container image tag, it will be resolved to a digest and the revision will always serve this particular digest.

- You can deploy a container using the Google Cloud console or the gcloud command line. For instructions see, Deploying container images.

.
