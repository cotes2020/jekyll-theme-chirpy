---
title: GCP Computing - Cloud Build
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, Container]
tags: [GCP]
toc: true
image:
---

- [GCP Computing - Cloud Build](#gcp-computing---cloud-build)
  - [Deploy](#deploy)

---

# GCP Computing - Cloud Build

---

## Deploy

```sh
gcloud auth list


nano quickstart.sh
#!/bin/sh
echo "Hello, world! The time is $(date)."


nano Dockerfile
# Dockerfile
FROM alpine
COPY quickstart.sh /
CMD ["/quickstart.sh"]

chmod +x quickstart.sh

gcloud artifacts repositories create quickstart-docker-repo \
  --repository-format=docker \
  --location=us-west1 \
  --description="Docker repository"

gcloud builds submit \
  --tag us-west1-docker.pkg.dev/${DEVSHELL_PROJECT_ID}/quickstart-docker-repo/quickstart-image:tag1

nano cloudbuild.yaml
# cloudbuild.yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: [ 'build', '-t', 'YourRegionHere-docker.pkg.dev/$PROJECT_ID/quickstart-docker-repo/quickstart-image:tag1', '.' ]
images:
- 'YourRegionHere-docker.pkg.dev/$PROJECT_ID/quickstart-docker-repo/quickstart-image:tag1'

export REGION=us-west1
sed -i "s/YourRegionHere/$REGION/g" cloudbuild.yaml

gcloud builds submit --config cloudbuild.yaml


nano quickstart.sh
#!/bin/sh
if [ -z "$1" ]
then
	echo "Hello, world! The time is $(date)."
	exit 0
else
	exit 1
fi

nano cloudbuild2.yaml

steps:
- name: 'gcr.io/cloud-builders/docker'
  args: [ 'build', '-t', 'YourRegionHere-docker.pkg.dev/$PROJECT_ID/quickstart-docker-repo/quickstart-image:tag1', '.' ]
- name: 'YourRegionHere-docker.pkg.dev/$PROJECT_ID/quickstart-docker-repo/quickstart-image:tag1'
  args: ['fail']
images:
- 'YourRegionHere-docker.pkg.dev/$PROJECT_ID/quickstart-docker-repo/quickstart-image:tag1'

sed -i "s/YourRegionHere/$REGION/g" cloudbuild2.yaml

gcloud builds submit --config cloudbuild2.yaml

```
