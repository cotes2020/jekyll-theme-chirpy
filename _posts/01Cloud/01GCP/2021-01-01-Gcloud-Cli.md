---
title: GCP - Gcloud
date: 2021-01-01 11:11:11 -0400
categories: [01GCP]
tags: [GCP]
toc: true
image:
---


- [Gcloud](#gcloud)
  - [setup \& secure the GCP credentials for Go apps](#setup--secure-the-gcp-credentials-for-go-apps)


---

# Gcloud


## setup & secure the GCP credentials for Go apps

Overview of GCP authentication in Go libraries
- how Go apps use credentials to authenticate their access to GCP. Especially, it's using Google Cloud client libraries.

GCP libraries
- The key areas in the underlying GCP libraries that explain their interaction with the Google’s APIs are in packages called options and internal
- settings.go at `google.golang.org/api/internal/settings.go`
- option.go at `google.golang.org/api/option/option.go`


Setup secure authentication to GCP from the go app

- **Setting up authentication**
  - Credentials File
  - The authentication relies on a GCP Service Account, which can be downloaded as a JSON file

![1*78vSBQK1cu9VFyHerwtdcw](/assets/img/1*78vSBQK1cu9VFyHerwtdcw.webp)


- Option 1 — **Reading the credentials from a file**
  - The Google Cloud client libraries expect the credentials file’s path to be declared as an environment variable and is setup to be checked by default
  - `export GOOGLE_APPLICATION_CREDENTIALS="/home/me/gcp-creds.json"`
  - Accessing credentials file from the local file system

![1*ZKEpjMw_COrG-1TqZawSbw](/assets/img/1*ZKEpjMw_COrG-1TqZawSbw.webp)


- Option 2 — **Encode the JSON file as env var**
  - encode the JSON as base64 string and pass it as the environment variable
  - Run the below command to print the base64 string of the credentials file
  - `bash> cat /home/me/gcp-creds.json | base64`
  - Then set a new environment variable
  - `export GCP_CREDS_JSON_BASE64="paste_base64_output_here"`
  - decode the string in code and then pass it to the `WithCredentialsJSON` option.
  - This way you don’t have to commit the credentials file

- Option 3 — **Use cryptography**
  - Another alternative is to use a cryptographic library to encrypt the file and pass the key as the environment variable to decrypt it during setup



.
