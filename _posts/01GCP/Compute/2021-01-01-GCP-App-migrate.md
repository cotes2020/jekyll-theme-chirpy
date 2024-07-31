---
title: GCP - Compute migrate
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, Compute]
tags: [GCP]
toc: true
image:
---

- [Migration](#migration)
  - [1 Application Migration](#1-application-migration)
  - [在 Google 云平台上部署 ML 模型](#在-google-云平台上部署-ml-模型)
  - [在 Google Cloud Function 上部署 ML 模型](#在-google-cloud-function-上部署-ml-模型)
  - [在 Google AI 云上部署 ML 模型](#在-google-ai-云上部署-ml-模型)
  - [Google Cloud Run](#google-cloud-run)
  - [App migrate](#app-migrate)

---

# Migration


---

## 1 Application Migration

- creating a VM, then using the container option, and then the advanced options to set the starting commands and environment variables.
- For something like Grafana I have setup a template, and use a group with a minimum instance of 1, then use the load balancer to provide certificate offloading.

---

## 在 Google 云平台上部署 ML 模型

> 本节中使用的代码可以在 /kurtispykes/gcp-deployment-example GitHub repo 中找到。

1. 在 Google App Engine 上部署 ML 模型

predict.py

- 本模块中的步骤包括：
  - 将持久化模型加载到内存中。
  - 创建一个将一些输入作为参数的函数。
  - 在函数中，将输入转换为 pandas DataFrame并进行预测。

```py
import joblib
import pandas as pd

model = joblib.load("logistic_regression_v1.pkl")

def make_prediction(inputs):
    """
    Make a prediction using the trained model
    """
    inputs_df = pd.DataFrame(
        inputs,
        columns=["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
    )
    predictions = model.predict(inputs_df)

    return predictions
```

main.py
- 推理逻辑必须封装在 web 服务中。用 Flask 包装模型。
- 在代码示例中创建了两个端点
  - index：主页
  - /predict：用于与部署的模型交互。

```py
import numpy as np
from flask import Flask, request
from predict import make_prediction

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to my Flask API</h1>"
        "</body>"
        "</html>"
    )
    return body

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()

    sepal_length_cm = data_json["sepal_length_cm"]
    sepal_width_cm = data_json["sepal_width_cm"]
    petal_length_cm = data_json["petal_length_cm"]
    petal_width_cm = data_json["petal_width_cm"]

    data = np.array(
      [
        [sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]
      ]
    )
    predictions = make_prediction(data)
    return str(predictions)

if __name__ == "__main__":
    app.run()
```


app.yaml
- 其中包含用于运行应用程序的运行时。
```yaml
runtime: python38
```

2. 在 Google Cloud 控制台:
   1. 选择 App Engine
   2. 从 App Engine 页面，选择 Create Application

3. App Engine
   3. 选择要在其中创建应用程序的区域。
   4. 将应用程序语言设置为 Python 并使用 Standard 环境。
   5. 在右上角，选择终端图标。这将激活cloud shell，这意味着您不必下载云 SDK 。
   6. 在部署应用程序之前，必须上载所有代码。从cloud shell中克隆此存储库。
      1. 将代码 URL 复制到剪贴板并导航回 GCP 上的cloud shell。向 shell 输入以下命令：
      2. git clone https://github.com/kurtispykes/gcp-deployment-example.git
      3. 输入以下命令导航到代码存储库：
      4.  cd gcp-deployment-example/app_engine
   7. 接下来， initialize the application 。确保您选择了最近创建的项目。

4. 部署应用程序。
   1. 从云 shell 运行以下命令。如果系统提示您继续，请输入Y。
   2. gcloud app deploy
   3. 部署完成后，您将获得服务部署位置的 URL 。
   4. 打开提供的 URL 以验证应用程序是否正常运行。您应该看到 欢迎使用我的 Flask API 消息。

5. 测试/predict端点。
   1. 使用 Postman 向/predict端点发送 POST 请求。
   2. 从那里，选择 Workspaces 、 My Workspace 、 New ，然后选择 HTTP Request 。
   3. 接下来，将 HTTP 请求从GET更改为POST，并在请求 URL 中插入到已部署服务的链接。
   4. 之后，导航到Body标头并选择raw，以便插入示例实例。选择send。
   5. 您向/predict端点发送了 POST 请求，其中包含一些定义模型输入的原始数据。在响应中，模型返回[‘Iris-setosa’]，这是模型成功部署的积极指示。

![predict-endpoint](/assets/img/predict-endpoint.png)

---

## 在 Google Cloud Function 上部署 ML 模型

> 最明显的区别是不再从本地存储库导入序列化模型。相反，您正在调用 Google 云存储中的模型。

1. 将模型上传到 Google 云存储
   1. 导航到 Cloud Storage 并选择 Buckets 、 Create Bucket 。命名为model-data-iris。
   2. 上传持久化模型。选择 Upload Files ，导航到存储模型的位置，然后选择它。
   3. 现在，您可以使用 Google Cloud 中的各种服务来访问此文件。要访问云存储，必须从google.cloud导入storage对象。

2. 从 Google 云存储中访问模型

    ```py
    import joblib
    import numpy as np
    from flask import request
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.get_bucket("model-iris-data")
    blob = bucket.blob("logistic_regression_v1.pkl")
    blob.download_to_filename("/tmp/logistic_regression_v1.pkl")
    model = joblib.load("/tmp/logistic_regression_v1.pkl")

    def predict(request):
        data_json = request.get_json()

        sepal_length_cm = data_json["sepal_length_cm"]
        sepal_width_cm = data_json["sepal_width_cm"]
        petal_length_cm = data_json["petal_length_cm"]
        petal_width_cm = data_json["petal_width_cm"]

        data = np.array([[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]])
        predictions = model.predict(data)

        return str(predictions)
    ```

3. Cloud Functions
   1. Create Function 。
      1. 要求您启用 API 。选择 Enable 继续。
      2. 函数名称= Predict 。
      3. Trigger type = HTTP 。
      4. Allow unauthenticated invocations =已启用。
      5. 默认值是可以的，因此选择 Next 。
   2. 设置运行时并定义源代码的来源。
      1. 在 Runtime 部分，选择您正在使用的 Python 版本。
      2. 确保在源代码头中选择了 Inline Editor 。
      3. 复制并粘贴云函数用作 main.py 文件入口点的以下代码示例。
        ```yaml
        {
          "sepal_length_cm" : 5.1,
          "sepal_width_cm" : 3.5,
          "petal_length_cm" : 1.4,
          "petal_width_cm" : 0.2
        }
        ```
      4. 使用内联编辑器更新 requirements.txt ：
        ```bash
        flask >= 2.2.2, <2.3.0
        numpy >= 1.23.3, <1.24.0
        scitkit-learn >=1.1.2, <1.2.0
        google-cloud-storage >=2.5.0, <2.6.0
        ```

   3. 将 Entry point 值更改为端点的名称。
      1. 在这种情况下，它是predict。

   4. 完成所有更改后，选择 Deploy 。
      1. 部署可能需要几分钟的时间来安装依赖项并启动应用程序。
      2. 完成后，您会看到成功部署的模型的函数名称旁边有一个绿色的勾号图标。

4. 现在，您可以在 Testing 选项卡上测试应用程序是否正常工作。
   1. 使用以下示例代码进行测试：
      ```yaml
      {
        "sepal_length_cm" : 5.1,
        "sepal_width_cm" : 3.5,
        "petal_length_cm" : 1.4,
        "petal_width_cm" : 0.2
      }
      ```

5. 使用此部署，您不必担心服务器管理。您的Cloud Function 仅在收到请求时执行，并且 Google 管理服务器。

---


## 在 Google AI 云上部署 ML 模型

> 之前的两个部署要求您编写不同程度的代码。在谷歌人工智能云上，你可以提供经过训练的模型，他们为你管理一切。

1. 导航到 AI Platform 。
   1. 在 Models 选项卡上，选择 Create Model 。
   2. 选择一个区域。选择区域后，选择 Create Model 。为模型命名，相应地调整区域，然后选择 Create 。
   3. 转到创建模型的区域，您应该可以看到模型。选择型号并选择 Create a Version 。

2. 接下来，将模型链接到云存储中存储的模型。本节有几个重要事项需要注意：

   1. AI 平台上scikit-learn的最新模型框架版本是 1.0.1 版，因此您必须使用此版本来构建模型。
   2. 模型必须存储为model.pkl或model.joblib。
   3. 为了遵守 GCP AI 平台的要求，我使用所需的模型版本创建了一个新的脚本，将模型序列化为model.pkl，并将其上传到谷歌云存储。有关更多信息，请参阅 /kurtispykes/gcp-deployment-example GitHub repo 中的更新代码。

   4. Model name: logistic_regression_model
   5. 选中 Use regional endpoint 复选框。
   6. Region: 欧洲西部 2
   7. 在 models 部分，确保仅选择 europe-west2 区域。
3. 为要创建的模型版本选择 Save 。创建模型版本可能需要几分钟的时间。

4. 通过选择模型版本并导航到 Test & Use 标题来测试模型。
5. 输入输入数据并选择 Test 。

---


##  Google Cloud Run

1. 应用打包成 Docker 之后，剩下的任务包括：
   1. 把镜像推送到 Google 镜像库。
   1. 运行 gcloud beta run deploy
   1. 只要几分钟，Cloud Run 就会使用一个可定制、可开放的域名启动新应用了。

示例：使用 Semaphore 进行持续部署, 为一个微服务配置 Serverless CI/CD Pipeline
1. 运行自动测试；
1. 构建 Docker 容器；
1. 将容器镜像推入 Google 镜像库；
1. 提供到 Cloud Run 预备环境的一键部署；
1. 在对 Master 分支的构建成功之后；自动部署到 Cloud Run 生产环境。

semaphore

可以在 Github 上找到相关的全部代码。

1. 启用 Cloud Run
   1. 中启用 Cloud Run API；
   2. 安装 Google Cloud SDK；
   3. 使用 gcloud components install beta 安装 Beta 组件。

2. 应用容器化
   1. Dockerfile 将一个简单的 Sinatra 应用打包
   2. 注意如果使用你自己的 Dockerfile，必须开放 8080 端口，否则可能会看到错误：

      ```dockerfile
      FROM ruby:2.5
      RUN apt-get update -qq &amp;&amp; apt-get install -y build-essential
      ENV APP_HOME /app
      RUN mkdir $APP_HOME
      WORKDIR $APP_HOME
      ADD Gemfile* $APP_HOME/
      RUN bundle install --without development test
      ADD . $APP_HOME
      EXPOSE 8080
      CMD ["bundle", "exec", "rackup", "--host", "0.0.0.0", "-p", "8080"]
      ```

3. 登录 Google Cloud 和 GCR
   1. 要在 CI/CD Pipeline 中自动地将镜像推送到 GCR，需要在 Semaphore 中登录到 Google Cloud。
   2. 为了安全起见，需要在 Semaphore 中根据 Google Cloud Service account 的认证密钥创建一个 Secret。
   3. 获取认证密钥之后，在 Semaphore 中用 Secret 的形式上传到 Semaphore。假设文件名是 .secrets.gcp.json：
   4. sem create secret google-cloud-stg --file ~/Downloads/account-name-27f3a5bcea2d.json:.secrets.gcp.json


4. 定义分发管线
   1. 编写一个 Pipeline 来构建、标记并推送镜像到 GCR 了：

      ```yaml
      # .semaphore/docker-build.yml
      # This pipeline runs after semaphore.yml
      version: v1.0
      name: Docker build
      agent:
        machine:
          # Use a machine type with more RAM and CPU power for faster container
          # builds:
          type: e1-standard-4
          os_image: ubuntu1804
      blocks:
        - name: Build
          task:
            # Mount a secret which defines an authentication key file.
            # For info on creating secrets, see:
            # - https://docs.semaphoreci.com/article/66-environment-variables-and-secrets
            # - https://docs.semaphoreci.com/article/72-google-container-registry-gcr
            secrets:
              - name: google-cloud-stg
            jobs:
            - name: Docker build
              commands:
                # Authenticate using the file injected from the secret
                - gcloud auth activate-service-account --key-file=.secrets.gcp.json
                # Configure access to container registry, silence confirmation prompts with -q
                - gcloud auth configure-docker -q
                - checkout
                # Tag the images with gcr.io/ACCOUNT_PROJECT_NAME/SERVICE_NAME pattern
                # Use Git SHA to produce unique artifacts
                - docker build -t "gcr.io/semaphore2-stg/semaphore-demo-cloud-run:${SEMAPHORE_GIT_SHA:0:7}" .
                - docker push "gcr.io/semaphore2-stg/semaphore-demo-cloud-run:${SEMAPHORE_GIT_SHA:0:7}"

      promotions:
        # Deployment to staging can be trigger manually:
        - name: Deploy to staging
          pipeline_file: deploy-staging.yml

        # Automatically deploy to production on successful builds on master branch:
        - name: Deploy to production
          pipeline_file: deploy-production.yml
          auto_promote_on:
            - result: passed
              branch:
                - master
      ```

      在 deploy-staging.yml 和 deploy-production.yml 中包含了同样的步骤，区别只是服务的名称。

      ```yaml
      # .semaphore/deploy-production.yml
      # This pipeline runs after docker-build.yml
      version: v1.0
      name: Deploy to production
      agent:
        machine:
          type: e1-standard-2
          os_image: ubuntu1804
      blocks:
        - name: Deploy to production
          task:
            secrets:
              - name: google-cloud-stg
            jobs:
            - name: run deploy
              commands:
                - gcloud auth activate-service-account --key-file=.secrets.gcp.json
                - gcloud auth configure-docker -q

                # Deploy to Cloud Run, using flags to avoid interactive prompt
                # See https://cloud.google.com/sdk/gcloud/reference/beta/run/deploy
                - gcloud beta run deploy markoci-demo-cloud-run --project semaphore2-stg --image gcr.io/semaphore2-stg/markoci-demo-cloud-run:${SEMAPHORE_GIT_SHA:0:7} --region us-central1
      ```

5. 上线运行
   1. 在本地终端或者 Semaphore 作业的日志中，最后一行会包含一个应用运行的网址：
   2. https://semaphore-demo-cloud-run-ud2bmvsmda-uc.a.run.app.
   3. 用浏览器打开这个网址会看到：forbidden
   4. 这是因为还没有完成最后一步：在 Google Cloud Run 控制台中开放服务
   5. 完成之后的浏览页面：hello

---

## App migrate

Setup:
1. Set Up the GCP Environment:

   1. Install Google Cloud SDK

   2. Authenticate and set the project:
      ```bash
      Copy code
      gcloud auth login
      gcloud config set project [The_PROJECT_ID]
      ```

2. Enable Vertex AI API
   1. Enable the Vertex AI API in your project:
      ```bash
      gcloud services enable aiplatform.googleapis.com
      ```

2. Set Up Authentication and Permissions

   1. Ensure that the service account used by your Cloud Run service has the necessary permissions. Assign the required roles to the service account.

   2. Create a Service Account:

      ```bash
      gcloud iam service-accounts create vertex-ai-sa \
          --description="Service account for Vertex AI access" \
          --display-name="Vertex AI Service Account"
      ```

   3. Grant the necessary roles to the service account: For accessing the Vertex AI, you might need roles like roles/aiplatform.admin or roles/aiplatform.user.

      ```bash
      gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] \
          --member="serviceAccount:vertex-ai-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com" \
          --role="roles/aiplatform.user"
      ```
<!--
   4. Create a key for the service account:

      ```bash
      gcloud iam service-accounts keys create key.json \
          --iam-account=vertex-ai-sa@[YOUR_PROJECT_ID].iam.gserviceaccount.com
      ```


3. Store Service Account Key in Secret Manager
   1. Enable Secret Manager API:

      ```bash
      gcloud services enable secretmanager.googleapis.com
      ```

   2. Create a new secret and store the service account key:
      ```bash
      gcloud secrets create vertex-ai-sa-key --data-file=key.json
      ```

   3. Grant the Secret Accessor role to the Cloud Run service account:

      ```bash
      gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] \
          --member="serviceAccount:[YOUR_CLOUD_RUN_SERVICE_ACCOUNT]" \
          --role="roles/secretmanager.secretAccessor"
      ```  -->

1. Upload the Configuration Files:

   1. Create a Cloud Storage bucket:
      ```bash
      gsutil mb gs://[YOUR_BUCKET_NAME]/
      ```

   2. Upload your configuration file to the bucket:
      ```bash
      gsutil cp tests/java-cwe/myconfig.yaml gs://[YOUR_BUCKET_NAME]/
      ```

On Container:

1. Build and Push Your Docker Image to Google Container Registry (GCR):

   1. Tag your Docker image:
      ```bash
      docker tag [YOUR_IMAGE] gcr.io/[YOUR_PROJECT_ID]/my_app:latest
      ```

   2. Authenticate with GCR:
      ```bash
      gcloud auth configure-docker
      ```

   3. Push your Docker image to GCR:
      ```bash
      docker push gcr.io/[YOUR_PROJECT_ID]/my_app:latest
      ```

2. Create a Script to Run Inside the Container:

   1. Create a script run_my_app.sh:

      ```bash
      #!/bin/bash

      # Initialize Google Cloud Logging
      pip install google-cloud-logging
      python -c "from google.cloud import logging; logging_client = logging.Client(); logging_client.setup_logging()"

      # Download configuration file from Cloud Storage
      gsutil cp gs://[YOUR_BUCKET_NAME]/my_app.yaml /app/my_app.yaml

      # Log the start of the operation
      echo "Starting my_app evaluation" | logger -s

      # Run the command
      my_app eval --config /app/my_app.yaml \
        --no-cache \
        --output /app/output-my_app-1.csv \
        --delay 100

      # Log the completion of the operation
      echo "Completed my_app evaluation" | logger -s

      # Upload the output file to Cloud Storage
      gsutil cp /app/output-my_app-1.csv gs://[YOUR_BUCKET_NAME]/

      # Log the file upload completion
      echo "Uploaded output file to Cloud Storage" | logger -s
      ```

      ```py
      import logging
      from google.cloud import logging as cloud_logging
      import subprocess
      import os

      # Initialize Google Cloud Logging
      cloud_logging_client = cloud_logging.Client()
      cloud_logging_client.setup_logging()

      # Set up logging
      logger = logging.getLogger()
      logger.setLevel(logging.INFO)

      def run_command(command):
          process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
          stdout, stderr = process.communicate()
          return process.returncode, stdout, stderr

      # Download configuration file from Cloud Storage
      logger.info("Downloading configuration file from Cloud Storage")
      return_code, stdout, stderr = run_command("gsutil cp gs://[YOUR_BUCKET_NAME]/my_app.yaml /app/my_app.yaml")
      if return_code != 0:
          logger.error(f"Failed to download configuration file: {stderr.decode()}")
          exit(1)

      # Run the command
      logger.info("Starting my_app evaluation")
      return_code, stdout, stderr = run_command("my_app eval --config /app/my_app.yaml --no-cache --output /app/output-my_app-1.csv --delay 1000")
      if return_code != 0:
          logger.error(f"Failed to run my_app: {stderr.decode()}")
          exit(1)
      logger.info(f"my_app evaluation completed: {stdout.decode()}")

      # Upload the output file to Cloud Storage
      logger.info("Uploading output file to Cloud Storage")
      return_code, stdout, stderr = run_command("gsutil cp /app/output-my_app-1.csv gs://[YOUR_BUCKET_NAME]/")
      if return_code != 0:
          logger.error(f"Failed to upload output file: {stderr.decode()}")
          exit(1)
      logger.info("Output file uploaded successfully")

      ```

   2. Update Your Python Script to Handle HTTP Requests:
      1. Use Flask to create a simple web server that handles HTTP requests and executes the command with the provided parameters.

      ```py
      import logging
      from google.cloud import logging as cloud_logging
      from google.cloud import aiplatform
      from flask import Flask, request, jsonify
      import subprocess
      import os
      import json

      # Initialize Google Cloud Logging
      cloud_logging_client = cloud_logging.Client()
      cloud_logging_client.setup_logging()

      # Set up logging
      logger = logging.getLogger()
      logger.setLevel(logging.INFO)

      # Function to get access token
      def get_access_token():
          metadata_server_token_url = 'http://metadata/computeMetadata/v1/instance/service-accounts/default/token'
          token_request_headers = {'Metadata-Flavor': 'Google'}
          token_response = requests.get(metadata_server_token_url, headers=token_request_headers)
          token_response.raise_for_status()
          return token_response.json()['access_token']

      def run_command(command):
          process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
          stdout, stderr = process.communicate()
          return process.returncode, stdout, stderr

      app = Flask(__name__)

      @app.route('/run', methods=['POST'])
      def run():
          # Get parameters from the request
          config = request.json.get('config', '/app/my_app.yaml')
          no_cache = request.json.get('no_cache', True)
          output = request.json.get('output', '/app/output-my_app-1.csv')
          delay = request.json.get('delay', 1000)

          # Download configuration file from Cloud Storage
          logger.info("Downloading configuration file from Cloud Storage")
          return_code, stdout, stderr = run_command(f"gsutil cp gs://[YOUR_BUCKET_NAME]/{config} /app/my_app.yaml")
          if return_code != 0:
              logger.error(f"Failed to download configuration file: {stderr.decode()}")
              return jsonify({'status': 'error', 'message': 'Failed to download configuration file', 'details': stderr.decode()}), 500

          # Build the my_app command
          no_cache_flag = '--no-cache' if no_cache else ''
          command = f"my_app eval --config /app/my_app.yaml {no_cache_flag} --output {output} --delay {delay}"

          # Get the access token
          access_token = get_access_token()
          os.environ['VERTEX_API_KEY'] = access_token

          # Run the command
          logger.info(f"Starting my_app evaluation with command: {command}")
          return_code, stdout, stderr = run_command(command)
          if return_code != 0:
              logger.error(f"Failed to run my_app: {stderr.decode()}")
              return jsonify({'status': 'error', 'message': 'Failed to run my_app', 'details': stderr.decode()}), 500

          logger.info(f"my_app evaluation completed: {stdout.decode()}")

          # Upload the output file to Cloud Storage
          logger.info("Uploading output file to Cloud Storage")
          return_code, stdout, stderr = run_command(f"gsutil cp {output} gs://[YOUR_BUCKET_NAME]/")
          if return_code != 0:
              logger.error(f"Failed to upload output file: {stderr.decode()}")
              return jsonify({'status': 'error', 'message': 'Failed to upload output file', 'details': stderr.decode()}), 500

          logger.info("Output file uploaded successfully")
          return jsonify({'status': 'success', 'message': 'Output file uploaded successfully'})

      if __name__ == '__main__':
          app.run(host='0.0.0.0', port=8080)
      ```

3. Create a requirements.txt File:

   1. List the dependencies for the Python script.
      ```bash
      Flask==2.1.1
      google-cloud-logging==3.0.0
      google-cloud-aiplatform
      ```

4. Modify Your Dockerfile to Include the Script:

   1. Add the script to your Docker image and set it as the entry point.

   2. Make sure the Google Cloud Logging library is installed.

      ```Dockerfile
      FROM python:3.8-slim

      # Install dependencies
      COPY requirements.txt /app/requirements.txt
      RUN pip install --no-cache-dir -r /app/requirements.txt

      # Copy the application
      COPY run_my_app.sh /app/run_my_app.sh
      RUN chmod +x /app/run_my_app.sh

      # Set default environment variables
      ENV VERTEX_API_KEY=""
      ENV VERTEX_PROJECT_ID=""
      ENV VERTEX_REGION="us-central1"
      ENV my_app_DISABLE_TELEMETRY="true"

      # Set the working directory
      WORKDIR /app

      # Run the application
      ENTRYPOINT ["/app/run_my_app.sh"]
      ```

5. Rebuild and Push Your Docker Image:
   1. Build your Docker image:
      ```bash
      docker build -t gcr.io/[YOUR_PROJECT_ID]/my_app:latest .
      ```

   2. Push your Docker image to GCR:
      ```bash
      docker push gcr.io/[YOUR_PROJECT_ID]/my_app:latest
      ```

   3. Create a Docker Repository in Artifact Registry to store your Docker images.
      ```bash
      gcloud artifacts repositories create [REPOSITORY_NAME] \
         --repository-format=docker \
         --location=[LOCATION] \
         --description="Docker repository"
      ```

   4. Configure Docker to Authenticate with Artifact Registry to use your Google credentials for pushing and pulling images.
      ```bash
      gcloud auth configure-docker [LOCATION]-docker.pkg.dev
      ```

   5. Update Your Docker Commands to Use Artifact Registry, Modify your Docker build and push commands to use the Artifact Registry path.
      ```bash
      docker build -t [LOCATION]-docker.pkg.dev/[YOUR_PROJECT_ID]/[REPOSITORY_NAME]/my_app:latest .
      docker push [LOCATION]-docker.pkg.dev/[YOUR_PROJECT_ID]/[REPOSITORY_NAME]/my_app:latest
      ```

6. Deploy/Update to Google Cloud Run Service:

   1. Deploy your Docker image to Cloud Run:
      ```bash
      gcloud run deploy my_app-service \
          --image gcr.io/[YOUR_PROJECT_ID]/my_app:latest \
          --image [LOCATION]-docker.pkg.dev/[YOUR_PROJECT_ID]/[REPOSITORY_NAME]/my_app:latest \
          --platform managed \
          --region [YOUR_PREFERRED_REGION] \
          --allow-unauthenticated
      # Note the URL provided after deployment. This is your Cloud Run service URL.
      ```

      ```bash
      gcloud run deploy my_app-service \
          --image gcr.io/[YOUR_PROJECT_ID]/my_app:latest \
          --platform managed \
          --region [YOUR_PREFERRED_REGION] \
          --allow-unauthenticated \
          --update-secrets "VERTEX_AI_SA_KEY=vertex-ai-sa-key:latest" \
          --port 8080
      ```

7. Trigger the Cloud Run Service:

   1. You can now trigger the Cloud Run service by making an HTTP request to the service URL.

   2. This can be done manually via curl or programmatically via any HTTP client.
   ```bash
   curl -X POST [CLOUD_RUN_SERVICE_URL]
   ```

   3. trigger the Cloud Run service by making an HTTP POST request with JSON payload containing the parameters.
   ```bash
   curl -X POST [CLOUD_RUN_SERVICE_URL]/run \
       -H "Content-Type: application/json" \
       -d '{
           "config": "my_app.yaml",
           "no_cache": true,
           "output": "/app/output-my_app-1.csv",
           "delay": 1000
       }'
   ```

8. Access Logs in Google Cloud Console:
   1. Go to the Logging section:
   2. Select Logging from the sidebar, then select Log Viewer.
   3. Filter Logs:
      1. In the Log Viewer, you can filter logs by resource type (Cloud Run Revision) and the specific service (my_app-service).
   4. View Logs:
      1. You should see logs for each request and execution of your Cloud Run service, including any logs written by your application using logger.



On VM:
1. Prepare a GCP Virtual Machine (VM):

   1. Create a VM instance:
      ```bash
      gcloud compute instances create my_app-vm \
          --machine-type=e2-medium \
          --image-project=debian-cloud \
          --image-family=debian-11 \
          --scopes=https://www.googleapis.com/auth/cloud-platform
      ```

   2. SSH into the VM:
      ```bash
      gcloud compute ssh my_app-vm
      ```

2. Install Required Software on the VM:

   1. Update and install dependencies:
      ```bash
      sudo apt-get update
      sudo apt-get install -y python3-pip
      ```

   2. Install my_app (assuming it's a Python package, replace if necessary):
      ```bash
      pip3 install my_app
      ```

   3. Install Google Cloud Storage client library:
      ```bash
      pip3 install google-cloud-storage
      ```

3. Download Your Configuration File from Cloud Storage:

   1. upload the files:
      ```bash
      gcloud compute scp ./test/myconfig.yaml my_app-vm:~/
      ```

   2. Create a script download_files.sh:
      ```bash
      #!/bin/bash
      gsutil cp gs://[YOUR_BUCKET_NAME]/myconfig.yaml ~/
      ```

   3. Upload the script and make it executable:
      ```bash
      gcloud compute scp download_files.sh my_app-vm:~/
      gcloud compute ssh my_app-vm
      chmod +x ~/download_files.sh
      ```

   4. Run the script to download the configuration file:
      ```bash
      ~/download_files.sh
      ```

4. Run the Command on the VM:

   1. SSH into the VM
      ```bash
      gcloud compute ssh my_app-vm
      my_app eval --config ~/myconfig.yaml \
        --no-cache \
        --output ~/output-config-1.csv \
        --delay 1000
      ```

5. Transfer Output File to Google Cloud Storage:

    ```bash
    gsutil cp ~/output-config-1.csv gs://[YOUR_BUCKET_NAME]/
    ```


7. Automate the Process with a Script (Optional):

   1. Create a script run_my_app.sh:
      ```bash
      #!/bin/bash
      my_app eval --config ~/config.yaml \
        --no-cache \
        --output ~/output-config-1.csv \
        --delay 1000

      gsutil cp ~/output-config-1.csv gs://[YOUR_BUCKET_NAME]/
      ```

   2. Upload the script and make it executable:
      ```bash
      gcloud compute scp run_my_app.sh my_app-vm:~/
      gcloud compute ssh my_app-vm
      chmod +x ~/run_my_app.sh
      ```

   3. Run the script:
      ```bash
      ~/run_my_app.sh
      ```

8. Schedule the Script (Optional):

   1. Use cron to schedule the script:
      ```bash
      crontab -e
      0 0 * * * ~/run_my_app.sh
      ```




.
