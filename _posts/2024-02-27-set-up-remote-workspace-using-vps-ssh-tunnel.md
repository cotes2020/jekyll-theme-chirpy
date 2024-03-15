---
published: false
date: 2024-02-27
title: Set up remote workspace using VPS, SSH tunnel
---
```
ssh -N -A \
    -L 80:devlocal.viclass.vn:80 \
    -L 443:devlocal.viclass.vn:443 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o GatewayPorts=yes \
    -p 22 root@103.186.101.191
```