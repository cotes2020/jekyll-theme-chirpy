---
published: true
date: 2023-05-05
title: Auto Process Limiter for Linux
---
Updated:

    ulimit -Su $number_of_process # limit number of running process
    ulimit -Sm $kilobytes # limit kilobytes of RAM

* * *

    #!/bin/bash
    
    limitAmount=100
    limitMemory=1500000
    
    while true; do
        sleep 3
    
        reduntantProcesses=$(ps aux --sort=start_time | grep "^$(whoami)" | sed -n "$(($limitAmount+2)),999999p")
        if test -n "${reduntantProcesses}"; then
            kill "$(echo "$reduntantProcesses" | awk '{print $2}')"
        fi
    
        processes=$(ps aux --sort=-start_time | grep "^$(whoami)" | sed -n "2,999999p")
        memoryUsed=$(echo "$processes" | awk '{sum += $6}; END {print sum}')
        if [ "$memoryUsed" -gt "$limitMemory" ]; then
            for proc in $(echo "$processes" | awk '{print $2}'); do
                kill $proc
                _processes=$(ps aux --sort=-start_time | grep "^$(whoami)" | sed -n "2,999999p")
                _memoryUsed=$(echo "$_processes" | awk '{sum += $6}; END {print sum}')
                if [ "$_memoryUsed" -lt "$limitMemory" ]; then
                    break
                fi
            done
        fi
    done