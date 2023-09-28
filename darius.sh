#!/bin/bash

msg=$1

git pull && git add . && git commit -m "Darius: update $msg" && git push