#!/bin/bash

if [ "$1" = "off" ]; then
    unset http_proxy
    unset https_proxy
    echo "Proxy disabled"
else
    export http_proxy=http://127.0.0.1:9090
    export https_proxy=http://127.0.0.1:9090
    echo "Proxy enabled: http://127.0.0.1:9090"
fi