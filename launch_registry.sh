#!/usr/bin/env bash
cd /home/randozart/Desktop/Projects/OurobourOS
pkill -f 'ouro-registr[y] --addr' 2>/dev/null
sleep 1
export OURO_SECRET_FILE=/home/randozart/Desktop/Projects/OurobourOS/enroll/secret
setsid nohup ./target/release/ouro-registry --addr 0.0.0.0:9501 --state registry.json > registry-live.log 2>&1 < /dev/null &
echo "registry launched"
