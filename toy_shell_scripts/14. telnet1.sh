#!/bin/sh
host=127.0.0.1
# telnet port
port=23
login=lee
passwd=123q456w
cmd="ls /tmp"

echo open ${host} ${port}
sleep 1
echo ${login}
sleep 1
echo ${passwd}
sleep 1
echo ${cmd}
sleep 1
echo exit

# ./telnet1.sh | telnet