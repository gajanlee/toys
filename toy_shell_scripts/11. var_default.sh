#!/bin/sh
echo -en "What is your name [ `whoami` ] "
read myname
echo "Your name is : ${myname:-`whoami`}"

# -代表默认值，如果myname不存在的话，用`whoami`命令返回值填充