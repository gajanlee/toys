#!/bin/sh
# This is a comment!
echo "Hello      World"       # This is a comment, too!
echo "Hello World"
echo "Hello * World"
# * 是当前目录下所有文件
echo Hello * World
echo Hello      World
echo "Hello" World
echo Hello "     " World
echo "Hello "*" World"
# 代表执行hello `hello`
echo `hello` world
echo 'hello' world

# chmod 755 3.\ first2.sh
# ./3.\ first2.sh