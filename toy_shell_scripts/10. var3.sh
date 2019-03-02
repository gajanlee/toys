#!/bin/sh
echo "I was called with $# parameters"
echo "My name is $0"
echo "My first parameter is $1"
echo "My second parameter is $2"
echo "All parameters are $@"
echo "Test \$* $*"

# $0 是调用的程序名
# $1 ... $9 是前9个参数
# $@ 是 $1...以后的所有参数
# $* 是后面的参数，但是不保留空格，会变成三个字符串
# $# 是参数个数
# $? 是上一个命令执行后的退出值
# $$ 是这个程序的PID
# $! 上一个后台运行程序的PID