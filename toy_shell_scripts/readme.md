# Shell学习记录

* 参考地址

```
https://www.shellscript.sh
```

##　笔记

* 参考资料
```
https://www.shellscript.sh/
```

* 命令与变量
```
VAR=value变量
VAR = value 调用VAR，参数为=和value
```

* 变量连接
```
使用大括号
touch "${USER_NAME}_file"
双引号防止变量之间有空格
```
* 通配符
```
Wildcards
```

* 列出目录
```
ls  /tmp/
echo /tmp/*
```
* 转义字符
```
双引号中的"$`\ 
```

* 命令中的循环
```
ls -ld {,usr,usr/local}/{bin,sbin,lib}
mkdir rc{0,1,2,3,4,5,6,S}.d
```

* 条件判断
```
if [ $foo = "bar" ]

其中[ 是一个程序，/usr/bin/[
连接到/usr/bin/test
所以如果[和$foo连在一起就成了
if test$foo = "bar" ]

if [ ... ]
then
    # if-code
else
    # else-code
fi

if [ ... ]; then
    ...
elif [ ... ]; then
    ...
else
    ...
fi

找出不是0-9的，用^
或者 grep -v [0-9]
echo $X | grep "[^0-9]" > /dev/null 2>&1
```

* cut命令，可以切分之后输出某一列
```
grep "^${USER}:" /etc/passwd | cut -d: -f5
```

* library
```
. ./library.sh
```

* tee命令
```
echo -n 是不带回车的
tee就是输出到终端,tee -a [file] 是 >>

2>&1 | tee   才能打印
echo是不行的
```

* 可以使用rename脚本
```
rename .html .html-bak
```
* tr命令
```
# 小写转大写
echo xxx | tr [a-z] [A-Z]
```

* sed 替换
```
sed s/eth0/eth1/g file1 >  file2
```