#!/bin/sh
while [ "$#" -gt "0" ]
do
  echo "\$1 is $1"
  # shift命令
  shift
done              


#!/bin/sh
/usr/local/bin/my-command
if [ "$?" -ne "0" ]; then
  echo "Sorry, we had a problem there!"
fi