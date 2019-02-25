
#!/bin/sh
# telnet2.sh | telnet > FILE1 
host=127.0.0.1
port=23
login=steve
passwd=hellothere
cmd="ls /tmp"
timeout=3
file=file1
prompt="$"

echo open ${host} ${port}
sleep 1
tout=${timeout}
while [ "${tout}" -ge 0 ]
do
    if tail -1 "${file}" 2>/dev/null | egrep -e "login:" > /dev/null
    then
        echo "${login}"
        sleep 1
        tout=-5
        continue
    else
        sleep 1
        tout=`expr ${tout} - 1`
    fi
done

if [ "${tout}" -ne "-5" ]; then
  exit 1
fi

tout=${timeout}
while [ "${tout}" -ge 0 ]
do
    if tail -1 "${file}" 2>/dev/null | egrep -e "Password:" > /dev/null
    then
        echo "${passwd}"
        sleep 1
        tout=-5
        continue
    else
      if tail -1 "${file}" 2>/dev/null | egrep -e "${prompt}" > /dev/null
      then
        tout=-5
      else
        sleep 1
        tout=`expr ${tout} - 1`
      fi
    fi
done

if [ "${tout}" -ne "-5" ]; then
  exit 1
fi

> ${file}

echo ${cmd}
sleep 1
echo exit


# $ ./telnet2.sh | telnet > file1