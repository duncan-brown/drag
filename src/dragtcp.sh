#!/bin/sh
DEFAULT_IP=129.89.57.90

echo "*** Starting remote process for simplex test"
rsh -n $DEFAULT_IP "`pwd`/dragtcp -r &" &
if [ "$?" -ne "0" ]
then
  echo "!!! Failed"
  exit 1
fi

sleep 15

echo "*** Starting local process for simplex test"
./dragtcp -l -i $DEFAULT_IP 1>"dragtcp-simplex.out"
if [ "$?" -ne "0" ]
then
  echo "!!! Failed"
  exit 1
fi

echo "*** Simplex test complete"


echo "*** Starting remote processes for duplex test"
rsh -n $DEFAULT_IP "`pwd`/dragtcp -r -p 2001 &" &
if [ "$?" -ne "0" ]
then
  echo "!!! Failed"
  exit 1
fi

rsh -n $DEFAULT_IP "`pwd`/dragtcp -r -p 2002 &" &
if [ "$?" -ne "0" ]
then
  echo "!!! Failed"
  exit 1
fi

sleep 30

echo "*** Starting local processes for duplex test"
./dragtcp -l -i $DEFAULT_IP -p 2001    1>"dragtcp-duplex-one.out" &
./dragtcp -l -i $DEFAULT_IP -p 2002 -R 1>"dragtcp-duplex-two.out" &

echo "*** Duplex test complete"
