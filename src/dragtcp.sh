#!/bin/sh

DEFAULT_IP=129.89.57.142

if test $# -gt 1
then
  echo "Usage: $0 [ipaddr]"
  exit 1
fi

if test $# -eq 0
then
  IPADDR=$DEFAULT_IP
else
  IPADDR=$1
fi

echo "*** Using IP address $IPADDR"

echo "*** Starting remote process for simplex test"
rsh -n $IPADDR "`pwd`/dragtcp -r &" &
if [ "$?" -ne "0" ]
then
  echo "!!! Failed"
  exit 1
fi

sleep 15

echo "*** Starting local process for simplex test"
./dragtcp -l -i $IPADDR 1>"dragtcp-simplex.out"
if [ "$?" -ne "0" ]
then
  echo "!!! Failed"
  exit 1
fi

echo "*** Simplex test complete"


echo "*** Starting remote processes for duplex test"
rsh -n $IPADDR "`pwd`/dragtcp -r -p 2001 &" &
if [ "$?" -ne "0" ]
then
  echo "!!! Failed"
  exit 1
fi

rsh -n $IPADDR "`pwd`/dragtcp -r -p 2002 &" &
if [ "$?" -ne "0" ]
then
  echo "!!! Failed"
  exit 1
fi

sleep 30

echo "*** Starting local processes for duplex test"
./dragtcp -l -i $IPADDR -p 2001    1>"dragtcp-duplex-one.out" &
./dragtcp -l -i $IPADDR -p 2002 -R 1>"dragtcp-duplex-two.out" &

echo "*** Duplex test complete"
echo "*** Results are in files:"
echo "        dragtcp-simplex.out"
echo "        dragtcp-duplex-one.out"
echo "        dragtcp-duplex-two.out"
exit 0
