#!/bin/bash

mkdir -p /tmp/Qpoases
cd /tmp/Qpoases
wget http://www.qpoases.org/go/release -O qpOASES-3.2.0.zip
unzip qpOASES-3.2.0.zip
cd qpOASES-3.2.0
make
cd ..
mkdir -p /opt/
cp -r qpOASES-3.2.0 /opt/
rm -rf /tmp/Qpoases
