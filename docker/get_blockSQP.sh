#!/bin/bash

set -e
mkdir -p /tmp/blockSQP
cd /tmp/blockSQP
git clone https://github.com/matthias-k/blockSQP
cd blockSQP
git checkout fix_missing_virtual_destructor
sed -i "s/QPOASESDIR = .*/QPOASESDIR = \/opt\/qpOASES-3.2.0/g" makefile
make
cd ..
mv blockSQP /opt/

