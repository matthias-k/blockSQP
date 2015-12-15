#!/bin/bash

set -e

echo "Installing in $CUTESTROOT"
mkdir -p $CUTESTROOT
cd $CUTESTROOT
export CUTESTROOT=/opt/cutest
svn checkout --username anonymous http://ccpforge.cse.rl.ac.uk/svn/cutest/archdefs/trunk ./archdefs
svn checkout --username anonymous http://ccpforge.cse.rl.ac.uk/svn/cutest/sifdecode/trunk ./sifdecode
svn checkout --username anonymous http://ccpforge.cse.rl.ac.uk/svn/cutest/cutest/trunk ./cutest
svn checkout --username anonymous http://ccpforge.cse.rl.ac.uk/svn/cutest/sif/trunk ./sif
cd cutest
expect /tmp/install_cutest
