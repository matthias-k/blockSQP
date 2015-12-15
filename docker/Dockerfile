FROM ipython/scipystack

RUN apt-get update && apt-get install -y wget git vim unzip make g++ subversion libgsl0-dev expect


RUN pip2 install --upgrade pip
RUN pip2 install --upgrade cython
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade cython

ADD get_qpOASES.sh /tmp/
RUN bash /tmp/get_qpOASES.sh

ENV LD_LIBRARY_PATH /opt/qpOASES-3.2.0/bin:$LD_LIBRARY_PATH

ADD get_blockSQP.sh /tmp/
RUN bash /tmp/get_blockSQP.sh

ENV LD_LIBRARY_PATH /opt/blockSQP/lib:$LD_LIBRARY_PATH

WORKDIR /opt/
RUN git clone --recursive https://github.com/r-gaia-cs/ugly.git
WORKDIR /opt/ugly
RUN ./configure && make && make install

#ADD get_cutest.sh /tmp/
#ADD install_cutest /tmp/
#ENV CUTESTROOT /opt/cutest/
#ENV ARCHDEFS $CUTESTROOT/archdefs
#ENV SIFDECODE $CUTESTROOT/sifdecode
#ENV CUTEST $CUTESTROOT/cutest
#RUN bash /tmp/get_cutest.sh
#ENV PATH ${SIFDECODE}/bin:${PATH}
#ENV PATH ${CUTEST}/bin:${PATH}
#ENV MANPATH ${SIFDECODE}/man:${MANPATH}
#ENV MANPATH ${CUTEST}/man:${MANPATH}
#ENV MYARCH pc64.lnx.gfo
#ENV MASTSIF $CUTESTROOT/sif

#ADD pycutest /opt/cutest/pycutest/
#ENV PYTHONPATH /opt/cutest/pycutest:$PYTHONPATH



#mkdir -p /tmp/blockSQP
#cd /tmp/blockSQP
#git clone https://github.com/djanka2/blockSQP
#cd blockSQP
#sed -i "s/QPOASESDIR = .*/QPOASESDIR = \/opt\/qpOASES-3.2.0/g" makefile
#make


#ADD get_ipopt.sh /tmp/
#ADD get_cyipopt.sh /tmp/
#RUN bash /tmp/get_ipopt.sh
#RUN source /etc/environment
#RUN bash /tmp/get_cyipopt.sh
#RUN rm /tmp/get_ipopt.sh /tmp/get_cyipopt.sh
