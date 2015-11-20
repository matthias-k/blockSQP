.PHONY: build
build:
	rm -rf build
	rm -f pyBlockSQP.so
	rm -f src/pyBlockSQP_api.h
	rm -f src/pyBlockSQP.cpp
	rm -f src/pyBlockSQP.h
	python setup.py build_ext --inplace

shell:
	LD_LIBRARY_PATH=/opt/blockSQP/lib/:/opt/qpOASES-3.2.0/bin ipython2
