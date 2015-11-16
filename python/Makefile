.PHONY: build
build:
	rm -rf build
	rm -f pyBlockSQP.so
	python setup.py build_ext --inplace

shell:
	LD_LIBRARY_PATH=/opt/blockSQP/lib/:/opt/qpOASES-3.2.0/bin ipython2
