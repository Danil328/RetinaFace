all:
	cd retinaface/cython/; python setup.py build_ext --inplace; rm -rf build; cd ../../
	cd retinaface/pycocotools/; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd retinaface/cython/; rm *.so *.c *.cpp; cd ../../
	cd retinaface/pycocotools/; rm *.so; cd ../../
