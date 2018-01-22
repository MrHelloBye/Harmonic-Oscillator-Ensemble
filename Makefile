ho-ensemble.x: main.cu vose.cpp vose.h
	export PATH=/Developer/NVIDIA/CUDA-9.1/bin${PATH:+:${PATH}}
	export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-9.1/lib ${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
	nvcc *.cu *.cpp -std=c++11 -o ho-ensemble.x
