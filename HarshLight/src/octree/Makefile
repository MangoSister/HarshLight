EXECUTABLE := octree

CU_FILES   := octree.cu

CU_DEPS    :=

CC_FILES   := main.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=os
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -std=c++11
LDFLAGS=-L/usr/local/cuda-7.5/lib64/ -lcudart
NVCC=nvcc -arch=sm_20
NVCCFLAGS=-O3 -m64


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/octree.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
