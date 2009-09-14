EXECUTABLE  := exec
OBJECTS := exec.o



include $(HOME)/paths.mk

CXXFLAGS := -O2 -use_fast_math -I$(CUDAINCPATH)
NVCCFLAGS := -O2 -use_fast_math -I$(CUTILPATH) -Xcompiler -fPIC

LDFLAGS := -lcutil -lcuda -lcudart -L$(CUDALIBPATH) -L$(CUTILPATH) -L$(HOME)/lib

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<
%.cubin: %.cu
	$(NVCC) -cubin $(NVCCFLAGS) -o $@ $<
%.ptx: %.cu
	$(NVCC) -ptx $(NVCCFLAGS) -o $@ $<

all: $(EXECUTABLE) insttest.cubin

$(EXECUTABLE): $(OBJECTS) $(LIB_OBJECTS)
	g++ $(OBJECTS)  $(LIB_OBJECTS) $(LDFLAGS) -o $(EXECUTABLE)

