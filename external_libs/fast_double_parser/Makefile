all: benchmark unit

LIBABSEIL:=benchmarks/dependencies/abseil-cpp/build/absl/strings/libabsl_strings.a
LIBABSEIL_INCLUDE:=-Ibenchmarks/dependencies/abseil-cpp
LIBABSEIL_LIBS:=-Lbenchmarks/dependencies/abseil-cpp/build/absl/base -Lbenchmarks/dependencies/abseil-cpp/build/absl/strings  -Lbenchmarks/dependencies/abseil-cpp/build/absl/numeric/  -labsl_strings   -labsl_raw_logging_internal -labsl_throw_delegate -labsl_int128


LIBDOUBLE:=benchmarks/dependencies/double-conversion/libdouble-conversion.a
LIBDOUBLE_INCLUDE:=-Ibenchmarks/dependencies/double-conversion
LIBDOUBLE_LIBS:=-Lbenchmarks/dependencies/double-conversion -ldouble-conversion


headers:=  include/fast_double_parser.h 

benchmark: ./benchmarks/benchmark.cpp $(headers) $(LIBABSEIL)  $(LIBDOUBLE) $(headers)
	$(CXX) -O2 -std=c++14 -march=haswell -o benchmark ./benchmarks/benchmark.cpp -Wall -Iinclude   $(LIBABSEIL_INCLUDE)  $(LIBDOUBLE_INCLUDE) $(LIBDOUBLE_LIBS) $(LIBABSEIL_LIBS)   -lm


unit: ./tests/unit.cpp $(headers) 
	$(CXX) -O2 -std=c++14 -march=native -o unit ./tests/unit.cpp -Wall -Iinclude 


bench: benchmark
	./benchmark 
	./benchmark benchmarks/data/canada.txt

submodules:
	-git submodule update --init --recursive
	-touch submodules


$(LIBABSEIL):submodules
	rm -r -f benchmarks/dependencies/abseil-cpp/build && cd benchmarks/dependencies/abseil-cpp && mkdir build && cd build && cmake .. -DABSL_RUN_TESTS=OFF -DABSL_USE_GOOGLETEST_HEAD=OFF -DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_TYPE=Release && cmake --build . --target base && cmake --build . --target strings

$(LIBDOUBLE):submodules
	cd benchmarks/dependencies/double-conversion && cmake .  -DCMAKE_BUILD_TYPE=Release && make

clean:
	rm -r -f benchmark unit benchmarks/dependencies/abseil-cpp/build
