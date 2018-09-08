cmake_minimum_required(VERSION 3.12)
project(Lik_ML)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++1y -Wall -O3")

add_executable(Lik_ML
        main.cpp
        type/type.cpp type/type.h
        function/math/activation.hpp
        function/math/loss.hpp
        function/math/operation.cpp function/math/operation.h
        function/tool/random.hpp
        layer/Layer.cpp layer/Layer.h
        layer/bases/FullyConnected.cpp layer/bases/FullyConnected.h
        layer/bases/Loss.cpp layer/bases/Loss.h
        module/SequentialModule.cpp module/SequentialModule.h example/example.h example/example_xor.cpp layer/conv/Convolution2D.cpp layer/conv/Convolution2D.h example/example_XO.cpp)