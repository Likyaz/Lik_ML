#include "type.h"

namespace type
{
    matrix convertToMatrix(tensor val) {return *val;}

    vector convertToVector(tensor val) {return **val;}
    vector convertToVector(matrix val) {return *val;}

    vector createVector(int size) {
        vector value;
        value = new scalar[size] {0};
        return value;
    }

    matrix createMatrix(int sizeX, int sizeY) {
        vector memorySpace;
        matrix value;
        memorySpace = new scalar[sizeX * sizeY] {0};
        value = new vector[sizeY];
        for(int i(0); i < sizeY; i++) {
            value[i] = memorySpace + (i * sizeX);
        }
        return value;
    }

    tensor createTensor(int sizeX, int sizeY, int sizeZ) {
        vector memorySpace;
        matrix pMemorySpace;
        tensor value;

        memorySpace = new scalar[sizeX * sizeY * sizeZ] {0};
        pMemorySpace = new vector[sizeY * sizeZ] {nullptr};
        value = new matrix[sizeZ] {nullptr};

        for(int i(0); i < sizeY * sizeZ; i++) {
            pMemorySpace[i] = memorySpace + (i * sizeX);
        }

        for(int i(0); i < sizeZ; i++) {
            value[i] = pMemorySpace + (i * sizeY);
        }
        return value;
    }


    vector createRandVector(int size)
    {
        vector v = createVector(size);
        for(int i(0); i < size; i++) {
            v[i] = function::tool::nextRandTool();
        }
        return v;
    }

    matrix createRandMatrix(int sizeX, int sizeY)
    {
        matrix m = createMatrix(sizeX, sizeY);
        for(int i(0); i < sizeX * sizeY; i++) {
            (*m)[i] = function::tool::nextRandTool();
        }
        return m;
    }

    tensor createRandTensor(int sizeX, int sizeY, int sizeZ)
    {
        tensor t = createTensor(sizeX, sizeY, sizeZ);
        for(int i(0); i < sizeX * sizeY * sizeZ; i++) {
            (**t)[i] = function::tool::nextRandTool();
        }
        return t;
    }


    void destroyVector(vector ptr)
    {
        delete[] ptr;
    }

    void destroyMatrix(matrix ptr)
    {
        delete[] *ptr;
        delete[] ptr;
    }

    void destroyTensor(tensor ptr)
    {
        delete[] **ptr;
        delete[] *ptr;
        delete[] ptr;
    }

}