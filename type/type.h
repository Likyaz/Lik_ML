#ifndef LIK_ML_TYPE_H
#define LIK_ML_TYPE_H

#include "../function/tool/random.hpp"


namespace type
{

    typedef long double scalar;     // :v
    typedef long double* vector;    // [x]:v
    typedef long double** matrix;   // [y][x]:v
    typedef long double*** tensor;  // [z][y][x]:v

    matrix convertToMatrix(tensor val);

    vector convertToVector(tensor val);
    vector convertToVector(matrix val);

    vector createVector(int size);
    matrix createMatrix(int sizeX, int sizeY);
    tensor createTensor(int sizeX, int sizeY, int sizeZ);

    vector createRandVector(int size);
    matrix createRandMatrix(int sizeX, int sizeY);
    tensor createRandTensor(int sizeX, int sizeY, int sizeZ);


    void destroyVector(vector ptr);
    void destroyMatrix(matrix ptr);
    void destroyTensor(tensor ptr);


}


#endif //LIK_ML_TYPE_H
