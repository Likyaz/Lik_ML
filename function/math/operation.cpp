#include "operation.h"


namespace operation
{
    void sumV(type::vector* res, type::vector a, type::vector b, int size[])
    {
        for(int i(0); i < size[0]; i++)
            (*res)[i] = a[i] + b[i];
    }

    void sumM(type::matrix* res, type::matrix a, type::matrix b, int size[])
    {
        for(int i(0); i < size[1]; i++)
            for(int j(0); j < size[0]; j++)
                (*res)[j][i] = a[j][i] + b[j][i];
    }

    void sumT(type::tensor* res, type::tensor a, type::tensor b, int size[])
    {

        for(int i(0); i < size[2]; i++)
            for(int j(0); j < size[1]; j++)
                for(int k(0); k < size[0]; k++)
                (*res)[k][j][i] = a[k][j][i] + b[k][j][i];
    }


    void subtractV(type::vector* res, type::vector a, type::vector b, int size[])
    {
        for(int i(0); i < size[0]; i++)
            (*res)[i] = a[i] - b[i];
    }

    void subtractM(type::matrix* res, type::matrix a, type::matrix b, int size[])
    {
        for(int i(0); i < size[1]; i++)
            for(int j(0); j < size[0]; j++)
                (*res)[j][i] = a[j][i] - b[j][i];
    }

    void subtractT(type::tensor* res, type::tensor a, type::tensor b, int size[])
    {
        for(int i(0); i < size[2]; i++)
            for(int j(0); j < size[1]; j++)
                for(int k(0); k < size[0]; k++)
                    (*res)[k][j][i] = a[k][j][i] - b[k][j][i];
    }



    void scalarProductV(type::vector* res, type::vector a, type::vector b, int size[])
    {
        for(int i(0); i < size[0]; i++)
            (*res)[i] = a[i] - b[i];
    }

    void scalarProductM(type::matrix* res, type::matrix a, type::matrix b, int size[])
    {
        for(int i(0); i < size[1]; i++)
            for(int j(0); j < size[0]; j++)
                (*res)[j][i] = a[j][i] * b[j][i];
    }

    void scalarProductT(type::tensor* res, type::tensor a, type::tensor b, int size[])
    {
        for(int i(0); i < size[2]; i++)
            for(int j(0); j < size[1]; j++)
                for(int k(0); k < size[0]; k++)
                    (*res)[k][j][i] = a[k][j][i] * b[k][j][i];
    }



}