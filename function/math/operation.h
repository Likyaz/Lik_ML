#ifndef LIK_ML_OPERATION_H
#define LIK_ML_OPERATION_H

#include "../../type/type.h"

namespace operation
{
    // size[] = {X, Y, Z}
    void sumV(type::vector* res, type::vector a, type::vector b, int size[]);
    void sumM(type::matrix* res, type::matrix a, type::matrix b, int size[]);
    void sumT(type::tensor* res, type::tensor a, type::tensor b, int size[]);

    void subtractV(type::vector* res, type::vector a, type::vector b, int size[]);
    void subtractM(type::matrix* res, type::matrix a, type::matrix b, int size[]);
    void subtractT(type::tensor* res, type::tensor a, type::tensor b, int size[]);

    void scalarProductV(type::vector* res, type::vector a, type::vector b, int size[]);
    void scalarProductM(type::matrix* res, type::matrix a, type::matrix b, int size[]);
    void scalarProductT(type::tensor* res, type::tensor a, type::tensor b, int size[]);

}


#endif //LIK_ML_OPERATION_H
