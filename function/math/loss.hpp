#ifndef LIK_ML_LOSS_HPP
#define LIK_ML_LOSS_HPP

#include <cmath>
#include <iostream>
#include "../../type/type.h"
#define LOSS_SQUAREDLOSS             01
#define LOSS_LOGLOSS                 02
#define LOSS_CROSSENTROPY            03
#define LOSS_NEGATIVELOGLIKELIHOOD   04

#define MIN 0.0000000000001
#define MAX 1 - MIN

namespace function::math::loss
{
    typedef void (*ptrFunction)(type::vector, type::vector, int);

    static ptrFunction getLoss(int FLAG_LOSS);

    inline void SquaredLoss(type::vector delta, type::vector in, int size)
    {
        for(int i(0); i <= size; i++)
            delta[i] = (delta[i] - in[i]) * powl(delta[i] - in[i], 2);
    }

    inline void NegativeLogLikelihood(type::vector delta, type::vector in, int size)
    {
        // a dev

    }

    inline void LogLoss(type::vector delta, type::vector in, int size)
    {
        type::scalar t;
        for(int i(0); i <= size; i++)
        {
            t = in[i] < MIN ? MIN : in[i];
            t = t > MAX ? MAX : t;
            delta[i] = ( delta[i] == 0 ? logl(1 - t) : -logl(t) );
        }
    }

    inline void CrossEntropy(type::vector delta, type::vector in, int size)
    {
        type::scalar t;
        for(int i(0); i <= size; i++)
        {
            t = in[i] < MIN ? MIN : in[i];
            t = t > MAX ? MAX : t;
            delta[i] = (1 -delta[i]) * logl(1 - t) - delta[i] * logl(t);
        }

    }

    ptrFunction getLoss(int FLAG_LOSS)
    {
        switch(FLAG_LOSS)
        {
            case LOSS_SQUAREDLOSS:
                return SquaredLoss;
            case LOSS_LOGLOSS:
                return LogLoss;
            case LOSS_CROSSENTROPY:
                return CrossEntropy;
            case LOSS_NEGATIVELOGLIKELIHOOD:
                return NegativeLogLikelihood;
            default:
                std::cout << "FLAG_LOSS not supported" << std::endl;
        }
        return nullptr;
    }
}

#endif //LIK_ML_LOSS_HPP
