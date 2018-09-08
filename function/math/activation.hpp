#ifndef LIK_ML_ACTIVATION_HPP
#define LIK_ML_ACTIVATION_HPP

#include <cmath>
#include <iostream>

#include "../../type/type.h"

#define ACTIVATION_LINEAR      01
#define ACTIVATION_SIGMOIDE    02
#define ACTIVATION_SOFT_SIGN   03
#define ACTIVATION_RELU        04
#define ACTIVATION_GAUSSIAN    05
#define ACTIVATION_SILU        06

namespace function::math::activation
{
    typedef type::scalar (*ptrFunctionActivation)(type::scalar);
    static ptrFunctionActivation getActivation(int FLAG_ACTIVATION);
    static ptrFunctionActivation getDerivative(int FLAG_ACTIVATION);

    namespace function
    {
        inline type::scalar linear(type::scalar x) {return x;}
        inline type::scalar sigmoid(type::scalar x) {return 1 / (1 + expl(-x));}
        inline type::scalar softSign(type::scalar x) {return x / (1 + fabsl(x));}
        inline type::scalar ReLU(type::scalar x) {return x >= 0 ? x : 0;}
        inline type::scalar gaussian(type::scalar x) {return expl(-(x*x));}
        inline type::scalar SiLU(type::scalar x) {return x * function::sigmoid(x);}
    }

    namespace derivative
    {
        inline type::scalar linear(type::scalar x) {return 1;}
        inline type::scalar sigmoid(type::scalar x) {return function::sigmoid(x) * (1 - function::sigmoid(x));}
        inline type::scalar softSign(type::scalar x) {return 1 / powl((1 + fabsl(x)),2);}
        inline type::scalar ReLU(type::scalar x) {return x >= 0 ? 1 : 0;}
        inline type::scalar gaussian(type::scalar x) {return -2 * x * expl(-(x*x));}
        inline type::scalar SiLU(type::scalar x) {return function::SiLU(x) + (function::sigmoid(x) * (1 - function::SiLU(x)));}
    }


    ptrFunctionActivation getActivation(int FLAG_ACTIVATION)
    {
        switch(FLAG_ACTIVATION)
        {
            case ACTIVATION_LINEAR:
                return function::linear;
            case ACTIVATION_SIGMOIDE:
                return function::sigmoid;
            case ACTIVATION_SOFT_SIGN:
                return function::softSign;
            case ACTIVATION_RELU:
                return function::ReLU;
            case ACTIVATION_GAUSSIAN:
                return function::gaussian;
            case ACTIVATION_SILU:
                return function::SiLU;
            default:
                std::cout << "FLAG_ACTIVATION not supported" << std::endl;
        }
        return nullptr;
    }

    ptrFunctionActivation getDerivative(int FLAG_ACTIVATION)
    {
        switch(FLAG_ACTIVATION)
        {
            case ACTIVATION_LINEAR:
                return derivative::linear;
            case ACTIVATION_SIGMOIDE:
                return derivative::sigmoid;
            case ACTIVATION_SOFT_SIGN:
                return derivative::softSign;
            case ACTIVATION_RELU:
                return derivative::ReLU;
            case ACTIVATION_GAUSSIAN:
                return derivative::gaussian;
            case ACTIVATION_SILU:
                return derivative::SiLU;
            default:
                std::cout << "FLAG_ACTIVATION not supported" << std::endl;
        }
        return nullptr;
    }
}

#endif //LIK_ML_ACTIVATION_HPP
