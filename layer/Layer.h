#ifndef LIK_ML_LAYER_H
#define LIK_ML_LAYER_H

#include "../type/type.h"
#include "../function/math/activation.hpp"

namespace layer
{
    class Layer {
    public:
        Layer(int sizeXOut, int sizeYOut, int sizeZOut, int FLAG_ACTIVATION);
        virtual ~Layer();

        void createMemorySpace(int sizeXIn, int sizeYIn, int sizeZIn);

        type::tensor getPtrOut();
        type::tensor getPtrDelta();
        type::tensor getPtrDerivative();
        type::tensor* getPtrWeight();

        int getSizeOutX() {return _sizeXOut;}
        int getSizeOutY() {return _sizeYOut;}
        int getSizeOutZ() {return _sizeZOut;}



        void setPtrIn(type::tensor ptr);
        void setPtrDeltaIn(type::tensor ptr);
        void setPtrDerivativeIn(type::tensor ptr);

        void copyDelta(type::tensor delta);


        virtual void conf(int sizeXIn, int sizeYIn, int sizeZIn) = 0;
        virtual void predict() = 0;
        virtual void propagate() = 0;
        virtual void backPropagate() = 0;
        virtual void majWeight(type::scalar learningRate, type::scalar add = 0, type::scalar multi = 1) = 0;


    protected:
        type::tensor _in;
        type::tensor _deltaIn;
        type::tensor _derivativeIn;

        type::tensor _out;
        type::tensor _delta;
        type::tensor _derivative;
        type::tensor* _ptrWeight;


        int _sizeXOut;
        int _sizeYOut;
        int _sizeZOut;

        int _numberOfWeight;
        int _sizeXWeight;
        int _sizeYWeight;
        int _sizeZWeight;

        function::math::activation::ptrFunctionActivation _fActivation;
        function::math::activation::ptrFunctionActivation _fDerivative;
    };
}



#endif //LIK_ML_LAYER_H
