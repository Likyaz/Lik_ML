#include "Layer.h"

namespace layer
{

    Layer::Layer(int sizeXOut, int sizeYOut, int sizeZOut, int FLAG_ACTIVATION)
    {
        _sizeXOut = sizeXOut;
        _sizeYOut = sizeYOut;
        _sizeZOut = sizeZOut;

        _fActivation = function::math::activation::getActivation(FLAG_ACTIVATION);
        _fDerivative = function::math::activation::getDerivative(FLAG_ACTIVATION);
    }

    Layer::~Layer()
    {
        if(_out != nullptr) type::destroyTensor(_out);
        if(_delta != nullptr) type::destroyTensor(_delta);
        if(_derivative != nullptr) type::destroyTensor(_derivative);
        if(_ptrWeight != nullptr)
            for(int i(0); i < _numberOfWeight; i++)
                type::destroyTensor(_ptrWeight[i]);
        delete[] _ptrWeight;
    }

    type::tensor Layer::getPtrOut() {return _out;}
    type::tensor Layer::getPtrDelta() {return _delta;}
    type::tensor Layer::getPtrDerivative() {return _derivative;}
    type::tensor* Layer::getPtrWeight() {return _ptrWeight;}

    void Layer::setPtrIn(type::tensor ptr) {_in = ptr;}
    void Layer::setPtrDeltaIn(type::tensor ptr) {_deltaIn = ptr;}
    void Layer::setPtrDerivativeIn(type::tensor ptr) {_derivativeIn = ptr;}

    void Layer::copyDelta(type::tensor delta)
    {
        for(int x(0); x < _sizeXOut; x++)
            for(int y(0); y < _sizeYOut; y++)
                for(int z(0); z < _sizeZOut; z++)
                    _delta[z][y][x] = delta[z][y][x];
    }

    void Layer::createMemorySpace(int sizeXIn, int sizeYIn, int sizeZIn)
    {
        conf(sizeXIn, sizeYIn, sizeZIn);

        _out = type::createTensor(_sizeXOut, _sizeYOut, _sizeZOut);
        _delta = type::createTensor(_sizeXOut, _sizeYOut, _sizeZOut);
        _derivative = type::createTensor(_sizeXOut, _sizeYOut, _sizeZOut);

        _ptrWeight = new type::tensor[_numberOfWeight] {nullptr};
        for(int i(0); i < _numberOfWeight; i++)
            _ptrWeight[i] = type::createRandTensor(_sizeXWeight, _sizeYWeight, _sizeZWeight);
    }

}