#include "FullyConnected.h"



namespace layer {

    FullyConnected::FullyConnected(int size, int FLAG_ACTIVATION, bool bias) :
            Layer(size, 1, 1, FLAG_ACTIVATION)
    {
        if (bias)
            _bias = type::createRandVector(size);
        else
            _bias = nullptr;
        _sizeYWeight = _sizeXOut;
        _sizeZWeight = 1;
        _numberOfWeight = 1;
    }

    FullyConnected::~FullyConnected() {
        type::destroyVector(_bias);
    }

    void FullyConnected::conf(int sizeXIn, int sizeYIn, int sizeZIn) {
        _sizeXWeight = sizeXIn;

    }

    void FullyConnected::predict() {
        for (int i(0); i < _sizeYWeight; i++) {
            (**_out)[i] = _bias == nullptr ? 0 : _bias[i];
            average_propagate(i);
            (**_out)[i] = _fActivation((**_out)[i]);
        }
    }

    void FullyConnected::propagate() {
        for (int i(0); i < _sizeYWeight; i++) {
            (**_out)[i] = _bias == nullptr ? 0 : _bias[i];
            average_propagate(i);
            (**_derivative)[i] = _fDerivative((**_out)[i]);
            (**_out)[i] = _fActivation((**_out)[i]);
        }
    }

    void FullyConnected::backPropagate() {
        for (int i(0); i < _sizeXWeight; i++) {
            (**_deltaIn)[i] = 0;
            average_backPropagate(i);
            (**_deltaIn)[i] *= (**_derivativeIn)[i];
        }

    }

    void FullyConnected::majWeight(type::scalar learningRate, type::scalar add, type::scalar multi) {
        for (int j = 0; j < _sizeYWeight; j++) {
            if (_bias != nullptr)
                _bias[j] += (learningRate * (**_delta)[j] * multi) + add;
            for (int i = 0; i < _sizeXWeight; i++) {
                (**_ptrWeight)[j][i] += (learningRate * (**_delta)[j] * (**_in)[i] * multi) + add;
            }
        }
    }


    void FullyConnected::average_propagate(int it) {
        for (int i(0); i < _sizeXWeight; i++) {
            (**_out)[it] += (**_in)[i] * (**_ptrWeight)[it][i];
        }
    }


    void FullyConnected::average_backPropagate(int it) {
        for (int i(0); i < _sizeYWeight; i++) {
            (**_deltaIn)[it] += (**_delta)[i] * (**_ptrWeight)[i][it];
        }
    }
}