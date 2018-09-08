#include "Convolution2D.h"

namespace layer
{

    Convolution2D::Convolution2D(int size, int sizeKernelX, int sizeKernelY, int FLAG_ACTIVATION, int zeroPaddingX, int zeroPaddingY, int stepX, int stepY) :
            Layer(0, 0, size, FLAG_ACTIVATION)
    {
        _sizeXWeight = sizeKernelX;
        _sizeYWeight = sizeKernelY;
        _numberOfWeight = size;

        _stepX = stepX;
        _stepY = stepY;

        _zeroPaddingX = zeroPaddingX;
        _zeroPaddingY = zeroPaddingY;
    }

    void Convolution2D::conf(int sizeXIn, int sizeYIn, int sizeZIn)
    {
        _sizeXIn = sizeXIn;
        _sizeYIn = sizeYIn;

        _sizeZWeight = sizeZIn;

        _sizeXOut = (sizeXIn + (_zeroPaddingX<<1) - (_sizeXWeight&1 ? (_sizeXWeight>>1)+1 : _sizeXWeight>>1)) / _stepX;
        _sizeYOut = (sizeYIn + (_zeroPaddingY<<1) - (_sizeYWeight&1 ? (_sizeYWeight>>1)+1 : _sizeYWeight>>1)) / _stepY;
    }

    void Convolution2D::predict()
    {
        int posXIn = 0;
        int posYIn = 0;


        for (int posZOut(0); posZOut < _sizeZOut; posZOut++) {
            for (int posYOut(0); posYOut < _sizeYOut; posYOut++) {
                for (int posXOut(0); posXOut < _sizeXOut; posXOut++) {
                    _out[posZOut][posYOut][posXOut] = 0;

                    for (int posZKernel(0); posZKernel < _sizeZWeight; posZKernel++) {
                        posYIn = (posYOut * _stepY) - _zeroPaddingY;
                        for (int posYKernel(0); posYKernel < _sizeYWeight; posYKernel++) {
                            if((posYIn >= 0) & (posYIn < _sizeYIn)) {
                                posXIn = (posXOut * _stepX) - _zeroPaddingX;
                                for (int posXKernel(0); posXKernel < _sizeXWeight; posXKernel++) {
                                    if((posXIn >= 0) & (posXIn < _sizeYIn)) {
                                        _out[posZOut][posYOut][posXOut] +=
                                                _ptrWeight[posZOut][posZKernel][posYKernel][posXKernel] *
                                                _in[posZKernel][posYIn][posXIn];
                                    }
                                    posXIn++;
                                }
                            }
                            posYIn++;
                        }
                    }
                    _out[posZOut][posYOut][posXOut] = _fActivation(_out[posZOut][posYOut][posXOut]);

                }
            }
        }
    }

    void Convolution2D::propagate()
    {
        int posXIn = 0;
        int posYIn = 0;


        for (int posZOut(0); posZOut < _sizeZOut; posZOut++) {
            for (int posYOut(0); posYOut < _sizeYOut; posYOut++) {
                for (int posXOut(0); posXOut < _sizeXOut; posXOut++) {
                    _out[posZOut][posYOut][posXOut] = 0;

                    for (int posZKernel(0); posZKernel < _sizeZWeight; posZKernel++) {
                        posYIn = (posYOut * _stepY) - _zeroPaddingY;
                        for (int posYKernel(0); posYKernel < _sizeYWeight; posYKernel++) {
                            if((posYIn >= 0) & (posYIn < _sizeYIn)) {
                                posXIn = (posXOut * _stepX) - _zeroPaddingX;
                                for (int posXKernel(0); posXKernel < _sizeXWeight; posXKernel++) {
                                    if((posXIn >= 0) & (posXIn < _sizeXIn)) {
                                                  _out[posZOut][posYOut][posXOut] +=
                                                _ptrWeight[posZOut][posZKernel][posYKernel][posXKernel] *
                                                _in[posZKernel][posYIn][posXIn];
                                    }
                                    posXIn++;
                                }
                            }
                            posYIn++;
                        }
                    }
                    _derivative[posZOut][posYOut][posXOut] = _fDerivative(_out[posZOut][posYOut][posXOut]);
                    _out[posZOut][posYOut][posXOut] = _fActivation(_out[posZOut][posYOut][posXOut]);

                }
            }
        }
    }

    void Convolution2D::backPropagate()
    {
        int posXIn = 0;
        int posYIn = 0;

        for(int i(0); i < _sizeZWeight * _sizeYIn * _sizeYIn; i++)
            (**_deltaIn)[i] = 0;

        for (int posZOut(0); posZOut < _sizeZOut; posZOut++) {
            for (int posYOut(0); posYOut < _sizeYOut; posYOut++) {
                for (int posXOut(0); posXOut < _sizeXOut; posXOut++) {

                    for (int posZKernel(0); posZKernel < _sizeZWeight; posZKernel++) {
                        posYIn = (posYOut * _stepY) - _zeroPaddingY;
                        for (int posYKernel(0); posYKernel < _sizeYWeight; posYKernel++) {
                            if((posYIn >= 0) & (posYIn < _sizeYIn)) {
                                posXIn = (posXOut * _stepX) - _zeroPaddingX;
                                for (int posXKernel(0); posXKernel < _sizeXWeight; posXKernel++) {
                                    if((posXIn >= 0) & (posXIn < _sizeYIn)) {
                                        _deltaIn[posZKernel][posYIn][posXIn] +=
                                                _delta[posZOut][posYOut][posXOut] *
                                                _ptrWeight[posZOut][posZKernel][posYKernel][posXKernel];
                                    }
                                    posXIn++;
                                }
                            }
                            posYIn++;
                        }
                    }
                }
            }
        }


        for(int i(0); i < _sizeZWeight * _sizeYIn * _sizeYIn; i++)
            (**_deltaIn)[i] = _fDerivative((**_deltaIn)[i]);
    }

    void Convolution2D::majWeight(type::scalar learningRate, type::scalar add, type::scalar multi)
    {

        int posXIn = 0;
        int posYIn = 0;


        for (int posZOut(0); posZOut < _sizeZOut; posZOut++) {
            for (int posYOut(0); posYOut < _sizeYOut; posYOut++) {
                for (int posXOut(0); posXOut < _sizeXOut; posXOut++) {

                    for (int posZKernel(0); posZKernel < _sizeZWeight; posZKernel++) {
                        posYIn = (posYOut * _stepY) - _zeroPaddingY;
                        for (int posYKernel(0); posYKernel < _sizeYWeight; posYKernel++) {
                            if((posYIn >= 0) & (posYIn < _sizeYIn)) {
                                posXIn = (posXOut * _stepX) - _zeroPaddingX;
                                for (int posXKernel(0); posXKernel < _sizeXWeight; posXKernel++) {
                                    if((posXIn >= 0) & (posXIn < _sizeYIn)) {
                                        _ptrWeight[posZOut][posZKernel][posYKernel][posXKernel] +=
                                                (learningRate * _delta[posZOut][posYOut][posXOut] *
                                                 _in[posZKernel][posYIn][posXIn] * multi) + add;
                                    }
                                    posXIn++;
                                }
                            }
                            posYIn++;
                        }
                    }
                }
            }
        }

    }
}