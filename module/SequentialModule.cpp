#include "SequentialModule.h"


namespace module
{

    SequentialModule::SequentialModule()
    {
        _in = nullptr;
        _sizeXIn = 0;
        _sizeYIn = 0;
        _sizeZIn = 0;

        _itMaxTraining = 1000;
        _acceptableError = 0.001;

        _learningRate = 0.1;
        _adaptiveLearningRate = 0.0000000001;
    }

    SequentialModule::~SequentialModule()
    {
        if(_in != nullptr) type::destroyTensor(_in);

        for(int i(0); i < _lLayers.size(); i++)
            delete _lLayers[i];
    }

    void SequentialModule::add(layer::Layer* layer)
    {
        _lLayers.push_back(layer);
    }

    void SequentialModule::setInput(int sizeX, int sizeY, int sizeZ)
    {
        _sizeXIn = sizeX;
        _sizeYIn = sizeY;
        _sizeZIn = sizeZ;

        _in = type::createTensor(_sizeXIn, _sizeYIn, _sizeZIn);
    }

    void SequentialModule::setStopParameter(int itMaxTraining, type::scalar acceptableError)
    {
        _itMaxTraining = itMaxTraining;
        _acceptableError = acceptableError;
    }

    void SequentialModule::setTrainingParameter(type::scalar learningRate, type::scalar adaptiveLearningRate)
    {
        _learningRate = learningRate;
        _adaptiveLearningRate = adaptiveLearningRate;
    }

    void SequentialModule::link()
    {

        _lLayers[0]->createMemorySpace(_sizeXIn, _sizeYIn, _sizeZIn);
        _lLayers[0]->setPtrIn(_in);
        for(int i(1); i < _lLayers.size(); i++)
        {
            _lLayers[i]->createMemorySpace(_lLayers[i-1]->getSizeOutX(), _lLayers[i-1]->getSizeOutY(), _lLayers[i-1]->getSizeOutZ());
            _lLayers[i]->setPtrIn(_lLayers[i-1]->getPtrOut());
            _lLayers[i]->setPtrDeltaIn(_lLayers[i-1]->getPtrDelta());
            _lLayers[i]->setPtrDerivativeIn(_lLayers[i-1]->getPtrDerivative());
        }
    }

    type::tensor SequentialModule::predict(type::tensor in)
    {
        for(int x(0); x < _sizeXIn; x++)
            for(int y(0); y < _sizeYIn; y++)
                for(int z(0); z < _sizeZIn; z++)
                    _in[z][y][x] = in[z][y][x];

        for(int itLayer(0); itLayer < _lLayers.size(); itLayer++)
            _lLayers[itLayer]->predict();
        return _lLayers.back()->getPtrOut();
    }

    void SequentialModule::training(type::tensor in[], type::tensor result[], int size, double perCentUseDataSet)
    {

        for(int it(0); it < _itMaxTraining; it++)
        {
            const type::scalar actualLearningRate = _learningRate / 1 + (it * _adaptiveLearningRate);
            for(int itEx(0); itEx < size * perCentUseDataSet; itEx++)
            {

                for(int x(0); x < _sizeXIn; x++)
                    for(int y(0); y < _sizeYIn; y++)
                        for(int z(0); z < _sizeZIn; z++)
                            _in[z][y][x] = in[itEx][z][y][x];

                for(int itLayer(0); itLayer < _lLayers.size(); itLayer++)
                    _lLayers[itLayer]->propagate();

                _lLayers.back()->copyDelta(result[itEx]);

                for(int itLayer(_lLayers.size() - 1); itLayer > 0; itLayer--)
                    _lLayers[itLayer]->backPropagate();

                for(int itLayer(0); itLayer < _lLayers.size(); itLayer++)
                    _lLayers[itLayer]->majWeight(actualLearningRate);
            }
        }
    }

}