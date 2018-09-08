#ifndef LIK_ML_SEQUENTIALMODULE_H
#define LIK_ML_SEQUENTIALMODULE_H


#include <vector>
#include "../type/type.h"
#include "../layer/Layer.h"


namespace module
{
    class SequentialModule {
    public:
        SequentialModule();
        virtual ~SequentialModule();


        void training(type::tensor in[], type::tensor result[], int size, double perCentUseDataSet);
        type::tensor predict(type::tensor in);


        void add(layer::Layer* layer);
        void setInput(int sizeX, int sizeY = 1, int sizeZ = 1);
        void setStopParameter(int itMaxTraining, type::scalar acceptableError = 0.01);
        void setTrainingParameter(type::scalar learningRate, type::scalar adaptiveLearningRate);
        void link();

        layer::Layer* getLayer(int n) {return _lLayers[n];}
        type::tensor getOut() {return _lLayers.back()->getPtrOut();}

    private:
        std::vector<layer::Layer*> _lLayers;
        type::tensor _in;
        int _sizeXIn;
        int _sizeYIn;
        int _sizeZIn;

        int _itMaxTraining;
        type::scalar _acceptableError;

        type::scalar _learningRate;
        type::scalar _adaptiveLearningRate;
    };
}



#endif //LIK_ML_SEQUENTIALMODULE_H
