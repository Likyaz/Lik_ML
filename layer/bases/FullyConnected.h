#ifndef LIK_ML_FULLYCONNECTED_H
#define LIK_ML_FULLYCONNECTED_H


#include "../Layer.h"


#define TYPE_FC_AVERAGE                     01
#define TYPE_FC_MEAN_ABSOLUTE_DEVIATION     02
#define TYPE_FC_AVERAGE_COMPENSATION        03

namespace layer
{
    class FullyConnected : public Layer
    {
    public:
        FullyConnected() = default;
        FullyConnected(int size, int FLAG_ACTIVATION, bool bias);
        virtual ~FullyConnected();

        virtual void conf(int sizeXIn, int sizeYIn, int sizeZIn);
        virtual void predict();
        virtual void propagate();
        virtual void backPropagate();
        virtual void majWeight(type::scalar learningRate, type::scalar add = 0, type::scalar multi = 1);


        void average_propagate(int it);
        void meanAbsoluteDeviation_propagate(int it);
        void averageCompensation_propagate(int it);

        void average_backPropagate(int it);
        void meanAbsoluteDeviation_backPropagate(int it);
        void averageCompensation_backPropagate(int it);

    protected:
        type::vector _bias;
    };
}



#endif //LIK_ML_FULLYCONNECTED_H
