#ifndef LIK_ML_CONVOLUTION2D_H
#define LIK_ML_CONVOLUTION2D_H


#include "../Layer.h"

namespace layer
{
    class Convolution2D : public Layer
    {
    public:
        Convolution2D(int size, int sizeKernelX, int sizeKernelY, int FLAG_ACTIVATION, int zeroPaddingX, int zeroPaddingY, int stepX, int stepY);
        virtual ~Convolution2D() = default;


        virtual void conf(int sizeXIn, int sizeYIn, int sizeZIn);
        virtual void predict();
        virtual void propagate();
        virtual void backPropagate();
        virtual void majWeight(type::scalar learningRate, type::scalar add = 0, type::scalar multi = 1);

    private:


        int _sizeXIn;
        int _sizeYIn;

        int _zeroPaddingX;
        int _zeroPaddingY;

        int _stepX;
        int _stepY;
    };
}



#endif //LIK_ML_CONVOLUTION2D_H
