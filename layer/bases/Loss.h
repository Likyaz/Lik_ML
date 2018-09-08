#ifndef LIK_ML_LOSS_H
#define LIK_ML_LOSS_H

#include "FullyConnected.h"
#include "../../function/math/loss.hpp"

namespace layer
{
    class Loss : virtual public FullyConnected
    {
    public:
        Loss(int size, int FLAG_ACTIVATION, int FLAG_LOSS, bool bias);
        virtual ~Loss() = default;

        virtual void backPropagate();

    private:
        function::math::loss::ptrFunction _fLoss;
    };
}



#endif //LIK_ML_LOSS_H
