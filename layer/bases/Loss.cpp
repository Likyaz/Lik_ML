#include "Loss.h"

namespace layer
{

    Loss::Loss(int size, int FLAG_ACTIVATION, int FLAG_LOSS, bool bias) :
            FullyConnected(size, FLAG_ACTIVATION, bias)
    {
        _fLoss = function::math::loss::getLoss(FLAG_LOSS);
    }


    void Loss::backPropagate()
    {
        _fLoss((**_delta), (**_out), _sizeXOut);
        for(int i(0); i < _sizeXWeight; i++){
            average_backPropagate(i);
            (**_deltaIn)[i] *= (**_derivativeIn)[i];
        }
    }

}