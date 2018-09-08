#include <iostream>

#include "example.h"

#include "../function/tool/random.hpp"
#include "../type/type.h"
#include "../module/SequentialModule.h"
#include "../layer/bases/FullyConnected.h"
#include "../layer/bases/Loss.h"


namespace example
{
    void learning_xor()
    {


        type::tensor in[4];
        in[0] = type::createTensor(2, 1, 1);
        in[1] = type::createTensor(2, 1, 1);
        in[2] = type::createTensor(2, 1, 1);
        in[3] = type::createTensor(2, 1, 1);

        type::tensor out[4];
        out[0] = type::createTensor(1, 1, 1);
        out[1] = type::createTensor(1, 1, 1);
        out[2] = type::createTensor(1, 1, 1);
        out[3] = type::createTensor(1, 1, 1);



        (**in[0])[0] = 0;
        (**in[0])[1] = 0;
        (***out[0]) = 0;

        (**in[1])[0] = 0;
        (**in[1])[1] = 1;
        (***out[1]) = 1;

        (**in[2])[0] = 1;
        (**in[2])[1] = 0;
        (***out[2]) = 1;

        (**in[3])[0] = 1;
        (**in[3])[1] = 1;
        (***out[3]) = 0;



        function::tool::initParamRandTool(-0.1, 0.1);
        function::tool::initRandTool(486446);
        module::SequentialModule moduleA;
        moduleA.setStopParameter(100, 0.001);
        moduleA.setTrainingParameter(0.2, 0.0000000001);
        moduleA.setInput(2);
        moduleA.add(new layer::FullyConnected(1, ACTIVATION_GAUSSIAN, false));
        moduleA.add(new layer::Loss(1, ACTIVATION_GAUSSIAN, LOSS_LOGLOSS, false));
        moduleA.link();
        moduleA.training(in, out, 4, 1);

        function::tool::initRandTool(486446);
        module::SequentialModule moduleB;
        moduleB.setStopParameter(1000, 0.001);
        moduleB.setTrainingParameter(0.2, 0.0000000001);
        moduleB.setInput(2);
        moduleB.add(new layer::FullyConnected(2, ACTIVATION_SIGMOIDE, true));
        moduleB.add(new layer::Loss(1, ACTIVATION_SIGMOIDE, LOSS_LOGLOSS, true));
        moduleB.link();
        moduleB.training(in, out, 4, 1);


        std::cout << "- Module A (ACTIVATION_GAUSSIAN): " << std::endl;
        for(int i(0); i < 4 ; i++)
        {
            std::cout << in[i][0][0][0] << " xor " << in[i][0][0][1] << " => " << std::endl;
            std::cout << "   Out : " << ***moduleA.predict(in[i]) << std::endl;
            std::cout << "   Layer  : " << (**moduleA.getLayer(0)->getPtrOut())[0] << std::endl;
        }


        std::cout << "\n\n- Module B (ACTIVATION_SIGMOIDE): " << std::endl;
        for(int i(0); i < 4 ; i++)
        {
            std::cout << in[i][0][0][0] << " xor " << in[i][0][0][1] << " => " << std::endl;
            std::cout << "   Out : " << ***moduleB.predict(in[i]) << std::endl;
            std::cout << "   Layer  : "
            << (**moduleB.getLayer(0)->getPtrOut())[0] << ", "
            << (**moduleB.getLayer(0)->getPtrOut())[1] << std::endl;
        }
    }
}