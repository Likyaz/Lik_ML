#include <iostream>

#include "example.h"

#include "../function/tool/random.hpp"
#include "../type/type.h"
#include "../module/SequentialModule.h"
#include "../layer/conv/Convolution2D.h"
#include "../layer/bases/FullyConnected.h"
#include "../layer/bases/Loss.h"


namespace example
{
    void learning_XO()
    {
        type::tensor in[2];
        in[0] = type::createTensor(8, 8, 1);
        in[1] = type::createTensor(8, 8, 1);

        type::tensor out[2];
        out[0] = type::createTensor(2, 1, 1);
        out[1] = type::createTensor(2, 1, 1);

        for(int i(0); i < 8; i++){
            (*in[0])[i][i] = 1;
            (*in[0])[7 - i][i] = 1;
        }
        (**out[0])[0] = 1;
        (**out[0])[1] = 0;

        {
            int x = 0;
            int y = 3;
            int d = 3 - 1;
            while (y >= x)
            {
                (*in[1])[4 + x][4 + y] = 1;
                (*in[1])[4 + y][4 + x] = 1;
                (*in[1])[4 - x][4 + y] = 1;
                (*in[1])[4 - y][4 + x] = 1;
                (*in[1])[4 + x][4 - y] = 1;
                (*in[1])[4 + y][4 - x] = 1;
                (*in[1])[4 - x][4 - y] = 1;
                (*in[1])[4 - y][4 - x] = 1;

                if (d >= 2 * x)
                {
                    d -= 2 * x + 1;
                    x++;
                }
                else if (d < 2 * (3 - y))
                {
                    d += 2 * y - 1;
                    y--;
                }
                else
                {
                    d += 2 * (y - x - 1);
                    y--;
                    x++;
                }
            }
        }
        (**out[1])[0] = 0;
        (**out[1])[1] = 1;

        for(int y(0); y < 8; y++) {
            for(int x(0); x < 8; x++) {
                std::cout << (*in[0])[y][x] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
        std::cout << '\n';

        for(int y(0); y < 8; y++) {
            for(int x(0); x < 8; x++) {
                std::cout << (*in[1])[y][x] << ' ';
            }
            std::cout << '\n';
        }

        function::tool::initParamRandTool(-0.1, 0.1);
        function::tool::initRandTool(486446);
        module::SequentialModule module;
        module.setStopParameter(500, 0.001);
        module.setTrainingParameter(0.2, 0.0000000001);
        module.setInput(8, 8, 1);
        module.add(new layer::Convolution2D(2, 3, 3, ACTIVATION_SIGMOIDE, 0, 0, 1, 1));
        module.add(new layer::Convolution2D(2, 3, 3, ACTIVATION_SIGMOIDE, 0, 0, 1, 1));
        module.add(new layer::FullyConnected(8, ACTIVATION_SIGMOIDE, true));
        module.add(new layer::Loss(2, ACTIVATION_SIGMOIDE, LOSS_LOGLOSS, true));
        module.link();

        std::cout << 'a' << std::endl;
        module.training(in, out, 2, 1);
        std::cout << 'b' << std::endl;

        std::cout << "- Resulta: " << std::endl;
        module.predict(in[0]);
        std::cout << "  X : " << (**module.getOut())[0] << ", " << (**module.getOut())[1] << std::endl;
        module.predict(in[1]);
        std::cout << "  O : " << (**module.getOut())[0] << ", " << (**module.getOut())[1] << std::endl;



    }
}