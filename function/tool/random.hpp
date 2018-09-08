#ifndef LIK_IA_RANDOM_HPP
#define LIK_IA_RANDOM_HPP

namespace function::tool
{
    static unsigned long long __registerRand = 0x8AC05EF799500CB7l;
    static double __min = -1;
    static double __max = 1;

    static void initRandTool(unsigned long long seed);
    static void initParamRandTool(double min, double max);
    static unsigned long long nextRandTool_Haynes();
    static double nextRandTool();

    void initRandTool(unsigned long long seed) {
        __registerRand = seed;
    }

    void initParamRandTool(double min, double max) {
        __min = min;
        __max = max;
    }

    unsigned long long nextRandTool_Haynes() {
        __registerRand = ((6364136223846793005l * __registerRand) & 0xFFFFFFFFFFFFFFFFl);
        return __registerRand >> 16;
    }

    double nextRandTool() {
        return (((double) (nextRandTool_Haynes() & 0xFFFFFF) / 0xFFFFFF) * (__max - __min)) + __min;
    }

}

#endif //LIK_IA_RANDOM_HPP
