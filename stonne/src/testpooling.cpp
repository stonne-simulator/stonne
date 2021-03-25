#include <iostream>
#include <cstdlib>
#include <ctime>

void max_pooling_layer(unsigned int R, unsigned int S, unsigned int C, unsigned int G,  unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, 
float* input, float * outputs) {

    unsigned int OX=(X - R + strides) / strides;
    unsigned int OY=(Y - S + strides) / strides;
    C=C/G;
    unsigned int output_size_n = G*C*OX*OY;
    unsigned int input_size_n = G*C*X*Y;
    unsigned int filter_size=R*S*C;
    unsigned int size_oy=OY*C*G;
    unsigned int size_y=Y*G*C;
    for(int n=0; n<N; n++) {
        for(int g=0; g<G; g++) {
            for(int k=0; k<C; k++) {
                for(int ox=0; ox<OX; ox++) {
                    for(int oy=0; oy<OY; oy++) {
                        outputs[n*output_size_n + ox*size_oy + oy*C*G + g*C + k]=-999.0;
                        for(int c=0; c<C;c++) {
                            for(int r=0;r<R;r++) {
                                for(int s=0;s<S;s++) {
                                    outputs[n*output_size_n + ox*size_oy + oy*C*G + g*C + k] = std::max(input[n*input_size_n+ ox*strides*size_y + oy*strides*C*G + r*size_y + s*C*G + g*C + c], outputs[n*output_size_n + ox*size_oy + oy*C*G + g*C + k]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void average_pooling_layer(unsigned int R, unsigned int S, unsigned int C, unsigned int G,  unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, 
float* input, float * outputs) {

    unsigned int OX=(X - R + strides) / strides;
    unsigned int OY=(Y - S + strides) / strides;
    C=C/G;
    unsigned int output_size_n = G*C*OX*OY;
    unsigned int input_size_n = G*C*X*Y;
    unsigned int filter_size=R*S*C;
    unsigned int size_oy=OY*C*G;
    unsigned int size_y=Y*G*C;
    for(int n=0; n<N; n++) {
        for(int g=0; g<G; g++) {
            for(int k=0; k<C; k++) {
                for(int ox=0; ox<OX; ox++) {
                    for(int oy=0; oy<OY; oy++) {
                        float acc = 0.0;
                        int counter = 0;
                        for(int c=0; c<C;c++) {
                            for(int r=0;r<R;r++) {
                                for(int s=0;s<S;s++) {
                                    acc += input[n*input_size_n+ ox*strides*size_y + oy*strides*C*G + r*size_y + s*C*G + g*C + c];
                                    ++counter;
                                }
                            }
                        }
                        outputs[n*output_size_n + ox*size_oy + oy*C*G + g*C + k] = acc / counter;
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int R = 2;
    int S = 2;
    int C = 1;
    int G = 1;
    int N = 1;
    int X = 4;
    int Y = 4;
    int strides = 2;

    int X_ = (X - R + strides) / strides;
    int Y_ = (Y - S + strides) / strides;

    float* input = new float[X*Y];
    float* output = new float[X_*Y_];
    float LO = 1.0;
    float HI = 100.0;

    srand(time(0));
    for (int i = 0; i < X * Y; ++i) 
    {
        // input[i] = LO + ((float)rand() / (float) (RAND_MAX/(HI-LO)));
        input[i] = i;
    }
    std::cout << "Input:" << std::endl;
    for (int i = 0; i < X; ++i)
    {
        std::cout << "[ ";
        for(int t = 0; t < Y; ++t)
        {
            std::cout << input[i * X + t] << " ";
        }
        std::cout << "]" << std::endl;
    }

    // max_pooling_layer(R, S, C, G, N, X, Y, strides, input, output);
    average_pooling_layer(R, S, C, G, N, X, Y, strides, input, output);

    std::cout << "Output:" << std::endl;
    for (int i = 0; i < X_; ++i)
    {
        std::cout << "[ ";
        for(int t = 0; t < Y_; ++t)
        {
            std::cout << output[i * X_ + t] << " ";
        }
        std::cout << "]" << std::endl;
    }

    return 0;
}