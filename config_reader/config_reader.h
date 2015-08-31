#ifndef CONFIG_READER_H
#define CONFIG_READER_H
#include <string>
using namespace std;
class config_reader
{
private:
 struct _layers
    {
        //Filter Size
        int Sx;
        int Sy;
        int Sz;
        int Sd;
        //Filter & Biases
        float *biases;
        float *filters;
        string type;
        //Pad all side by
        float pad;

        string poolType;
        int poolSize;
    };
public:
    int nlayers;
    struct _layers *filterLayers;
    float* averageImage;
    bool read(string config_filepath);
    void release();
};
#endif // CONFIG_READER_H
