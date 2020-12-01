
//Created by Francisco Munoz Martinez on 25/06/2019

// This class is used in the simulator in order to limit the size of the fifo.

#ifndef __Fifo_h__
#define __Fifo_h__

#include <queue>
#include "DataPackage.h"
#include "types.h"
#include "Stats.h"

class Fifo {
private:
    std::queue<DataPackage*> fifo;
    unsigned int capacity; //Capacity in number of bits
    unsigned int capacity_words; //Capacity in number of words allowed. i.e., capacity_words = capacity / size_word
    FifoStats fifoStats; //Tracking parameters
public:
    Fifo(unsigned int capacity);
    bool isEmpty();
    bool isFull();
    void push(DataPackage* data);
    DataPackage* pop();
    DataPackage* front();
    unsigned int size(); //Return the number of elements in the fifo
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
};
#endif
