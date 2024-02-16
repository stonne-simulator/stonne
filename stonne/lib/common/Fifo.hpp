// This class is used in the simulator in order to limit the size of the fifo.

#ifndef __Fifo_h__
#define __Fifo_h__

#include <queue>
#include "comm/DataPackage.hpp"
#include "common/Stats.hpp"
#include "common/types.hpp"

class Fifo {
 private:
  std::queue<DataPackage*> fifo;
  std::size_t capacity;        //Capacity in number of bits
  std::size_t capacity_words;  //Capacity in number of words allowed. i.e., capacity_words = capacity / size_word
  FifoStats fifoStats;         //Tracking parameters
 public:
  Fifo(std::size_t capacity);
  ~Fifo();
  bool isEmpty();
  bool isFull();
  void push(DataPackage* data);
  DataPackage* pop();
  DataPackage* front();
  std::size_t size();  //Return the number of elements in the fifo
  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);
};
#endif
