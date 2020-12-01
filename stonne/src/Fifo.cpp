#include "Fifo.h"
#include <assert.h>
#include <iostream>
#include "utility.h"
Fifo::Fifo(unsigned int capacity) {
    this->capacity = capacity;
    this->capacity_words = capacity / sizeof(data_t); //Data size
}

bool Fifo::isFull() {
    return  this->fifo.size() >= this->capacity_words;  // > is forbidden
}

bool Fifo::isEmpty() {
    return this->fifo.size()==0;
}

void Fifo::push(DataPackage* data) {
//    assert(!isFull());  //The fifo must not be full
    fifo.push(data); //Inserting at the end of the queue
    if(this->size() > this->fifoStats.max_occupancy) {
        this->fifoStats.max_occupancy = this->size();
    }
    this->fifoStats.n_pushes+=1; // To track information
    
}

DataPackage* Fifo::pop() {
    assert(!isEmpty());
    this->fifoStats.n_pops+=1; //To track information
    DataPackage* pck = fifo.front(); //Accessing the first element of the queue
    fifo.pop(); //Extracting the first element
    return pck; 
}

DataPackage* Fifo::front() {
    assert(!isEmpty());
    DataPackage* pck = fifo.front();
    this->fifoStats.n_fronts+=1; //To track information
    return pck;
}

unsigned int Fifo::size() {
    return fifo.size();
}

void Fifo::printStats(std::ofstream& out, unsigned int indent) {
    this->fifoStats.print(out, indent);
}

void Fifo::printEnergy(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "FIFO PUSH=" << fifoStats.n_pushes; //Same line
    out << ind(indent) << " POP=" << fifoStats.n_pops;  //Same line
    out << ind(indent) << " FRONT=" << fifoStats.n_fronts << std::endl; //New line 
}


