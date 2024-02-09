#ifndef STONNE_MEMORY_H
#define STONNE_MEMORY_H

#include <vector>
#include <queue>
#include <DataPackage.h>
#include "types.h"

template <typename T>
class Memory {
public:
    Memory(unsigned int size) : memory(size, 0) {}

    void load(uint64_t addr, DataPackage * pck) {
        pck->set_data(memory[addr]);
        read_buffer.push(pck);
    }

    void store(uint64_t addr, DataPackage * pck) {
        memory[addr] = pck->get_data();
        write_buffer.push(pck);
    }

    typename std::vector<T>::iterator begin() {
        return memory.begin();
    }

    typename std::vector<T>::iterator end() {
        return memory.end();
    }

    unsigned int size() {
        return memory.size();
    }

    void cycle() {}

    unsigned int get_read_buffer_size() {
        return read_buffer.size();
    }

    unsigned int get_write_buffer_size() {
        return write_buffer.size();
    }

    DataPackage * get_read_buffer_front() {
        return read_buffer.front();
    }

    DataPackage * get_write_buffer_front() {
        return write_buffer.front();
    }

    void pop_read_buffer() {
        read_buffer.pop();
    }

    void pop_write_buffer() {
        write_buffer.pop();
    }

private:
    std::vector<T> memory;
    std::queue<DataPackage *> read_buffer;
    std::queue<DataPackage *> write_buffer;
};


#endif //STONNE_MEMORY_H
