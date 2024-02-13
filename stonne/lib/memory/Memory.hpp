#ifndef STONNE_MEMORY_HPP
#define STONNE_MEMORY_HPP

#include <queue>
#include <vector>
#include "../comm/DataPackage.hpp"
#include "../common/types.hpp"

template <typename T>
class Memory {
 public:
  Memory(std::size_t size) : memory(size, 0) {}

  void load(uint64_t addr, DataPackage* pck) {
    pck->set_data(memory[addr]);
    read_buffer.push(pck);
  }

  void store(uint64_t addr, DataPackage* pck) {
    memory[addr] = pck->get_data();
    write_buffer.push(pck);
  }

  typename std::vector<T>::iterator begin() { return memory.begin(); }

  typename std::vector<T>::iterator end() { return memory.end(); }

  std::size_t size() { return memory.size(); }

  void cycle() {}

  std::size_t get_read_buffer_size() { return read_buffer.size(); }

  std::size_t get_write_buffer_size() { return write_buffer.size(); }

  DataPackage* get_read_buffer_front() { return read_buffer.front(); }

  DataPackage* get_write_buffer_front() { return write_buffer.front(); }

  void pop_read_buffer() { read_buffer.pop(); }

  void pop_write_buffer() { write_buffer.pop(); }

 private:
  std::vector<T> memory;
  std::queue<DataPackage*> read_buffer;
  std::queue<DataPackage*> write_buffer;
};

#endif  //STONNE_MEMORY_HPP
