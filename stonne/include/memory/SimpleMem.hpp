#ifndef STONNE_SIMPLEMEM_HPP
#define STONNE_SIMPLEMEM_HPP

#include <queue>
#include <vector>
#include "comm/DataPackage.hpp"
#include "common/types.hpp"
#include "memory/Memory.hpp"

template <typename T>
class SimpleMem : public Memory<T> {
 public:
  SimpleMem() = delete;

  explicit SimpleMem(std::size_t size) : memory(size, 0) {}

  constexpr void load(uint64_t addr, DataPackage* pck) noexcept override {
    assert(addr < memory.size());
    pck->set_data(memory[addr]);
    readBuffer.push(pck);
  }

  constexpr void store(uint64_t addr, DataPackage* pck) noexcept override {
    assert(addr < memory.size());
    memory[addr] = pck->get_data();
    writeBuffer.push(pck);
  }

  [[nodiscard]] constexpr std::size_t pendingLoads() const noexcept override { return readBuffer.size(); }

  [[nodiscard]] constexpr std::size_t pendingStores() const noexcept override { return writeBuffer.size(); }

  [[nodiscard]] DataPackage* nextLoad() noexcept override {
    if (readBuffer.empty())
      return nullptr;

    auto pck = readBuffer.front();
    readBuffer.pop();
    return pck;
  }

  [[nodiscard]] DataPackage* nextStore() noexcept override {
    if (writeBuffer.empty())
      return nullptr;

    auto pck = writeBuffer.front();
    writeBuffer.pop();
    return pck;
  }

  template <typename Iter>
  constexpr void fill(std::size_t start, Iter begin, Iter end) noexcept {
    assert(start + std::distance(begin, end) <= memory.size());
    std::copy(begin, end, memory.begin() + start);
  }

  constexpr void cycle() override {}

 private:
  std::vector<T> memory;

  // Simulated memory buffer to attend requests 1 cycle after they are issued
  std::queue<DataPackage*> readBuffer;
  std::queue<DataPackage*> writeBuffer;
};

#endif  //STONNE_SIMPLEMEM_HPP
