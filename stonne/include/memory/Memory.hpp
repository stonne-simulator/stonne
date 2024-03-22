#ifndef STONNE_MEMORY_HPP
#define STONNE_MEMORY_HPP

#include <queue>
#include <vector>
#include "comm/DataPackage.hpp"
#include "common/types.hpp"

/**
 * @brief Interface for a memory module.
 *
 * This class defines an interface for a memory module that can handle load and store requests,
 * track pending requests, and simulate memory cycles. Requests are queued and processed in FIFO order.
 *
 * @tparam T The type of data stored in the memory.
 */
template <typename T = float>
class Memory {
 public:
  virtual ~Memory() = default;

  /**
   * @brief Sends a load request to the memory
   * @param addr Address to load from
   * @param pck DataPackage to store the result
   */
  virtual void load(uint64_t addr, DataPackage* pck) = 0;

  /**
   * @brief Sends a store request to the memory
   * @param addr Address to store to
   * @param pck DataPackage containing the data to store
   */
  virtual void store(uint64_t addr, DataPackage* pck) = 0;

  /**
   * @brief Returns the number of pending load requests
   * @return Number of pending load requests
   */
  [[nodiscard]] virtual std::size_t pendingLoads() const = 0;

  /**
   * @brief Returns the number of pending store requests
   * @return Number of pending store requests
   */
  [[nodiscard]] virtual std::size_t pendingStores() const = 0;

  /**
   * @brief Returns the next completed load request, removing it from the queue
   * @return Next completed load request
   */
  [[nodiscard]] virtual DataPackage* nextLoad() = 0;

  /**
   * @brief Returns the next completed store request, removing it from the queue
   * @return Next completed store request
   */
  [[nodiscard]] virtual DataPackage* nextStore() = 0;

  /**
   * @brief Simulates one cycle of the memory
   */
  virtual void cycle() = 0;
};

#endif  //STONNE_MEMORY_HPP
