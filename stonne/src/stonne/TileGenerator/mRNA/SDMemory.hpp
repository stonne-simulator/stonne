#ifndef SDMEMORY_H_
#define SDMEMORY_H_

#include <cstddef>

namespace mRNA {

class AddressAtribute {
 public:
  AddressAtribute();

 private:
  void setBank(std::size_t bank) { bank_num = bank; }

  std::size_t bank_num;
};

class SDMemory {};

}  // namespace mRNA

#endif  //SDMEMORY_H_
