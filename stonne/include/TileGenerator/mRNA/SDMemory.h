//
// Created by Zhongyuan Zhao on 1/17/19.
//

#ifndef SDMEMORY_H_
#define SDMEMORY_H_

namespace mRNA {

    class AddressAtribute {
    public:
        AddressAtribute();

    private:
        void setBank(unsigned int bank) { bank_num = bank; }

        unsigned int bank_num;
    };


    class SDMemory {

    };

}


#endif //SDMEMORY_H_
