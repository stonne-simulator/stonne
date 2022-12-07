//
// Created by Zhongyuan Zhao on 9/20/18.
//

#ifndef DEFINE_H_
#define DEFINE_H_

#define PUSH_LENGTH 12
#define PULL_LENGTH 12
#define NORMALIZE_MAC 1000

namespace mRNA {

    enum OptGoal {
        none = 0, // *** Added for STONNE compatibility ***
        performance = 1,
        energy = 2,
        energy_efficiency = 3,
        all = 4,
    };

    enum Opcode {
        Add = 0,
        Compare = 1,
        Add_fowd = 2,
        Multiply = 3,
        Mul_fowd = 4,
        Mpush = 5,
        Cpush = 6,
        Pull = 7,
        Distribute = 8,
    };

    enum ConfigType {
        conv = 0,
        ps = 1,
        fc = 2,
        lstm = 3,
        hmdpadd = 4,
        singlehmdp = 5,
    };
}

#endif //DEFINE_H_
