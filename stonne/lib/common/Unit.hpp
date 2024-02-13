#ifndef _UNIT_h_
#define _UNIT_h_

#include <fstream>
#include <iostream>
#include <string>
#include "Config.hpp"

class Unit {
 private:
  stonne_id_t id;    //Id of the component
  std::string name;  //Name of the component

 public:
  Unit(stonne_id_t id, std::string name) {
    this->id = id;
    this->name = name;
  }

  virtual ~Unit() {}

  virtual void printStats(std::ofstream& out, std::size_t indent) = 0;  //Print the stats of the component

  virtual void printEnergy(std::ofstream& out, std::size_t indent) = 0;  //Print the counters to get the consumption of the unit

  virtual void cycle() = 0;  //Execute a cycle in the component
};

#endif
