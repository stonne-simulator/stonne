#ifndef STONNE_UTILS_HPP
#define STONNE_UTILS_HPP

#include <cstdlib>
#include <random>
#include <vector>

template <typename T>
std::vector<T> genRandom(std::size_t sz, T min, T max) {
  std::uniform_real_distribution<T> dist(min, max);
  std::mt19937 gen(0);

  std::vector<T> vec(sz);
  std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });

  return vec;
}

template <typename T>
void prune(std::vector<T>& vec, T epsilon = 0) {
  for (std::size_t i = 0; i < vec.size(); i++) {
    if (std::abs(vec[i]) < epsilon) {
      vec[i] = 0;
    }
  }
}

template <typename T>
bool equals(const std::vector<T>& a, const std::vector<T>& b, T epsilon = 0) {
  if (a.size() != b.size()) {
    return false;
  }

  for (std::size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > epsilon) {
      std::cerr << "a[" << i << "] = " << a[i] << " != b[" << i << "] = " << b[i] << std::endl;
      return false;
    }
  }

  return true;
}

#endif  //STONNE_UTILS_HPP
