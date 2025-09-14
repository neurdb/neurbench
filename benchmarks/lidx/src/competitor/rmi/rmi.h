#include "books_200M_uint32_0.h"
#include "../indexInterface.h"
// #include <vector>
// #include <algorithm>
// #include <cstdlib>
// #include <iostream>

template<class KEY_TYPE, class PAYLOAD_TYPE>
class rmiInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  void init(Param* param = nullptr) override {
    const std::string rmi_path =
        (std::getenv("TLI_RMI_PATH") == NULL ? "/workspace/GRE_SALI/rmi_data"
                                             : std::getenv("TLI_RMI_PATH"));
    std::cout<<rmi_path.c_str()<<std::endl;
    if (!books_200M_uint32_0::load(rmi_path.c_str())) {
      std::cerr << "Could not load RMI data from rmi_data/\n";
      std::exit(1);
    }
  }

  void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE>* key_value, size_t num, Param* param = nullptr) override {
    data_.assign(key_value, key_value + num);
  }

  bool get(KEY_TYPE key, PAYLOAD_TYPE& val, Param* param = nullptr) override {
    size_t error;
    uint64_t guess = books_200M_uint32_0::lookup(key, &error);
    uint64_t start = (guess < error ? 0 : guess - error);
    uint64_t stop = std::min<size_t>(guess + error, data_.size());

    auto it = std::lower_bound(data_.begin() + start, data_.begin() + stop, key,
      [](const std::pair<KEY_TYPE, PAYLOAD_TYPE>& a, const KEY_TYPE& b) {
        return a.first < b;
      });

    if (it != data_.end() && it->first == key) {
      val = it->second;
      return true;
    }
    return false;
  }

  bool put(KEY_TYPE, PAYLOAD_TYPE, Param* = nullptr) override { return false; }
  bool update(KEY_TYPE, PAYLOAD_TYPE, Param* = nullptr) override { return false; }
  bool remove(KEY_TYPE, Param* = nullptr) override { return false; }

  size_t scan(KEY_TYPE key_low_bound, size_t key_num,
              std::pair<KEY_TYPE, PAYLOAD_TYPE>* result,
              Param* param = nullptr) override {
    size_t error;
    uint64_t guess = books_200M_uint32_0::lookup(key_low_bound, &error);
    uint64_t start = (guess < error ? 0 : guess - error);
    uint64_t stop = std::min<size_t>(guess + error, data_.size());

    auto it = std::lower_bound(data_.begin() + start, data_.begin() + stop, key_low_bound,
      [](const std::pair<KEY_TYPE, PAYLOAD_TYPE>& a, const KEY_TYPE& b) {
        return a.first < b;
      });

    size_t count = 0;
    while (it != data_.end() && count < key_num) {
      result[count++] = *it;
      ++it;
    }
    return count;
  }
  
  long long memory_consumption() override {
    return sizeof(std::pair<KEY_TYPE, PAYLOAD_TYPE>) * data_.size();
  }

private:
  std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> data_;
};