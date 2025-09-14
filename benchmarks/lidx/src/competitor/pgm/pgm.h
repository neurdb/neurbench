#include"./src/include/pgm/pgm_index.hpp"
#include"../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class pgmInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  void init(Param *param = nullptr) {}

  void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr);

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool remove(KEY_TYPE key, Param *param = nullptr);

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr);

  long long memory_consumption() { return sizeof(std::pair<KEY_TYPE, PAYLOAD_TYPE>) * data_.size(); }

private:
  pgm::PGMIndex<KEY_TYPE> *index;
  std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> data_;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void pgmInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                                                     Param *param) {
  std::vector<KEY_TYPE> keys;
  keys.reserve(num);
  for (size_t i = 0; i < num; ++i) {
      keys.push_back(key_value[i].first);
  }
  data_.assign(key_value, key_value + num);
  index = new pgm::PGMIndex<KEY_TYPE>(keys);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool pgmInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
  auto approx_range = index->search(key);
  auto pos = approx_range.pos;
  auto lo = approx_range.lo;
  auto hi = approx_range.hi;

  auto it = std::lower_bound(data_.begin() + lo, data_.begin() + hi, key,
  [](const std::pair<KEY_TYPE, PAYLOAD_TYPE>& a, const KEY_TYPE& b) {
    return a.first < b;
  });

  if (it != data_.end() && it->first == key) {
    val = it->second;
    return true;
  }
  return false; 
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool pgmInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  return true;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool pgmInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  return true;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool pgmInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  return true;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t pgmInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                  std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                                                  Param *param) {
  auto approx_range = index->search(key_low_bound);
  auto pos = approx_range.pos;
  auto lo = approx_range.lo;
  auto hi = approx_range.hi;

  auto it = std::lower_bound(data_.begin() + lo, data_.begin() + hi, key_low_bound,
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