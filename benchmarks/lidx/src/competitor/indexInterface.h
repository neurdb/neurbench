#include <iomanip>
#include <cmath>
#include <algorithm>
#include <atomic>

#pragma once

struct Param { // for xindex
  size_t worker_num;
  uint32_t thread_id;

  Param(size_t worker_num, uint32_t thread_id) : worker_num(worker_num), thread_id(thread_id) {}
};

// Prediction error statistics
struct PredictionStats {
  bool supported = false;        // whether this index supports prediction error tracking
  double avg_error = 0.0;        // average |predicted_pos - actual_pos|
  double max_error = 0.0;        // maximum prediction error
  double p50_error = 0.0;        // 50th percentile error
  double p99_error = 0.0;        // 99th percentile error
  long long total_lookups = 0;   // number of lookups tracked
  long long total_error = 0;     // sum of all errors

  void reset() {
    avg_error = max_error = p50_error = p99_error = 0.0;
    total_lookups = total_error = 0;
  }
};

struct BaseCompare {
  template<class T1, class T2>
  bool operator()(const T1 &x, const T2 &y) const {
    static_assert(
      std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
      "Comparison types must be numeric.");
    return x < y;
  }
};

template<class KEY_TYPE, class PAYLOAD_TYPE, class KeyComparator=BaseCompare>
class indexInterface {
public:
  virtual void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr) = 0;

  virtual bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) = 0;

  virtual bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) = 0;

  virtual bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) = 0;

  virtual bool remove(KEY_TYPE key, Param *param = nullptr) = 0;

  virtual size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                      Param *param = nullptr) = 0;

  virtual void init(Param *param = nullptr) = 0;

  virtual long long memory_consumption() = 0; // bytes

  // Get prediction error statistics (for learned indexes)
  virtual PredictionStats get_prediction_stats() {
    PredictionStats stats;
    stats.supported = false;
    return stats;
  }

  // Reset prediction error tracking
  virtual void reset_prediction_stats() {}
};