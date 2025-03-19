#include <filesystem>
#include <iostream>
#include <random>

#include "competitors/finedex/include/stx/btree_multimap.h"
#include "util.h"
#include "utils/cxxopts.hpp"

using namespace std;

// Maximum number of retries to find a lookup key that has at most
// `max_num_retries`.
constexpr size_t max_num_retries = 100;

static mt19937_64 g(42);

enum InsertPat { Equality = 0, Delta = 1, Hotspot = 2 };

vector<size_t> generate_permute(size_t lo, size_t hi, bool is_shuffle = false) {
  vector<size_t> permute;
  permute.reserve(hi - lo);
  for (size_t i = lo; i < hi; i++) {
    permute.push_back(i);
  }
  if (is_shuffle) {
    shuffle(permute.begin(), permute.end(), g);
  }
  return permute;
}

const string to_nice_number(uint64_t num) {
  const uint64_t THOUSAND = 1000;
  const uint64_t MILLION = 1000 * THOUSAND;
  const uint64_t BILLION = 1000 * MILLION;

  if (num >= BILLION && (num / BILLION) * BILLION == num) {
    return to_string(num / BILLION) + "B";
  }
  if (num >= MILLION && (num / MILLION) * MILLION == num) {
    return to_string(num / MILLION) + "M";
  }
  if (num >= THOUSAND && (num / THOUSAND) * THOUSAND == num) {
    return to_string(num / THOUSAND) + "K";
  }
  return to_string(num);
}

template <class KeyType>
void print_op_stats(vector<Operation<KeyType>> const* ops, size_t thread_num) {
  for (size_t i = 0; i < thread_num; i++) {
    size_t negative_count = 0, lookup_count = 0, rq_count = 0, insert_count = 0;
    for (const auto& op : ops[i]) {
      if (op.op == util::LOOKUP) {
        if (op.result == util::NOT_FOUND) {
          ++negative_count;
        }
        ++lookup_count;
        continue;
      }
      if (op.op == util::RANGE_QUERY) {
        ++rq_count;
        continue;
      }
      if (op.op == util::INSERT) {
        ++insert_count;
        continue;
      }
    }
    cout << "Thread's operation count: " << ops[i].size() << endl;
    cout << "Negative lookup ratio: "
         << static_cast<double>(negative_count) / lookup_count << endl;
    cout << "Range query ratio: "
         << static_cast<double>(rq_count) / ops[i].size() << endl;
    cout << "Insert ratio: "
         << static_cast<double>(insert_count) / ops[i].size() << endl;
  }
}

template <class KeyType>
void generate_drift_workload(const string& bulkload_filename,
                             const string& insert_key_filename, size_t op_cnt,
                             double range_query_ratio, size_t range_scope,
                             double negative_lookup_ratio, bool mix, bool fix) {
  util::FastRandom ranny(42);
  // Load data
  vector<KeyType> bulkload_keys = util::load_data<KeyType>(bulkload_filename);
  vector<KeyType> insert_keys = util::load_data<KeyType>(insert_key_filename);

  vector<KeyType> keys = bulkload_keys;
  keys.insert(keys.end(), insert_keys.begin(), insert_keys.end());

  sort(keys.begin(), keys.end(),
       [](const KeyType& a, const KeyType& b) { return a < b; });

  sort(bulkload_keys.begin(), bulkload_keys.end(),
       [](const KeyType& a, const KeyType& b) { return a < b; });

  shuffle(insert_keys.begin(), insert_keys.end(), g);

  std::cout << "Load data done" << std::endl;

  size_t insert_opt_cnt = insert_keys.size();
  if (insert_opt_cnt > op_cnt) {
    util::fail("Insert count exceeds operation count, the insert count is " +
               to_string(insert_opt_cnt) + " and the operation count is " +
               to_string(op_cnt));
    exit(EXIT_FAILURE);
  }

  size_t range_query_cnt = 0, lookup_cnt = 0;
  range_query_cnt = op_cnt * range_query_ratio;
  if (range_query_cnt + insert_opt_cnt >= op_cnt) {
    range_query_cnt = op_cnt - insert_opt_cnt;
  } else {
    lookup_cnt = op_cnt - insert_opt_cnt - range_query_cnt;
  }

  if (fix) {
    lookup_cnt = int(op_cnt * 0.25);
    range_query_cnt = 0;
    op_cnt = insert_opt_cnt + lookup_cnt + range_query_cnt;
  }

  std::filesystem::path p(insert_key_filename);
  std::string workload_dir = p.parent_path().string();
  string op_filename =
      workload_dir + "/workload" + "_ops_" + to_nice_number(op_cnt) + "_" +
      to_string(range_query_ratio) + "rq_" + to_string(range_scope) + "rs_" +
      to_string(negative_lookup_ratio) + "nl_" + "0.500000i";

  // lookup ratio default opt_cnt * 0.25
  if (fix) {
    op_filename += "_fix";
  }

  // totally mixed the ops
  if (mix) {
    op_filename += "_mix";
  }

  // default insert pattern is equality
  op_filename += "_2m";
  string bulkload_op_filename = op_filename + "_bulkload";

  std::cout << "Generate operation idx , mix is " << mix << std::endl;
  std::cout << "Opt count: " << op_cnt << std::endl;
  std::cout << "Insert count: " << insert_opt_cnt << std::endl;
  std::cout << "Lookup count: " << lookup_cnt << std::endl;
  std::cout << "Range query count: " << range_query_cnt << std::endl;
  std::cout << "Range Scope " << range_scope << std::endl;

  std::map<KeyType, int> map_keys_gt_idx;
  for (size_t i = 0; i < keys.size(); ++i) {
    map_keys_gt_idx[keys[i]] = i;
  }

  // finish bulkload file content
  vector<KeyValue<KeyType>> bulk_loads;
  vector<KeyValue<KeyType>> insert_kv;
  bulk_loads.reserve(bulkload_keys.size());
  insert_kv.reserve(insert_keys.size());
  // fill the ground-truth value (position in sorted keys)
  for (size_t i = 0; i < bulkload_keys.size(); i++) {
    KeyValue<KeyType> kv;
    kv.key = bulkload_keys[i];
    kv.value = map_keys_gt_idx[kv.key];
    bulk_loads.push_back(kv);
  }

  for (size_t i = 0; i < insert_keys.size(); i++) {
    KeyValue<KeyType> kv;
    kv.key = insert_keys[i];
    kv.value = map_keys_gt_idx[kv.key];
    insert_kv.push_back(kv);
  }

  vector<Operation<KeyType>> tot_ops(op_cnt);

  vector<size_t> op_id = generate_permute(0, op_cnt, mix);
  std::cout << "Generating operations" << std::endl;

  // fill insert operations
  for (size_t i = 0; i < insert_opt_cnt; i++) {
    tot_ops[op_id[i]].op = util::INSERT;
    tot_ops[op_id[i]].lo_key = insert_kv[i].key;
    tot_ops[op_id[i]].result = insert_kv[i].value;
  }

  for (size_t j = insert_opt_cnt; j < insert_opt_cnt + lookup_cnt; j++) {
    tot_ops[op_id[j]].op = util::LOOKUP;
    if (fix) {
      size_t idx = (j - insert_opt_cnt) * 4;
      idx = idx % keys.size();
      tot_ops[op_id[j]].lo_key = keys[idx];
      tot_ops[op_id[j]].result = idx;
    }
  }

  for (size_t j = insert_opt_cnt + lookup_cnt; j < op_cnt; j++) {
    tot_ops[op_id[j]].op = util::RANGE_QUERY;
  }

  if (!fix) {
    std::cout << "Fill operations done" << std::endl;
    // generate equality lookups and range queries
    std::map<KeyType, uint64_t> data_map;
    vector<KeyValue<KeyType>> data_key_vec = bulk_loads;
    // assign equal to bulkloading keys and values
    auto it = data_map.begin();
    for (const auto& kv : bulk_loads) {
      std::pair<KeyType, uint64_t> e = std::make_pair(kv.key, kv.value);
      it = data_map.insert(it, e);
    }

    float error = 0.05;
    size_t max_num = range_scope ==  0 ? 100 : range_scope;
    std::cout << "Generating equality lookups and range queries" << std::endl;
    for (size_t i = 0; i < tot_ops.size(); i++) {
      if (tot_ops[i].op == util::INSERT) {
        data_map.emplace(static_cast<KeyType>(tot_ops[i].lo_key),
                         static_cast<uint64_t>(tot_ops[i].result));
        KeyValue<KeyType> kv;
        kv.key = tot_ops[i].lo_key;
        kv.value = tot_ops[i].result;
        data_key_vec.push_back(kv);
        continue;
      } else if (tot_ops[i].op == util::LOOKUP) {
        KeyType max_key, min_key;
        max_key = data_map.rbegin()->first;
        min_key = data_map.begin()->first;

        if constexpr (std::is_same<KeyType, uint64_t>::value) {
          if (insert_key_filename.find("fb_200M_uint64") != std::string::npos &&
              max_key > 77308821508) {
            max_key = 77308821508;
          }
        }

        // negative lookup
        if (negative_lookup_ratio > 0 &&
            ranny.ScaleFactor() < negative_lookup_ratio) {
          KeyType negative_lookup;
          bool is_exist = true;
          while (is_exist) {
            negative_lookup =
                (ranny.ScaleFactor() * (max_key - min_key)) + min_key;
            is_exist = data_map.find(negative_lookup) != data_map.end();
          }
          tot_ops[i].lo_key = negative_lookup;
          tot_ops[i].result = util::NOT_FOUND;
          continue;
        }

        const uint64_t offset = ranny.RandUint32(0, data_key_vec.size() - 1);
        const KeyType lookup_key = data_key_vec[offset].key;
        tot_ops[i].lo_key = lookup_key;
        tot_ops[i].result = data_key_vec[offset].value;
        continue;
      } else if (tot_ops[i].op == util::RANGE_QUERY) {
        size_t num_retries = 0;
        bool generated = false;
        KeyType lo_key, hi_key;
        size_t num_qualifying;
        size_t min_num = 0;
        uint64_t result;

        while (!generated) {
          KeyType min_key, max_key;
          min_key = data_map.begin()->first;
          auto tmp_it = data_map.end();
          advance(tmp_it, -max_num - 1);
          max_key = tmp_it->first;

          lo_key = (ranny.ScaleFactor() * (max_key - min_key)) + min_key;
          num_qualifying = 0;
          result = 0;

          auto lo = data_map.lower_bound(lo_key);
          tmp_it = lo;

          if (range_scope <= 0) {
            advance(tmp_it, (1 - ranny.ScaleFactor() * error) * max_num);
          } else {
            advance(tmp_it, max_num);
            // fix the range scope, instead of random
          }
          
          hi_key = tmp_it->first;

          while (lo != data_map.end() && lo->first <= hi_key) {
            result += lo->second;
            ++num_qualifying;
            ++lo;
          }

          if (num_qualifying > max_num || num_qualifying < min_num) {
            ++num_retries;
            if (num_retries > max_num_retries) {
              util::fail(
                  "Generate_equality_lookups: exceeded max number of retries");
              continue;
            }
          }
          tot_ops[i].lo_key = lo_key;
          tot_ops[i].hi_key = hi_key;
          tot_ops[i].result = result;
          generated = true;
        }
      } else {
        util::fail("Undefined operation");
      }
    }
  }

  // print_op_stats
  print_op_stats(&tot_ops, 1);

  // save files
  util::write_data(bulk_loads, bulkload_op_filename);
  util::write_data(tot_ops, op_filename);
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("generate", "Generate operations on sorted data");
  options.positional_help("<bulkloading_data> <insert_data> <operation-count>");
  options.add_options()("bulkload_data", "bulkload_data file with keys",
                        cxxopts::value<string>())(
      "insert_data", "insert_data file with keys", cxxopts::value<string>())(
      "operation-count", "Number of operations", cxxopts::value<size_t>())(
      "help", "Displays help")("n,negative-lookup-ratio",
                               "Negative lookup ratio",
                               cxxopts::value<double>()->default_value("0"))(
      "r, scan-range", "Range query scope",
      cxxopts::value<size_t>()->default_value("0"))(
      "s,scan-ratio", "Range query ratio",
      cxxopts::value<double>()->default_value("0"))(
      "mix", "Mix lookups, range queries and inserts together")(
      "fix", "fixed-read-workload");

  options.parse_positional({"bulkload_data", "insert_data", "operation-count"});

  const auto result = options.parse(argc, argv);

  if (result.count("help")) {
    cout << options.help({}) << "\n";
    exit(0);
  }

  const bool mix = result.count("mix");
  const bool fix = result.count("fix");
  const string bulkload_filename = result["bulkload_data"].as<string>();
  const string insert_filename = result["insert_data"].as<string>();
  const DataType type = util::resolve_type(bulkload_filename);

  size_t range_scope = result["scan-range"].as<size_t>();
  size_t op_cnt = result["operation-count"].as<size_t>();

  double range_query_ratio = result["scan-ratio"].as<double>(),
         negative_lookup_ratio = result["negative-lookup-ratio"].as<double>();

  if (negative_lookup_ratio < 0 || negative_lookup_ratio > 1 ||
      range_query_ratio < 0 || range_query_ratio > 1) {
    util::fail("workload ratio must be between 0 and 1.");
  }

  switch (type) {
    case DataType::UINT32: {
      generate_drift_workload<uint32_t>(bulkload_filename, insert_filename,
                                        op_cnt, range_query_ratio, range_scope,
                                        negative_lookup_ratio, mix, fix);
      break;
    }
    case DataType::UINT64: {
      std::cout << "Generating workload for UINT64" << std::endl;
      generate_drift_workload<uint64_t>(bulkload_filename, insert_filename,
                                        op_cnt, range_query_ratio, range_scope,
                                        negative_lookup_ratio, mix, fix);
      break;
    }
    case DataType::STRING: {
      // generate_drift_workload<string>(bulkload_filename,
      // insert_filename, op_cnt, range_query_ratio, negative_lookup_ratio,
      // mix); doesn't support STRING type
      util::fail("STRING type is not supported.");
      break;
    }
  }

  return 0;
}
