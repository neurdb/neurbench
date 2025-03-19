#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <string>
#include <set>

#include "math.h"
#include <getopt.h>
#include <numa.h>
#include <random>
#include <stdlib.h>
#include <unistd.h>
#include "chrono"

#include "../macros.h"
#include "../varkey.h"
#include "../thread.h"
#include "../util.h"
#include "../spinbarrier.h"
#include "../core.h"
#include "../txn.h"

#include "bench.h"

using namespace std;
using namespace util;

static size_t nkeys;
static const size_t YCSBRecordSize = 100;
static const int ycsb_records_per_partition = 10000;

enum YCSBOpt {
  ReadOpt = 0,
  WriteOpt,
  ScanReadOpt,
  ScanWriteOpt
};

static double g_txn_access_distribution[32] = {0};
static YCSBOpt g_txn_op_distribution[32] = {ReadOpt};

static std::uint64_t g_txn_length = 10;
static bool g_access_partitioned = false;

static int key_distribution[100] = {0};

enum ycsb_tx_types
{
  ycsb_type = 1,
  end_type
};

inline double zeta(int n, double theta) {
  double z = 0.0;
  for (int i = 1; i <= n; ++i) {
    z += pow((double)1.0 / i, theta);
  }
  return z;
}

class ZipfianGenerator {
public:
  static const double ZipfianConstant = 0.99;

  ZipfianGenerator(int64_t items, double zipfianConstant = ZipfianConstant)
      : ZipfianGenerator(0, items - 1, zipfianConstant) {}

  ZipfianGenerator(int64_t min, int64_t max, double zipfianConstant = ZipfianConstant)
      : base(min), items(max - min + 1), zipfianConstant(zipfianConstant), theta(zipfianConstant) {
    zetan = zeta(0, items, theta, 0);
    zeta2Theta = zeta(0, 2, theta, 0);
    alpha = 1.0 / (1.0 - theta);
    eta = (1 - pow(2.0 / static_cast<double>(items), 1 - theta)) / (1 - zeta2Theta / zetan);

    // Seed the random number generator
    rng.seed(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
  }

  int64_t next_value() {
    std::lock_guard<std::mutex> lock(mutex_);
    double u = uniform_dist(rng);
    double uz = u * zetan;

    if (uz < 1.0) {
      return base;
    }

    if (uz < 1.0 + pow(0.5, theta)) {
      return base + 1;
    }

    return base + static_cast<int64_t>(items * pow(eta * u - eta + 1, alpha));
  }

private:
  int64_t base;
  int64_t items;
  double zipfianConstant;
  double theta;
  double alpha;
  double zetan;
  double zeta2Theta;
  double eta;

  std::mt19937 rng;
  std::uniform_real_distribution<double> uniform_dist{0.0, 1.0};
  std::mutex mutex_;

  double zeta(int64_t start, int64_t n, double thetaVal, double initialSum) {
    double sum = initialSum;
    for (int64_t i = start; i < n; ++i) {
      sum += 1.0 / pow(static_cast<double>(i + 1), thetaVal);
    }
    return sum;
  }
};

static conflict_graph* cgraph = NULL;
ZipfianGenerator * key_gen_list[32] = {nullptr};

class ycsb_worker : public bench_worker {
public:
  ycsb_worker(unsigned int worker_id,
              unsigned long seed, abstract_db *db,
              const map<string, abstract_ordered_index *> &open_tables,
              spin_barrier *barrier_a, spin_barrier *barrier_b)
    : bench_worker(worker_id, true, seed, db,
                   open_tables, barrier_a, barrier_b),
      tbl(open_tables.at("USERTABLE")),
      computation_n(0)
  {
    obj_key0.reserve(str_arena::MinStrReserveLength);
    obj_key1.reserve(str_arena::MinStrReserveLength);
    obj_v.reserve(str_arena::MinStrReserveLength);
  }

  txn_result
  txn()
  {
    void *txn = db->new_txn(txn_flags, arena, txn_buf(), abstract_db::HINT_KV_GET_PUT);
    db->init_txn(txn, cgraph, ycsb_type, policy, pg);
    db->set_failed_records(txn, failed_records);
    auto partition_size = (int)nkeys/g_txn_length;

    scoped_str_arena s_arena(arena);
    std::pair<bool, uint32_t> expose_ret;
    try {
      auto keys = new uint64_t [int(g_txn_length)];
      for (int i=0;i<g_txn_length;i++) {
        if (g_access_partitioned) {
          // all accesses locate at different partition.
          if (g_txn_op_distribution[i] == ScanWriteOpt ||
              g_txn_op_distribution[i] == ScanReadOpt) {
            keys[i] = key_gen_list[i]->next_value() % (partition_size-10) +
                      i * partition_size;
          }
          else keys[i] = key_gen_list[i]->next_value()  % partition_size + i * partition_size;
        } else {
          if (g_txn_op_distribution[i] == ScanWriteOpt ||
              g_txn_op_distribution[i] == ScanReadOpt)
            keys[i] = key_gen_list[i]->next_value() % (nkeys/-10);
          else keys[i] = key_gen_list[i]->next_value() % nkeys;
          if (keys[i] < 100) {
            key_distribution[keys[i]] ++;
          }
        }
      }
      int acc_id = 0; // 0 <= acc_id <= txn_length *2
      for (int i=0;i<g_txn_length;) {
        auto row_id = keys[i];
        auto op = g_txn_op_distribution[i];
        obj_key0 = u64_varkey(row_id).str(obj_key0);
        if (op == ReadOpt) {
          // read operation,
          ALWAYS_ASSERT(tbl->get(txn, u64_varkey(row_id).str(obj_key0), obj_v, acc_id));
        } else if (op == WriteOpt) {
          // read modify write.
          ALWAYS_ASSERT(tbl->get(txn, u64_varkey(row_id).str(obj_key0), obj_v, acc_id));
          tbl->put(txn, obj_key0, str().assign(YCSBRecordSize, 'a' + rand() % 26), acc_id+1);
        } else if (op == ScanReadOpt) {
          for (int j = 0; j < 10; j++) {
            ALWAYS_ASSERT(tbl->get(txn, u64_varkey(row_id + j).str(obj_key0),
                                   obj_v, acc_id));
          }
        } else if (op == ScanWriteOpt) {
          for (int j = 0; j < 10; j++) {
            ALWAYS_ASSERT(tbl->get(txn, u64_varkey(row_id + j).str(obj_key0),
                                   obj_v, acc_id));
            tbl->put(txn, obj_key0, str().assign(YCSBRecordSize, 'a' + rand() % 26), acc_id+1);
          }
        } else {
          // unsupported yet.
          assert(false);
        }
        expose_ret = db->expose_uncommitted(txn, acc_id + ACCESSES /*access_id*/);
        if (!expose_ret.first) {
          auto next = expose_ret.second == MAX_ACC_ID? 0: expose_ret.second;
          i = next;
        } else {
          i ++;
          acc_id += (op == ScanWriteOpt || op == WriteOpt)? 2: 1;
        }
      }
      measure_txn_counters(txn, "txn_read_write");
      bool res = db->commit_txn(txn);
      set_failed_records(db->get_failed_records(txn));
      finished_txn_contention = db->get_txn_contention(txn);
      return txn_result(res, ycsb_type);
    } catch(transaction_abort_exception &ex) {
      db->abort_txn(txn);
    } catch (abstract_db::abstract_abort_exception &ex) {
      db->abort_txn(txn);
    }
    return txn_result(false, 0);
  }

  static txn_result
  Txn(bench_worker *w)
  {
    return static_cast<ycsb_worker *>(w)->txn();
  }


  class worker_scan_callback : public abstract_ordered_index::scan_callback {
  public:
    worker_scan_callback() : n(0) {}
    virtual bool
    invoke(const char *, size_t, const string &value)
    {
      n += value.size();
      return true;
    }
    size_t n;
  };

  virtual workload_desc_vec
  get_workload() const
  {
    workload_desc_vec w;
    w.emplace_back("RW_TX",  1.0, Txn);
    return w;
  }

protected:

  virtual void
  on_run_setup() OVERRIDE
  {
    if (!pin_cpus)
      return;
    const size_t a = worker_id % coreid::num_cpus_online();
    const size_t b = a % nthreads;
    rcu::s_instance.pin_current_thread(b);
  }

  inline ALWAYS_INLINE string &
  str() {
    return *arena.next();
  }

private:
  abstract_ordered_index *tbl;

  string obj_key0;
  string obj_key1;
  string obj_v;

  uint64_t computation_n;
};

static void
ycsb_load_keyrange(
    uint64_t keystart,
    uint64_t keyend,
    unsigned int pinid,
    abstract_db *db,
    abstract_ordered_index *tbl,
    str_arena &arena,
    uint64_t txn_flags,
    void *txn_buf)
{
  if (pin_cpus) {
    ALWAYS_ASSERT(pinid < nthreads);
    rcu::s_instance.pin_current_thread(pinid);
    rcu::s_instance.fault_region();
  }

  const size_t batchsize = (db->txn_max_batch_size() == -1) ?
    10000 : db->txn_max_batch_size();
  ALWAYS_ASSERT(batchsize > 0);
  const size_t nkeys = keyend - keystart;
  ALWAYS_ASSERT(nkeys > 0);
  const size_t nbatches = nkeys < batchsize ? 1 : (nkeys / batchsize);
  for (size_t batchid = 0; batchid < nbatches;) {
    scoped_str_arena s_arena(arena);
    void * const txn = db->new_txn(txn_flags, arena, txn_buf);
    try {
      const size_t rend = (batchid + 1 == nbatches) ?
        keyend : keystart + ((batchid + 1) * batchsize);
      for (size_t i = batchid * batchsize + keystart; i < rend; i++) {
        ALWAYS_ASSERT(i >= keystart && i < keyend);
        const string k = u64_varkey(i).str();
        const string v(YCSBRecordSize, 'a');
        tbl->insert(txn, k, v);
      }
      if (db->commit_txn(txn))
        batchid++;
      else
        db->abort_txn(txn);
    } catch (abstract_db::abstract_abort_exception &ex) {
      db->abort_txn(txn);
    }
  }
  if (verbose)
    cerr << "[INFO] finished loading USERTABLE range [kstart="
      << keystart << ", kend=" << keyend << ") - nkeys: " << nkeys << endl;
}

class ycsb_usertable_loader : public bench_loader {
public:
  ycsb_usertable_loader(unsigned long seed,
                        abstract_db *db,
                        const map<string, abstract_ordered_index *> &open_tables)
    : bench_loader(seed, db, open_tables)
  {}

protected:
  virtual void
  load()
  {
    abstract_ordered_index *tbl = open_tables.at("USERTABLE");
    const size_t nkeysperthd = nkeys / nthreads;
    for (size_t i = 0; i < nthreads; i++) {
      const size_t keystart = i * nkeysperthd;
      const size_t keyend = min((i + 1) * nkeysperthd, nkeys);
      ycsb_load_keyrange(
          keystart,
          keyend,
          i,
          db,
          tbl,
          arena,
          txn_flags,
          txn_buf());
    }
  }
};

class ycsb_parallel_usertable_loader : public bench_loader {
public:
  ycsb_parallel_usertable_loader(unsigned long seed,
                                 abstract_db *db,
                                 const map<string, abstract_ordered_index *> &open_tables,
                                 unsigned int pinid,
                                 uint64_t keystart,
                                 uint64_t keyend)
    : bench_loader(seed, db, open_tables),
      pinid(pinid), keystart(keystart), keyend(keyend)
  {
    INVARIANT(keyend > keystart);
    if (verbose)
      cerr << "[INFO] YCSB par loader cpu " << pinid
           << " [" << keystart << ", " << keyend << ")" << endl;
  }

protected:
  virtual void
  load()
  {
    abstract_ordered_index *tbl = open_tables.at("USERTABLE");
    ycsb_load_keyrange(
        keystart,
        keyend,
        pinid,
        db,
        tbl,
        arena,
        txn_flags,
        txn_buf());
  }

private:
  unsigned int pinid;
  uint64_t keystart;
  uint64_t keyend;
};


class ycsb_bench_runner : public bench_runner {
public:
  ycsb_bench_runner(abstract_db *db)
    : bench_runner(db)
  {
    open_tables["USERTABLE"] = db->open_index("USERTABLE", YCSBRecordSize);
  }

protected:
  virtual vector<bench_loader *>
  make_loaders()
  {
    vector<bench_loader *> ret;
    const unsigned long ncpus = coreid::num_cpus_online();
    if (enable_parallel_loading && nkeys >= nthreads) {
      // divide the key space amongst all the loaders
      const size_t nkeysperloader = nkeys / ncpus;
      if (nthreads > ncpus) {
        for (size_t i = 0; i < ncpus; i++) {
          const uint64_t kend = (i + 1 == ncpus) ?
            nkeys : (i + 1) * nkeysperloader;
          ret.push_back(
              new ycsb_parallel_usertable_loader(
                0, db, open_tables, i,
                i * nkeysperloader, kend));
        }
      } else {
        // load balance the loaders amongst numa nodes in RR fashion
        //
        // XXX: here we hardcode an assumption about the NUMA topology of
        // the system
        const vector<unsigned> numa_nodes_used = get_numa_nodes_used(nthreads);

        // assign loaders to cores based on numa node assignment in RR fashion
        const unsigned loaders_per_node = ncpus / numa_nodes_used.size();

        vector<unsigned> node_allocations(numa_nodes_used.size(), loaders_per_node);
        // RR the remaining
        for (unsigned i = 0;
             i < (ncpus - loaders_per_node * numa_nodes_used.size());
             i++)
          node_allocations[i]++;

        size_t loader_i = 0;
        for (size_t i = 0; i < numa_nodes_used.size(); i++) {
          // allocate loaders_per_node loaders to this numa node
          const vector<unsigned> cpus = numa_node_to_cpus(numa_nodes_used[i]);
          const vector<unsigned> cpus_avail = exclude(cpus, nthreads);
          const unsigned nloaders = node_allocations[i];
          for (size_t j = 0; j < nloaders; j++, loader_i++) {
            const uint64_t kend = (loader_i + 1 == ncpus) ?
              nkeys : (loader_i + 1) * nkeysperloader;
            ret.push_back(
                new ycsb_parallel_usertable_loader(
                  0, db, open_tables, cpus_avail[j % cpus_avail.size()],
                  loader_i * nkeysperloader, kend));
          }
        }
      }
    } else {
      ret.push_back(new ycsb_usertable_loader(0, db, open_tables));
    }
    return ret;
  }

  virtual vector<bench_worker *>
  make_workers()
  {
    const unsigned alignment = coreid::num_cpus_online();
    const int blockstart =
      coreid::allocate_contiguous_aligned_block(nthreads, alignment);
    ALWAYS_ASSERT(blockstart >= 0);
    ALWAYS_ASSERT((blockstart % alignment) == 0);
    fast_random r(8544290);
    vector<bench_worker *> ret;
    for (size_t i = 0; i < nthreads; i++)
      ret.push_back(
        new ycsb_worker(
          blockstart + i, r.next(), db, open_tables,
          &barrier_a, &barrier_b));
    return ret;
  }

private:

  static vector<unsigned>
  get_numa_nodes_used(unsigned nthds)
  {
    // assuming CPUs [0, nthds) are used, what are all the
    // NUMA nodes touched by [0, nthds)
    set<unsigned> ret;
    for (unsigned i = 0; i < nthds; i++) {
      const int node = numa_node_of_cpu(i);
      ALWAYS_ASSERT(node >= 0);
      ret.insert(node);
    }
    return vector<unsigned>(ret.begin(), ret.end());
  }

  static vector<unsigned>
  numa_node_to_cpus(unsigned node)
  {
    struct bitmask *bm = numa_allocate_cpumask();
    ALWAYS_ASSERT(!::numa_node_to_cpus(node, bm));
    vector<unsigned> ret;
    for (int i = 0; i < numa_num_configured_cpus(); i++)
      if (numa_bitmask_isbitset(bm, i))
        ret.push_back(i);
    numa_free_cpumask(bm);
    return ret;
  }

  static vector<unsigned>
  exclude(const vector<unsigned> &cpus, unsigned nthds)
  {
    vector<unsigned> ret;
    for (auto n : cpus)
      if (n < nthds)
        ret.push_back(n);
    return ret;
  }

};

void
ycsb_do_test(abstract_db *db, int argc, char **argv)
{
  nkeys = size_t(scale_factor * ycsb_records_per_partition);
  ALWAYS_ASSERT(nkeys > 0);

  // parse options
  optind = 1;
  while (1) {
    static struct option long_options[] = {
        {"access-dist" , required_argument , 0 , 'a'},
        {"opt-dist" , required_argument , 0 , 'o'},

        // for single part only.
        {"length", required_argument, 0, 'l'},
        {"partition", no_argument, 0, 'p'},
        {0, 0, 0, 0}
    };
    int option_index = 0;
    int c = getopt_long(argc, argv, "a:o:p:l", long_options, &option_index);
    if (c == -1)
      break;
    switch (c) {
    case 0:
      if (long_options[option_index].flag != 0)
        break;
      abort();
      break;

    case 'a':
    {
      const vector<string> toks = split(optarg, ',');
      for (size_t i = 0; i < toks.size(); i++) {
        double p = strtof(toks[i].c_str(), nullptr);
        ALWAYS_ASSERT(p >= 0 && p <= 1);
        g_txn_access_distribution[i] = p;
      }
    }
    break;

    case 'o':
    {
      const vector<string> toks = split(optarg, ',');
      for (size_t i = 0; i < toks.size(); i++) {
        if (toks[i] == "r") {
          g_txn_op_distribution[i] = ReadOpt;
        } else if (toks[i] == "w") {
          g_txn_op_distribution[i] = WriteOpt;
        } else if (toks[i] == "sr") {
          g_txn_op_distribution[i] = ScanReadOpt;
        } else if (toks[i] == "sw") {
          g_txn_op_distribution[i] = ScanWriteOpt;
        }
      }
    }
    break;

    case 'p':
      g_access_partitioned = true;
      break;

    case 'l':
    {
      int op = static_cast<int>(strtol(optarg, nullptr, 10));
      ALWAYS_ASSERT(op > 0);
      g_txn_length = op;
    }
    break;

    case '?':
      /* getopt_long already printed an error message. */
      exit(1);

    default:
      abort();
    }
  }

  for (int i=0;i<g_txn_length;i++)
    key_gen_list[i] =
        new ZipfianGenerator(int(ycsb_records_per_partition), g_txn_access_distribution[i]);

  if (verbose) {
    cerr << "ycsb settings:" << endl;
    cerr << "  access_dist: "
         << format_list(g_txn_access_distribution, g_txn_access_distribution + (g_txn_length)) << std::endl
         << "  opt_dist: "
         << format_list(g_txn_op_distribution, g_txn_op_distribution + (g_txn_length)) << std::endl
         << "  table size: "
         << nkeys
         << std::endl;
  }

  ycsb_bench_runner r(db);
  r.run();
}
