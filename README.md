# NeuralBench

> Benchmarking Learned Databases with Data and Workload Drift Modeling

Learned databases have been extensively studied in recent years. Given the dynamism of databases, where data and workloads continuously drift, it is crucial for their components to effectively adapt to such drift. *Adaptability*, therefore, becomes a key factor in assessing their practical applicability.

However, existing benchmarks for learned databases either overlook or oversimplify the treatment of data and workload drift, failing to evaluate learned database components across a broad range of drift conditions.

**NeuralBench** is a new benchmark framework designed with controllable data and workload drift. It creates a unified concept called *drift factor* to precisely quantify and generate drift. Based on this formulation, a benchmark suite is then developed to enable standardized performance evaluations of various learned database components under measurable and adjustable drift conditions.

This repository contains the code of NeuralBench, which is used to evaluate learned query optimizers, concurrency control, and indexes.

## Benchmarks

### Learned Query Optimziers

Please check the documentation [here](./benchmarks/lqos/README.md).

### Learn Concurrency Control

The benchmarks are conducted for Polyjuice. Please check the documentation at

<https://github.com/derFischer/Polyjuice/tree/master/ae-tpcc-polyjuice>

to set up the testbed.

The experiments are done with the default config, i.e.,

```ini
selection=truncation
psize=8
random_branch=4
mutate_rate=0.05
pickup_policy=./training/input-RL-ic3-new-tpcc.txt
```

### Learned Index

To be updated

## Tools & Utilities

### TPC-H Data Generator (Official)

```bash
cd tpch-kit/dbgen
make
mkdir 1g
./dbgen -s 1 -v # generate 1GB data
mv *.tbl 1g/
```

### TBL to CSV

```bash
python tbl2csv.py -d tpch -i tpch-kit/dbgen/1g -o tpch-kit/dbgen/1g_csv
```

### DB Processor (dbproc)

Use `customer` table as an example:

```bash
# drift factor
d=0.4
# number of bins
nbins=20 
# whether to skew the data distribution 
# 0 means use uniform distribution to flatten the distribution
skewed=1 
python dbproc.py -t customer -i tpch-kit/dbgen/1g/customer.tbl -o 1.tbl -b $nbins -D $d -s $skewed
```

You can chain the processor to simulate the continous distribution drifts:

```bash
python dbproc.py -t customer -i tpch-kit/dbgen/1g/customer.tbl -o 1.tbl -b $nbins -D $d -s $skewed
python dbproc.py -t customer -i 1.tbl -o 2.tbl -b $nbins -D $d -s $skewed
python dbproc.py -t customer -i 2.tbl -o 3.tbl -b $nbins -D $d -s $skewed
# ...
```

### Query Processor (qpre, qproc)

```bash
# drift factor
d=0.4
# whether to skew the data distribution 
# 0 means use uniform distribution to flatten the distribution
skewed=1 
# number of samples
n=1000
# type of metadata to drift. for available types, see `python qproc.py -h`
type=tables
python qproc.py -t $type -I devtest/testdata/queries -o 1.sql -D $d -s $skewed -n $n
```
