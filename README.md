# NeurBench

**NeurBench** is a benchmark suite designed to evaluate end-to-end learned DBMSs containing all learned components under controllable data and workload drift.


<!-- ## Dependencies -->


## Tools & Utilities

NeurBench provides a drift-aware data and workload generation tool that effectively simulates real-world drift while preserving inherent correlations.

### Data and Workload Generator

Run the code to generate data and workloads according to a specified drift factor with the following command:

```
python gen.py --dataset-name=[dataset] --table-name=[table] --drift=[drift factor]
```

For example, to generate a drifted `Name` table for the default dataset (`IMDB`) with a drift factor of 0.1, we can run the following command:

```
python gen.py --dataset-name=imdb --table-name=name --drift=0.1
```

To generate default workloads with  a drift factor of 0.1, we can can run this command:

```
python gen.py --dataset-name=imdb --table-name=workload --drift=0.1
```



## Benchmarks

We employ NeurBench to evaluate state-of-the-art learned query optimizers, learned indexes, and learned concurrency control within a consistent experimental process.

### Learned Query Optimziers

Please check the documentation [here](./benchmarks/lqos/README.md).

The main code for the benchmarks is in `benchmarks/lqos` and `neurbench/query`.

### Learned Index

Please check the documentation [here](./benchmarks/lidx/README.md).

The main code for the benchmarks is in `benchmarks/lidx` and `neurbench/index`.


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

