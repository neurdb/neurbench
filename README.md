# NeurBench

**NeurBench** is a benchmark suite designed to evaluate end-to-end learned DBMSs containing all learned components under controllable data and workload drift.


## Dependencies

We provide an `environment.yml` configured for CUDA 11.8. You may modify this file to match your local CUDA version if needed.  
To create the conda environment, run:

```
conda env create -f environment.yml
```


## Tools & Utilities

NeurBench provides a drift-aware data and workload generation tool that effectively simulates real-world drift while preserving inherent correlations.

### Data and Workload Generator

Run the code to generate data according to a specified drift factor with the following command:

```
python dbproc.py --dataset-name=[dataset] --table-name=[table] --drift=[drift factor]
```

For example, to generate a drifted `Name` table for the default dataset (`IMDB`) with a drift factor of `0.1`, we can run the following command:

```
python dbproc.py --dataset-name=imdb --table-name=name --drift=0.1
```

Run the code to generate workloads according to a specified drift factor with the following command:

```
python qproc.py --input_file=[original workload] --output=[drifted workload] --drift=[drift factor]
```

For example, to generate default workloads with a drift factor of 0.1, we can can run this command:

```
python qproc.py --input_file=orig_queries.sql --output=drifted_01_queries.sql --drift=0.1
```


## Benchmarks

We employ NeurBench to evaluate state-of-the-art learned query optimizers, learned indexes, and learned concurrency control. We include the codes of evaluators that we used in `benchmark` folder.
