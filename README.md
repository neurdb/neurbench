# NeurBench

> Benchmarking Learned Databases with Data and Workload Drift Modeling

## Usage

### Generate TPC-H data

```bash
cd tpch-kit/dbgen
make
mkdir 1g
./dbgen -s 1 -v # generate 1GB data
mv *.tbl 1g/
```

### Run DB Processor (dbproc)

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

### Run Query Processor (qpre, qproc)

#### Generate base queries

```bash
DSS_QUERY=./queries ./qgen -v -d -s 1 -r 42 > tpch-stream.sql
```

#### Preprocess

```bash
python qpre.py -i tpch-kit/dbgen/tpch-stream.sql
```

#### Convert tbl to csv

```bash
python tbl2csv.py -i tpch-kit/dbgen/1g -o tpch-kit/dbgen/1g_csv
```

#### Process

```bash
python parse_sql_metadata.py -i ./testdata/tpch-pp.sql -o ./result
```

TODO
