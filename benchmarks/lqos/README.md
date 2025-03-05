# Code

How to clone all codes [Development Documentation](./doc/dev.md)

# Debugging

```bash
# py-spy to check which part of code is slowest
```



# Dyanmic Workload Generation

## Set the env

```bash
conda env remove --name neurbench
conda env create -f environment.yml
conda activate neurbench
```

## Set the config

```bash
# drift factor
d=0.05
# number of bins
nbins=20
# whether to skew the data distribution
# 0 means use uniform distribution to flatten the distribution
skewed=1
```

## Data shift

1. customer

   ```bash
   python dbproc.py -t customer -i ./neurbench/tpch-kit/dbgen/data_0/customer.tbl -o ./neurbench/tpch-kit/dbgen/data_1/customer.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t customer -i ./neurbench/tpch-kit/dbgen/data_1/customer.tbl -o ./neurbench/tpch-kit/dbgen/data_2/customer.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t customer -i ./neurbench/tpch-kit/dbgen/data_2/customer.tbl -o ./neurbench/tpch-kit/dbgen/data_3/customer.tbl -b 20 -D 0.05 -s 1
   ```

2. lineitem

   Here we run with script

   ```bash
   cd ./neurbench
   chmod +x ../../experiments/benchmark/gen_data.sh
   nohup   ../../experiments/benchmark/gen_data.sh   > output.log 2>&1 &
   ```

   ```bash
   python dbproc.py -t lineitem -i ./neurbench/tpch-kit/dbgen/data_0/lineitem.tbl -o ./neurbench/tpch-kit/dbgen/data_1/lineitem.tbl -b 5 -D 0.05 -s 1

   python dbproc.py -t lineitem -i ./neurbench/tpch-kit/dbgen/data_1/lineitem.tbl -o ./neurbench/tpch-kit/dbgen/data_2/lineitem.tbl -b 5 -D 0.05 -s 1

   python dbproc.py -t lineitem -i ./neurbench/tpch-kit/dbgen/data_2/lineitem.tbl -o ./neurbench/tpch-kit/dbgen/data_3/lineitem.tbl -b 5 -D 0.05 -s 1
   ```



3. nation

   ```bash
   python dbproc.py -t nation -i ./neurbench/tpch-kit/dbgen/data_0/nation.tbl -o ./neurbench/tpch-kit/dbgen/data_1/nation.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t nation -i ./neurbench/tpch-kit/dbgen/data_1/nation.tbl -o ./neurbench/tpch-kit/dbgen/data_2/nation.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t nation -i ./neurbench/tpch-kit/dbgen/data_2/nation.tbl -o ./neurbench/tpch-kit/dbgen/data_3/nation.tbl -b 20 -D 0.05 -s 1
   ```

4. ordersw

   ```bash
   python dbproc.py -t orders -i ./neurbench/tpch-kit/dbgen/data_0/orders.tbl -o ./neurbench/tpch-kit/dbgen/data_1/orders.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t orders -i ./neurbench/tpch-kit/dbgen/data_1/orders.tbl -o ./neurbench/tpch-kit/dbgen/data_2/orders.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t orders -i ./neurbench/tpch-kit/dbgen/data_2/orders.tbl -o ./neurbench/tpch-kit/dbgen/data_3/orders.tbl -b 20 -D 0.05 -s 1
   ```

5. part

   ```bash
   python dbproc.py -t part -i ./neurbench/tpch-kit/dbgen/data_0/part.tbl -o ./neurbench/tpch-kit/dbgen/data_1/part.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t part -i ./neurbench/tpch-kit/dbgen/data_1/part.tbl -o ./neurbench/tpch-kit/dbgen/data_2/part.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t part -i ./neurbench/tpch-kit/dbgen/data_2/part.tbl -o ./neurbench/tpch-kit/dbgen/data_3/part.tbl -b 20 -D 0.05 -s 1
   ```

6. partsupp

   ```bash
   python dbproc.py -t partsupp -i ./neurbench/tpch-kit/dbgen/data_0/partsupp.tbl -o ./neurbench/tpch-kit/dbgen/data_1/partsupp.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t partsupp -i ./neurbench/tpch-kit/dbgen/data_1/partsupp.tbl -o ./neurbench/tpch-kit/dbgen/data_2/partsupp.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t partsupp -i ./neurbench/tpch-kit/dbgen/data_2/partsupp.tbl -o ./neurbench/tpch-kit/dbgen/data_3/partsupp.tbl -b 20 -D 0.05 -s 1
   ```

7. region

   ```bash
   python dbproc.py -t region -i ./neurbench/tpch-kit/dbgen/data_0/region.tbl -o ./neurbench/tpch-kit/dbgen/data_1/region.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t region -i ./neurbench/tpch-kit/dbgen/data_1/region.tbl -o ./neurbench/tpch-kit/dbgen/data_2/region.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t region -i ./neurbench/tpch-kit/dbgen/data_2/region.tbl -o ./neurbench/tpch-kit/dbgen/data_3/region.tbl -b 20 -D 0.05 -s 1
   ```

8. supplier

   ````bash
   python dbproc.py -t supplier -i ./neurbench/tpch-kit/dbgen/data_0/supplier.tbl -o ./neurbench/tpch-kit/dbgen/data_1/supplier.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t supplier -i ./neurbench/tpch-kit/dbgen/data_1/supplier.tbl -o ./neurbench/tpch-kit/dbgen/data_2/supplier.tbl -b 20 -D 0.05 -s 1

   python dbproc.py -t supplier -i ./neurbench/tpch-kit/dbgen/data_2/supplier.tbl -o ./neurbench/tpch-kit/dbgen/data_3/supplier.tbl -b 20 -D 0.05 -s 1
   ````

## Query Shift Pre

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

## Query Shift (IMDB)

### For Lero

```bash
# join queries
python qproc.py -t joins  -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o q_train_imdb_join_lero_01 -D 0.1 -s 1 -n 100


python qproc.py -t joins  -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o q_train_imdb_join_lero_03 -D 0.3 -s 1 -n 100


python qproc.py -t joins  -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o q_train_imdb_join_lero_05 -D 0.5 -s 1 -n 100


python qproc.py -t joins  -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o q_train_imdb_join_lero_07 -D 0.7 -s 1 -n 100


# predicates join
python qproc.py -d imdb -t predicates  -I ./neurbench/datasets/dynamic_datasets/imdb/q_train_sample_80% -o q_train_imdb_predicates_01 -D 0.1 -s 1 -n 70

# table join
python qproc.py -d imdb -t tables  -I  ./neurbench/datasets/dynamic_datasets/imdb/q_train_sample_80% -o q_train_imdb_table_01 -D 0.1 -s 1 -n 70
```

### For Bao

d=0.5

```bash
# train data
python qproc.py -d imdb -t joins  -I ./neurbench/datasets/dynamic_datasets/imdb/q_train_sample_80% -o q_train_imdb_join_05 -D 0.5 -s 1 -n 70
```

d=0.3

```bash
# train data
python qproc.py -d imdb -t joins  -I ./neurbench/datasets/dynamic_datasets/imdb/q_train_sample_80% -o q_train_imdb_join_03 -D 0.3 -s 1 -n 70
```

d=0.7

```bash
# train data
python qproc.py -d imdb -t joins  -I ./neurbench/datasets/dynamic_datasets/imdb/q_train_sample_80% -o q_train_imdb_join_07 -D 0.7 -s 1 -n 70
```

## Query Shift (TPCH)

```bash
# train data 1
python qproc.py -t predicates  -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_0 -o q_train_predicates -D 0.5 -s 1 -n 100

# train data 2
python qproc.py -t joins  -I  ./neurbench/datasets/dynamic_datasets/tpch/q_train_0 -o q_train_join -D 0.5 -s 1 -n 100

# train data 3
python qproc.py -t tables  -I  ./neurbench/datasets/dynamic_datasets/tpch/q_train_0 -o q_train_table -D 0.5 -s 1 -n 100
```

Query shifts for the lero

Since Lero has its own acceptable training query set, we therefore shift based on that.

```bash
# train data 1
python qproc.py -t predicates  -i ./neurbench/datasets/dynamic_datasets/tpch/tpch_train_all.txt -o q_train_predicates_lero -D 0.5 -s 1 -n 100

# train data 2
python qproc.py -t joins  -i  ./neurbench/datasets/dynamic_datasets/tpch/tpch_train_all.txt -o q_train_join_lero -D 0.5 -s 1 -n 100

# train data 3
python qproc.py -t tables  -i  ./neurbench/datasets/dynamic_datasets/tpch/tpch_train_all.txt -o q_train_table_lero -D 0.5 -s 1 -n 100
```

```bash
python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_join_lero -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_join_lero_merge.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_predicates_lero -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_predicates_lero_merge.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_table_lero -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_table_lero_merge.txt
```

### Micro benchmark

#### For Lero

For lero d=0.1

```bash
# train data 1
python qproc.py -t predicates  -i ./neurbench/datasets/dynamic_datasets/tpch/tpch_train_all.txt -o q_train_predicates_lero_01 -D 0.1 -s 1 -n 100

# train data 2
python qproc.py -t joins  -i  ./neurbench/datasets/dynamic_datasets/tpch/tpch_train_all.txt -o q_train_join_lero_01 -D 0.1 -s 1 -n 100
```

```bash
python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_table_lero_01 -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_join_lero_merge_01.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_predicates_lero_01 -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_predicates_lero_merge_01.txt
```



For lero d=0.3

```bash
# train data 1
python qproc.py -t predicates  -i ./neurbench/datasets/dynamic_datasets/tpch/tpch_train_all.txt -o q_train_predicates_lero_03 -D 0.3 -s 1 -n 100

# train data 2
python qproc.py -t joins  -i  ./neurbench/datasets/dynamic_datasets/tpch/tpch_train_all.txt -o q_train_join_lero_03 -D 0.3 -s 1 -n 100
```

```bash
python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_join_lero_03 -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_join_lero_merge_03.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_predicates_lero_03 -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_predicates_lero_merge_03.txt
```



For lero d=0.7

```bash
# train data 1
python qproc.py -t predicates  -i ./neurbench/datasets/dynamic_datasets/tpch/tpch_train_all.txt -o q_train_predicates_lero_07 -D 0.7 -s 1 -n 100

# train data 2
python qproc.py -t joins  -i  ./neurbench/datasets/dynamic_datasets/tpch/tpch_train_all.txt -o q_train_join_lero_07 -D 0.7 -s 1 -n 100
```

```bash
python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_join_lero_07 -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_join_lero_merge_07.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_predicates_lero_01 -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_predicates_lero_merge_07.txt
```

#### For Bao

d=0.1

```bash
# train data 1
python qproc.py -t predicates  -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_0 -o q_train_predicates_01 -D 0.1 -s 1 -n 100

# train data 2
python qproc.py -t joins  -I  ./neurbench/datasets/dynamic_datasets/tpch/q_train_0 -o q_train_join_01 -D 0.1 -s 1 -n 100
```

d=0.3

```bash
# train data 1
python qproc.py -t predicates  -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_0 -o q_train_predicates_03 -D 0.3 -s 1 -n 100

# train data 2
python qproc.py -t joins  -I  ./neurbench/datasets/dynamic_datasets/tpch/q_train_0 -o q_train_join_03 -D 0.3 -s 1 -n 100
```

d=0.7

```bash
# train data 1
python qproc.py -t predicates  -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_0 -o q_train_predicates_07 -D 0.7 -s 1 -n 100

# train data 2
python qproc.py -t joins  -I  ./neurbench/datasets/dynamic_datasets/tpch/q_train_0 -o q_train_join_07 -D 0.7 -s 1 -n 100
```

### Dist Visualize

TPCH

```bash
python qproc.py -t joins  -I ./neurbench/datasets/dynamic_datasets/tpch/q_test_0 -o q_train_join_vis -D 0.5 -s 1 -n 100

python qproc.py -t predicates  -I  ./neurbench/datasets/dynamic_datasets/tpch/q_test_0 -o q_train_predicates_vis -D 0.5 -s 1 -n 100

python qproc.py -t tables  -I  ./neurbench/datasets/dynamic_datasets/tpch/q_test_0 -o q_train_tables_vis -D 0.5 -s 1 -n 100
```

IMDB

```bash
python qproc.py -t joins  -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o q_train_joins_05_vis -D 0.5 -s 1 -n 100

python qproc.py -t predicates  -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o q_train_predicates_05_vis -D 0.5 -s 1 -n 100

python qproc.py -t tables  -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o q_train_tables_05_vis -D 0.5 -s 1 -n 100

# factor with 0.1
python qproc.py -t joins  -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o q_train_joins_01_vis -D 0.1 -s 1 -n 100

python qproc.py -t predicates  -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o q_train_predicates_01_vis -D 0.1 -s 1 -n 100

python qproc.py -t tables  -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o q_train_tables_01_vis -D 0.1 -s 1 -n 100
```



# PostgreSQL config

## PG12 Local Test

```bash
docker run --name postgres12 -e POSTGRES_PASSWORD=123 -d --shm-size=4g -p 5432:5432 -v ./neurbench:/code postgres:12


# 3. test & check version
docker exec -it  postgres12 bash
psql -U postgres
SELECT version();
```

## Install PG12

```bash
docker pull postgres:12
docker volume create postgres12-data

# 1. run
docker run --name postgres12 \
-e POSTGRES_PASSWORD=123  \
-v postgres12-data:/var/lib/postgresql/data  \
-v .//workloads/datasets:/datasets \
-v ./neurbench:/code/neurbench \
-p 5432:5432 \
--shm-size=4g \
-d postgres:12
```

## Install PG16

```bash
docker pull postgres:16.3
docker volume create postgres-data

# 1. run
docker run --name postgres16 \
-e POSTGRES_PASSWORD=123  \
-v postgres-data:/var/lib/postgresql/data  \
-v .//workloads/datasets:/datasets \
-v ./neurbench:/code/neurbench \
-p 5432:5432 \
--shm-size=4g \
-d postgres:16.3
```

## Install pkgs

```bash
docker exec -it  postgres12 bash
docker exec -it  postgres16 bash

# install related things
apt-get update
apt-get install -y vim
apt-get install -y curl
apt-get install -y wget
apt-get install -y bc

# install python
apt-get install -y python3 python3-pip

# install gcc
apt-get install build-essential

# install based on versions
apt-get install postgresql-client-12
apt-get install postgresql-client-16

apt-get install postgresql-server-dev-12 postgresql-common
apt-get install postgresql-server-dev-16 postgresql-common
```

Install conda inside docker container

```bash
# enter
docker exec -it  postgres12 bash
docker exec -it  postgres16 bash
# install
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /root/miniconda3
export PATH=/root/miniconda3/bin:$PATH
rm /tmp/miniconda.sh
/root/miniconda3/bin/conda init
```

## Update DB cfgs

```
# config databse
docker exec -it postgres12 bash

# set shared_buffers = 2GB
vi /var/lib/postgresql/data/postgresql.conf

# restart the container
docker restart postgres12
```

## Test pgsql

```bash
# 3. test & check pg
docker exec -it  postgres12 bash
docker exec -it  postgres16 bash
psql -U postgres
SELECT version();
```



## psqls

```bash
# show all dbs
\l
# use one db
\c tpch

# show all tables
\d

# check index
\di
```



# Data Introduction

| DataSet Name | Desp                                                         |
| :----------- | ------------------------------------------------------------ |
| TPCH         |                                                              |
| TPCH Skew    |                                                              |
| JCC-H        | introduces join-crossing-correlations and skews into TPC-H   |
| CH Benchmark | combines TPC-H and TPC-C to evaluate database systems on hybrid transaction and analytical (HTAP) workloads |
| DSB          |                                                              |
| TPC-DS       |                                                              |
| IMDB+JOB     |                                                              |
| STACK        |                                                              |
|              |                                                              |



# Data source

```bash
TPCDS:https://www.tpc.org/tpcds/
TPC-H:https://www.tpc.org/tpch/  https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp
STATS:https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
IMDB: https://github.com/gregrahn/join-order-benchmark
```

## 1. IMDB dataset

### Download

1. download SQL templates `https://github.com/gregrahn/join-order-benchmark`

2. download datasets

   ```bash
   wget -c http://homepages.cwi.nl/~boncz/job/imdb.tgz && tar -xvzf imdb.tgz
   ```

The original Step-by-step instructions are not working.

2. Or just bao's datasets
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2QYZBT

### Load via script

```bash
# cd /code/neurbench/script/load_to_db/imdb
bash load_job_postgres.sh /code/neurbench/datasets/origin_datasets/imdb/imdb
# if in docker,
bash load_job_postgres.sh /data/datasets/imdb_data/imdb imdb_ori
bash load_job_postgres.sh /datasets/imdb_data/imdb imdb_ori
# if in Lero docker,
bash load_job_postgres.sh /datasets/imdb/imdb imdb_ori
```

### Load via restore

```bash
# in host
docker cp /Users/kevin/Downloads/ai4db_dataset/imdb_pg11 postgres-container:/dataset
psql -U postgres
```

```bash
# in docker bash.
psql -U postgres -c "DROP DATABASE IF EXISTS imdb_ori"
psql -U postgres -c "CREATE DATABASE imdb_ori"
```

```bash
pg_restore -U postgres -d imdb_ori -v imdb_pg11
```

Use dataset

```sql
\c imdb
EXPLAIN SELECT count(*) FROM title;
```

Create Indexs

```bash
psql imdb_ori -f "fkindexes.sql"
psql imdb_ori -f "add_fks.sql"
psql imdb_ori -c "ANALYZE VERBOSE;"
```

### Load via SQLs

Original generated dataset, e.g., data_01 etc has type mismatch, thus requires post-processing.

Add header to each csv file, update in place.

```bash
cd script/load_to_db/imdb

python3 prepend_imdb_headers.py --csv_dir imdb

python3 prepend_imdb_headers.py --csv_dir data_01
python3 prepend_imdb_headers.py --csv_dir data_03
python3 prepend_imdb_headers.py --csv_dir data_05
python3 prepend_imdb_headers.py --csv_dir data_07
```

Create DB & Load & Build

```bash
psql -U postgres -c "DROP DATABASE IF EXISTS imdb_01"

./load_job_postgres.sh /data/datasets/imdb_data/imdb imdb_ori
./load_job_postgres.sh /data/datasets/imdb_data/data_01 imdb1
./load_job_postgres.sh /data/datasets/imdb_data/data_03 imdb3
./load_job_postgres.sh /data/datasets/imdb_data/data_05 imdb5
./load_job_postgres.sh /data/datasets/imdb_data/data_07 imdb7
```

For v2, we need to do the postprocessing

```bash
# postprocessing, cd .//neurbench/datasets/imdb_v2
./convert_all.sh ./data_01 ./data_01_postprocess
./convert_all.sh ./data_03 ./data_03_postprocess
./convert_all.sh ./data_05 ./data_05_postprocess
./convert_all.sh ./data_07 ./data_07_postprocess


grep "Dunn uploads a file from an Apple Powerbook in " ./data_01/movie_info.csv
grep "Dunn uploads a file from an Apple Powerbook in " ../imdb/imdb/movie_info.csv

grep "Pelham Street, Nottingham, Nottinghamshire, Engla" ./data_01/movie_info.csv
grep "Pelham Street, Nottingham, Nottinghamshire, Engla" ../imdb/imdb/movie_info.csv

grep  "who wrote the story for this film" ./data_01/movie_info.csv
grep  "who wrote the story for this film" ../imdb/imdb/movie_info.csv

grep  "The Rutting Season" ./data_01/movie_info.csv
grep  "The Rutting Season" ../imdb/imdb/movie_info.csv


# process only one
python3 postprocess.py ./data_01/person_info.csv ./data_01_postprocess/title.csv title

# load test
createdb imdb_01v2
psql imdb_01v2 -f "./schema.sql"
psql imdb_01v2 -f "./fkindexes.sql"
psql imdb_01v2 -f "./add_fks.sql"
psql imdb_01v2 -c "ANALYZE VERBOSE;"


# Then, we load to the DB, those are v2
./load_job_postgres.sh /data/datasets/imdb_v2/data_01_postprocess/ imdb_01v2
./load_job_postgres.sh /data/datasets/imdb_v2/data_03_postprocess/ imdb_03v2
./load_job_postgres.sh /data/datasets/imdb_v2/data_05_postprocess/ imdb_05v2
./load_job_postgres.sh /data/datasets/imdb_v2/data_07_postprocess/ imdb_07v2
```

### Query

Mainly use the join-order-benchmark

## 2. TPCH dataset

### Download

1. Follow doc https://dev.mysql.com/doc/heatwave/en/mys-hw-tpch-sample-data.html

   https://gist.github.com/yunpengn/6220ffc1b69cee5c861d93754e759d08

   Note: this should be run in Linux, macOS cannot run those.

   ```bash
   cd dbgen && mkdir gdata/tables
   DSS_PATH=./gdata/tables ./dbgen -s 1

   # Now, you should see a few `XXX.tbl` files. However, a bug in dbgen generates an extra `|` at the end of each line. To fix it, run the following command:
   for i in `ls *.tbl`; do sed 's/|$//' $i > ${i/tbl/csv}; echo $i; done;
   ```


### Load

```bash
psql -U postgres -c "DROP DATABASE IF EXISTS tpch"
psql -U postgres -c "CREATE DATABASE tpch"

# Craete tables
psql -U postgres -d tpch -f dss.ddl
```

Database generate scripts

Note the default `dss.ddl` don't include indexs and `dss.ri` has the wrong syntax. `psql -U postgres -d tpch -f dss.ri` are not working.
So modify the `dss.ri` with this those:

```sql
--ALTER TABLE REGION DROP PRIMARY KEY;
--ALTER TABLE NATION DROP PRIMARY KEY;
--ALTER TABLE PART DROP PRIMARY KEY;
--ALTER TABLE SUPPLIER DROP PRIMARY KEY;
--ALTER TABLE PARTSUPP DROP PRIMARY KEY;
--ALTER TABLE ORDERS DROP PRIMARY KEY;
--ALTER TABLE LINEITEM DROP PRIMARY KEY;
--ALTER TABLE CUSTOMER DROP PRIMARY KEY;

-- For table REGION
ALTER TABLE REGION
ADD PRIMARY KEY (R_REGIONKEY);

-- For table NATION
ALTER TABLE NATION
ADD PRIMARY KEY (N_NATIONKEY);

ALTER TABLE NATION
ADD FOREIGN KEY (N_REGIONKEY) REFERENCES REGION (R_REGIONKEY);

COMMIT;

-- For table PART
ALTER TABLE PART
ADD PRIMARY KEY (P_PARTKEY);

COMMIT;

-- For table SUPPLIER
ALTER TABLE SUPPLIER
ADD PRIMARY KEY (S_SUPPKEY);

ALTER TABLE SUPPLIER
ADD FOREIGN KEY (S_NATIONKEY) REFERENCES NATION (N_NATIONKEY);

COMMIT;

-- For table PARTSUPP
ALTER TABLE PARTSUPP
ADD PRIMARY KEY (PS_PARTKEY, PS_SUPPKEY);

COMMIT;

-- For table CUSTOMER
ALTER TABLE CUSTOMER
ADD PRIMARY KEY (C_CUSTKEY);

ALTER TABLE CUSTOMER
ADD FOREIGN KEY (C_NATIONKEY) REFERENCES NATION (N_NATIONKEY);

COMMIT;

-- For table LINEITEM
ALTER TABLE LINEITEM
ADD PRIMARY KEY (L_ORDERKEY, L_LINENUMBER);

COMMIT;

-- For table ORDERS
ALTER TABLE ORDERS
ADD PRIMARY KEY (O_ORDERKEY);

COMMIT;

-- For table PARTSUPP
ALTER TABLE PARTSUPP
ADD FOREIGN KEY (PS_SUPPKEY) REFERENCES SUPPLIER (S_SUPPKEY);

COMMIT;

ALTER TABLE PARTSUPP
ADD FOREIGN KEY (PS_PARTKEY) REFERENCES PART (P_PARTKEY);

COMMIT;

-- For table ORDERS
ALTER TABLE ORDERS
ADD FOREIGN KEY (O_CUSTKEY) REFERENCES CUSTOMER (C_CUSTKEY);

COMMIT;

-- For table LINEITEM
ALTER TABLE LINEITEM
ADD FOREIGN KEY (L_ORDERKEY) REFERENCES ORDERS (O_ORDERKEY);

COMMIT;

ALTER TABLE LINEITEM
ADD FOREIGN KEY (L_PARTKEY, L_SUPPKEY) REFERENCES PARTSUPP (PS_PARTKEY, PS_SUPPKEY);

COMMIT;
```

or if we don;t wanna primary key but only indexs

```sql
-- For table REGION
CREATE INDEX idx_region_regionkey ON REGION (R_REGIONKEY);

-- For table NATION
CREATE INDEX idx_nation_nationkey ON NATION (N_NATIONKEY);
CREATE INDEX idx_nation_regionkey ON NATION (N_REGIONKEY);

-- For table PART
CREATE INDEX idx_part_partkey ON PART (P_PARTKEY);

-- For table SUPPLIER
CREATE INDEX idx_supplier_suppkey ON SUPPLIER (S_SUPPKEY);
CREATE INDEX idx_supplier_nationkey ON SUPPLIER (S_NATIONKEY);

-- For table PARTSUPP
CREATE INDEX idx_partsupp_partkey_suppkey ON PARTSUPP (PS_PARTKEY, PS_SUPPKEY);

-- For table CUSTOMER
CREATE INDEX idx_customer_custkey ON CUSTOMER (C_CUSTKEY);
CREATE INDEX idx_customer_nationkey ON CUSTOMER (C_NATIONKEY);

-- For table LINEITEM
CREATE INDEX idx_lineitem_orderkey_linenumber ON LINEITEM (L_ORDERKEY, L_LINENUMBER);
CREATE INDEX idx_lineitem_partkey_suppkey ON LINEITEM (L_PARTKEY, L_SUPPKEY);

-- For table ORDERS
CREATE INDEX idx_orders_orderkey ON ORDERS (O_ORDERKEY);
CREATE INDEX idx_orders_custkey ON ORDERS (O_CUSTKEY);

-- For table PARTSUPP (additional indexes for foreign keys)
CREATE INDEX idx_partsupp_suppkey ON PARTSUPP (PS_SUPPKEY);
CREATE INDEX idx_partsupp_partkey ON PARTSUPP (PS_PARTKEY);

-- For table LINEITEM (additional indexes for foreign keys)
CREATE INDEX idx_lineitem_orderkey ON LINEITEM (L_ORDERKEY);
```

Now execut it.

```bash
# Add primary forigne keys,
psql -U postgres -d tpch -f dss.ri
```

```bash
# Loads data.
cd gdata/tables
psql -U postgres -d tpch -c "\COPY region FROM '/data/datasets/data_0/region.tbl' WITH (DELIMITER '|')";
psql -U postgres -d tpch -c "\COPY nation FROM '/data/datasets/data_0/nation.tbl' WITH (DELIMITER '|')";
psql -U postgres -d tpch -c "\COPY customer FROM '/data/datasets/data_0/customer.tbl' WITH (DELIMITER '|')";
psql -U postgres -d tpch -c "\COPY supplier FROM '/data/datasets/data_0/supplier.tbl' WITH (DELIMITER '|')";
psql -U postgres -d tpch -c "\COPY part FROM '/data/datasets/data_0/part.tbl' WITH (DELIMITER '|')";
psql -U postgres -d tpch -c "\COPY partsupp FROM '/data/datasets/data_0/partsupp.tbl' WITH (DELIMITER '|')";
psql -U postgres -d tpch -c "\COPY orders FROM '/data/datasets/data_0/orders.tbl' WITH (DELIMITER '|')";
psql -U postgres -d tpch -c "\COPY lineitem FROM '/data/datasets/data_0/lineitem.tbl' WITH (DELIMITER '|')";
```

### Query

Generate by default.

```bash
# training data generation
for ((i=1;i<=22;i++)); do
  DSS_QUERY=./queries ./qgen -v -d -s 1 -r 42 ${i} > .//neurbench/datasets/q_train_0/${i}_0.sql
done

for ((j=1;j<=7;j++)); do
  for ((i=1;i<=22;i++)); do
    if [ $i -eq 15 ]; then
      continue  # Skip iteration when i equals 15
    fi
    echo "DSS_QUERY=./queries ./qgen -v -s 1 -r $((100*${j})) ${i} > .//neurbench/datasets/q_train_0/${i}_${j}.sql"

    # Fix: use $ for arithmetic evaluation
    DSS_QUERY=./queries ./qgen -v -s 1 -r $((100*${j})) ${i} > .//neurbench/datasets/q_train_0/${i}_${j}.sql
  done
done



# testing data generation
for ((i=1;i<=22;i++)); do
  DSS_QUERY=./queries ./qgen -v -s 1 -r 2024 ${i} > .//neurbench/datasets/q_test_0/${i}.sql
done
```

Or

The default queries folder in the `TPC-H V3.0.1/dbgen` cannot execute.

So we use the template provided in https://github.com/tvondra/pg_tpch

```bash
# here it generates many .sql and .explain.sql, and their results
for q in `seq 1 22`
do
    DSS_QUERY=gdata/templates ./qgen $q >> gdata/queries/$q.sql
    sed 's/^select/explain select/' gdata/queries/$q.sql > gdata/queries/$q.explain.sql
    cat gdata/queries/$q.sql >> gdata/queries/$q.explain.sql;
done
```

Execute sql

```bash
psql -U postgres -d tpch -f /data/datasets/q_0/1.sql

# execute
for i in $(seq 1 22); do
  psql -U postgres -d tpch -f "/data/datasets/q_train_0/${i}.sql"
done

# execute
for i in $(seq 1 22); do
  psql -U postgres -d tpch -f "/data/datasets/tpch_query/q_test_0/${i}.sql"
done
```

Transfer query format for benchmarking

```
cd baseline/neurbench
```

For data shifts

```bash
python tbl2csv.py -i ./neurbench/datasets/dynamic_datasets/tpch/data_0 -o ./neurbench/datasets/dynamic_datasets/tpch/data_0_csv
```

```bash
python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_0 -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_0_merge.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_test_0 -o ./neurbench/datasets/dynamic_datasets/tpch/q_test_0_merge.txt
```

```bash
python qpre.py -i ./neurbench/datasets/dynamic_datasets/tpch/dss.ddl -o ./neurbench/datasets/dynamic_datasets/tpch/dss.sql
```

For query shifts

```bash
python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_join -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_join_merge.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_predicates -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_predicates_merge.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/tpch/q_train_table -o ./neurbench/datasets/dynamic_datasets/tpch/q_train_table_merge.txt
```

Parse sqls

```bash
python parse_sql_metadata.py -i ./testdata/tpch-pp.sql -o ./result
```

## 3. STACK dataset

### Download

```bash
# data for PG13
wget https://www.dropbox.com/s/55bxfhilcu19i33/so_pg13
# data for PG12
wget https://www.dropbox.com/s/98u5ec6yb365913/so_pg12

# query
wget https://rmarcus.info/so_queries.tar.zst
sudo apt-get install zstd tar
unzstd so_queries.tar.zst
tar -xvf so_queries.tar
```

### Load

```bash
psql -U postgres -c "DROP DATABASE IF EXISTS stack"
psql -U postgres -c "CREATE DATABASE stack"
```

```bash
pg_restore -U postgres -d stack -v so_pg13
```

## 4. STATS

### Download

```
https://drive.google.com/file/d/18V9MhlN_5PmFOhKdzUcAIIOXYc1sYtvK/view?usp=sharing
```

### Load

In docker

1. create db

```sql
create database stats;
```

2. create tables

```sql
\i /code/baseline/BaoForPostgreSQL/exps_datasets/stats/create_table.sql
```

3. analyze tables

```sql
\i /code/baseline/BaoForPostgreSQL/exps_datasets/stats/analyze_table.sql
```

4. Load datasets

Run the copy cmds as in `worklaod.sql` file

```sql
COPY votes FROM '/code/datasets/processed_datasets/stats/ins_heavy/init_data/votes.csv' CSV header;
COPY comments FROM '/code/datasets/processed_datasets/stats/ins_heavy/init_data/comments.csv' CSV header;
COPY badges FROM '/code/datasets/processed_datasets/stats/ins_heavy/init_data/badges.csv' CSV header;
COPY posts FROM '/code/datasets/processed_datasets/stats/ins_heavy/init_data/posts.csv' CSV header;
COPY tags FROM '/code/datasets/processed_datasets/stats/ins_heavy/init_data/tags.csv' CSV header;
COPY posthistory FROM '/code/datasets/processed_datasets/stats/ins_heavy/init_data/posthistory.csv' CSV header;
COPY users FROM '/code/datasets/processed_datasets/stats/ins_heavy/init_data/users.csv' CSV header;
COPY postlinks FROM '/code/datasets/processed_datasets/stats/ins_heavy/init_data/postlinks.csv' CSV header;
```

5. Check

```sql
SELECT
    relname AS tablename,
    n_live_tup AS row_count
FROM pg_stat_user_tables
WHERE schemaname = 'public'
AND relname IN ('badges', 'comments', 'posthistory', 'postlinks', 'posts', 'tags', 'users', 'votes')
ORDER BY row_count DESC;
```

## 5. DVD Rental

### Download

### Load

```bash
# in host
docker cp dvdrental.tar postgres-container:/dataset
psql -U postgres
```

```shell
# inside psql
create database dvdrental
```

```bash
# in host
pg_restore -U postgres -d dvdrental /dataset/dvdrental.tar
```



# Baseline configs

## PG4IMDB

Run pg for imdb

Baselines

```bash
python qpre.py -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o ./neurbench/datasets/dynamic_datasets/imdb/imdb_q_test_0.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 -o ./neurbench/datasets/dynamic_datasets/imdb/imdb_q_test_0.txt
```

Run

```bash
docker exec -it pg_balsa bash
python3 -m venv myenv
source myenv/bin/activate

# run
python3 run_pg_only.py --dbname imdb_ori
```

## Bao

### Deploy based on the LQO framework

Following the doc at https://github.com/NLGithubWP/lqo_ml_perspective/blob/6651cd356036c3292c219b27cb4ff91732c8d260/docker/postgres_bao/README.md

Must be use PostgreSQL 12, PostgreSQL 16/13 cannot be used here, since it don't have **utils/relfilenodemap.h** after installing below packages.

### Deploy based on the origin repo

Deploy Extension

```bash
# Inside docker, go to /code/neurbench
docker exec -it postgres12 bash
cd /code/neurbench/pg_extension
make USE_PGXS=1 install
echo "shared_preload_libraries = 'pg_bao'" >> /var/lib/postgresql/data/postgresql.conf

# Restart the container
docker restart postgres12
```

Verify extension is installed successfully.

```sql
# inside the docker
docker exec -it postgres12 bash
psql -U postgres
# in psql
SHOW enable_bao;
```

Deploy Server on host

```bash
. activate_env.sh
conda create -n ai4db python=3.8
conda activate ai4db
pip3 install scikit-learn numpy jobliby
pip3 install torch==1.5.0
pip install psycopg2-binary

# run bao server
# taskset -c 5-8 env CUDA_VISIBLE_DEVICES=1
cd bao_server
CUDA_VISIBLE_DEVICES=1 python3 main.py

CUDA_VISIBLE_DEVICES="" python3 main.py
```

Verify the server is running and can receive requests from the BAO extension.

```sql
# inside the docker
docker exec -it postgres12 bash
psql -U postgres
# run those
\c imdb;
SET enable_bao TO on;
SELECT count(*) FROM users;
EXPLAIN SELECT count(*) FROM title;
```

Dev BAO

If updating the extension C code, do the following.

```bash
cd /code/BaoForPostgreSQL/pg_extension
make USE_PGXS=1 install

# restart the container
docker restart postgres12
docker exec -it  postgres-container bash
export PATH=/usr/lib/postgresql/12/bin:$PATH
# then run the verify above
```

Clear

Remove the trained model

```bash
cd ./BaoForPostgreSQL/bao_server
rm -rf bao_default_model
```

### Data shifts IMDB_v2

1. Run baseline: postgresql on datasets + query

```bash
# execute the script
python3 run_test_queries.py --use_postgres --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb_data_shift_v2/test__postgres_01.txt --database_name imdb_01v2

python3 run_test_queries.py --use_postgres --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb_data_shift_v2/test__postgres_05.txt --database_name imdb_05v2

python3 run_test_queries.py --use_postgres --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb_data_shift_v2/test__postgres_07.txt --database_name imdb_07v2

# ori datasets
python3 run_test_queries.py --use_postgres --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb_data_shift_v2/test__postgres_00.txt --database_name imdb_ori
```

2. Test trained PG on the new workloads.

```bash
chmod +x ./schedule_datashift_v2.sh
nohup ./schedule_datashift_v2.sh > schedule_datashift_v2_output.log 2>&1 &
nohup bash imdb_v2_load_schedule_lero.sh > imdb_v2_load_sched.log 2>&1 &
```



```bash
# Training Bao on data0+query1, join query
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_1/joins --output_file log_sigmod/train__bao__base_query_join_imdb.txt --database_name imdb_01v2
# Testing Bao data01v2 +query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb_data_shift_v2/test__bao__query_data_v2_01.txt --database_name imdb_01v2

# Training Bao on data0+query1, predicated query
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_1/predicates --output_file log_sigmod/train__bao__base_query_predicates_imdb.txt --database_name imdb_ori
# Testing Bao data05v2 +query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb_data_shift_v2/test__bao__query_data_v2_05.txt --database_name imdb_05v2

# Training Bao on data0+query1, table query
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_1/tables --output_file log_sigmod/train__bao__base_query_table_imdb.txt --database_name imdb_ori
# Testing Bao data07v2+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb_data_shift_v2/test__bao__query_data_v2_07.txt --database_name imdb_07v2


python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_vldb_data_shift_v2/test__bao__query_data_v2_00.txt --database_name imdb_ori
```

### Data shifts TPCH

1. run postgresql

   ```bash
   # execute the script
   python run_pg_only.py --dbname tpch
   python run_pg_only.py --dbname tpch1
   python run_pg_only.py --dbname tpch2
   python run_pg_only.py --dbname tpch3
   ```



```bash
# run in the data_1 + q_test_0
python3 run_test_queries.py --use_postgres --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__postgres__data0_round5.txt --database_name tpch
```

```bash
# run in the data_1 + q_test_0
python3 run_test_queries.py --use_postgres --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__postgres__data1.txt --database_name tpch1
```

```bash
# run in the data_2 + q_test_0
python3 run_test_queries.py --use_postgres --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__postgres__data2.txt --database_name tpch2
```

```bash
# run in the data_1 + q_test_0
python3 run_test_queries.py --use_postgres --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__postgres__data3.txt --database_name tpch3
```

2. Run bao

```bash
# Training Bao on data0+query0
python3 run_queries.py --query_dir /data/datasets/q_train_0 --output_file log_sigmod/train__bao__base_data0.txt --database_name tpch

# Testing Bao on data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__data0.txt --database_name tpch

# Testing Bao on data1+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__data1.txt --database_name tpch1
```

```bash
# Training Bao data1+query0
python3 run_queries.py --query_dir /data/datasets/q_train_0 --output_file log_sigmod/train__bao__base_data1.txt --database_name tpch1

# Testing Bao data2+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__data2.txt --database_name tpch2
```

```bash
# Training Bao data1+query0
python3 run_queries.py --query_dir /data/datasets/q_train_0 --output_file log_sigmod/train__bao__base_data2.txt --database_name tpch2

# Testing Bao data2+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__data3.txt --database_name tpch3
```

### Query Shifts TPCH

d = 0.5

```bash
# Training Bao on data0+query1
python3 run_queries.py --query_dir /data/datasets/q_train_join --output_file log_sigmod/train__bao__base_query_join.txt --database_name tpch
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__query_join.txt --database_name tpch

# Training Bao on data0+query2
python3 run_queries.py --query_dir /data/datasets/tpch_query/q_train_predicates --output_file log_sigmod/train__bao__base_query_predicates.txt --database_name tpch
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__query_predicates.txt --database_name tpch

# Training Bao on data1+query3
python3 run_queries.py --query_dir /data/datasets/q_train_table --output_file log_sigmod/train__bao__base_query_table.txt --database_name tpch
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__query_tables.txt --database_name tpch
```

d = 0.1

```bash
# Training Bao on data0+query1
python3 run_queries.py --query_dir /data/datasets/q_train_join_01 --output_file log_sigmod/train__bao__base_query_join_01.txt --database_name tpch
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__query_join_01.txt --database_name tpch

# Training Bao on data0+query1
python3 run_queries.py --query_dir /data/datasets/tpch_query/q_train_predicates_01 --output_file log_sigmod/train__bao__base_query_predicates_01.txt --database_name tpch
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__query_predicates_01.txt --database_name tpch
```



d = 0.3

```bash
# Training Bao on data0+query1
python3 run_queries.py --query_dir /data/datasets/q_train_join_03 --output_file log_sigmod/train__bao__base_query_join_03.txt --database_name tpch
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__query_join_03.txt --database_name tpch

# Training Bao on data0+query2
python3 run_queries.py --query_dir /data/datasets/tpch_query/q_train_predicates_03 --output_file log_sigmod/train__bao__base_query_predicates_03.txt --database_name tpch
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__query_predicates_03.txt --database_name tpch
```



d = 0.7

```bash
# Training Bao on data0+query1
python3 run_queries.py --query_dir /data/datasets/q_train_join_07 --output_file log_sigmod/train__bao__base_query_join_07.txt --database_name tpch
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__query_join_07.txt --database_name tpch

# Training Bao on data0+query2
python3 run_queries.py --query_dir /data/datasets/tpch_query/q_train_predicates_07 --output_file log_sigmod/train__bao__base_query_predicates_07.txt --database_name tpch
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/tpch_query/q_test_0 --output_file log_sigmod/test__bao__query_predicates_07.txt --database_name tpch
```

Schedule to run

```bash
nohup ./schedule.sh > output.log 2>&1 &
```

### Query Shfits JOB

start the server

```bash
. activate_env.sh
python3 main.py
```

d = 0.1 with imdb

```bash
# Training Bao on data0+query1, join query
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_1/joins --output_file log_sigmod/train__bao__base_query_join_imdb.txt --database_name imdb_ori
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_sigmod/test__bao__query_join_imdb.txt --database_name imdb_ori

# Training Bao on data0+query1, predicated query
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_1/predicates --output_file log_sigmod/train__bao__base_query_predicates_imdb.txt --database_name imdb_ori
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_sigmod/test__bao__query_predicates_imdb.txt --database_name imdb_ori

# Training Bao on data0+query1, table query
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_1/tables --output_file log_sigmod/train__bao__base_query_table_imdb.txt --database_name imdb_ori
# Testing Bao data0+query0
python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/ori_queries --output_file log_sigmod/test__bao__query_table_imdb.txt --database_name imdb_ori
```

The following is running via script

```bash
chmod +x ./schedule_queryshift_v2.sh
nohup ./schedule_queryshift_v2.sh > schedule_queryshift_v2.log 2>&1 &
```



d = 0.3 with imdb

```bash
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_imdb_join_03 --output_file log_vldb_query_shifts/train__bao__query_shift_imdb_join_03.txt --database_name imdb_ori

python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/q_test_sample_20% --output_file log_vldb_query_shifts/test__bao__query_shift_imdb_join_03.txt --database_name imdb_ori
```

d = 0.5 with imdb

```bash
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_imdb_join_05 --output_file log_vldb_query_shifts/train__bao__query_shift_imdb_join_05.txt --database_name imdb_ori

python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/q_test_sample_20% --output_file log_vldb_query_shifts/test__bao__query_shift_imdb_join_05.txt --database_name imdb_ori
```

d = 0.7 with imdb

```bash
python3 run_queries.py --query_dir /data/datasets/imdb_queries/q_train_imdb_join_07 --output_file log_vldb_query_shifts/train__bao__query_shift_imdb_join_07.txt --database_name imdb_ori

python3 run_test_queries.py --use_bao --query_dir /data/datasets/imdb_queries/q_test_sample_20% --output_file log_vldb_query_shifts/test__bao__query_shift_imdb_join_07.txt --database_name imdb_ori
```



## Env Config for others

As in the repo of lqo_benchmark.

after building docker image

1. Start the service docker

```bash
# in panda15, hdd1
docker run --gpus all -d \
  --name balsa \
  --net postgresbalsa_network \
  --ip 10.6.0.4 \
  -v .//neurbench/datasets:/data/datasets \
  -v ./:/app/baseline \
  -e POSTGRES_DB=imdbload \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 9381:9381 \
  --shm-size="10gb" \
  balsa_img

# in panda16, hdd2
docker run --gpus all -d \
  --name balsa \
  --net postgresbalsa_network \
  --ip 10.6.0.4 \
  -v .//neurbench/datasets:/data/datasets \
  -v ./neurbench/baseline/:/app/baseline \
  -e POSTGRES_DB=imdbload \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 9381:9381 \
  --shm-size="10gb" \
  balsa_img
```

Config the environment for the service docker

```bash
# Start interactive shell in the Balsa container

docker exec -it balsa bash

cd /app/baseline/lqo_ml_perspective/balsa
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

2. Start the PG docker

```bash
# Start the container(s), if you have not already done so in the previous step
docker-compose up -d
```

## LEON

Test

```bash
cd /app/baseline/lqo_ml_perspective/leon
export PYTHONPATH=$PYTHONPATH:$(pwd)/util


nohup python3 -u train_job.py --experiment job_join > ./run_backend_log.txt 2>&1 &

nohup python3 -u train_job.py --experiment job_join > ./run_backend_log.txt 2>&1 &
# or
python3 train_job.py --experiment job_join
python3 train_job.py --experiment job_predicate
python3 train_job.py --experiment job_table
```

evaluate the model

```bash
python3 test_job.py --log_file_path job_join.csv --experiment job_join --model_path job_join --logs_name job_join
python3 test_job.py --log_file_path job_predicate.csv --experiment job_predicate --model_path job_predicate --logs_name job_predicate
python3 test_job.py --log_file_path job_table.csv --experiment job_table --model_path job_table --logs_name job_table
```

## LQO Benchmarking

Stack datasets:

- original there are **6191** queries across 16 base queries.

- it down-samples to **14 base queries** with **8** randomly sampled variations, each.

Convert `.ipynb` into python script.

```bash
jupyter nbconvert --to script notebooks/SplitSTACKQueries.ipynb
```

README sequences:

```bash
# download dataset
https://github.com/NLGithubWP/lqo_ml_perspective/blob/6651cd356036c3292c219b27cb4ff91732c8d260/data/README.md

# run docker
https://github.com/NLGithubWP/lqo_ml_perspective/blob/6651cd356036c3292c219b27cb4ff91732c8d260/docker/postgres_bao/README.md
```

Analyze result

1. Each query execute 3 times, only pick third time as final result.
2. Train and test are using various datasets for all systems.

## Lero

### Change Query format

```bash
# this is for the database shift only
python qpre.py -I ./neurbench/datasets/dynamic_datasets/imdb/q_test_sample_20% -o ./neurbench/datasets/dynamic_datasets/imdb/imdb_q_test_small_set.txt
python qpre.py -I ./neurbench/datasets/dynamic_datasets/imdb/q_train_sample_80% -o ./neurbench/datasets/dynamic_datasets/imdb/imdb_q_train_small_set.txt

# join with various shifts
python qpre.py -I ./neurbench/datasets/dynamic_datasets/imdb/q_train_1/joins -o ./neurbench/datasets/dynamic_datasets/imdb/q_train_queryshift_join_01.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/imdb/q_train_imdb_join_03 -o ./neurbench/datasets/dynamic_datasets/imdb/q_train_queryshift_join_03.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/imdb/q_train_imdb_join_05 -o ./neurbench/datasets/dynamic_datasets/imdb/q_train_queryshift_join_05.txt

# predict/table for d=0.1
python qpre.py -I ./neurbench/datasets/dynamic_datasets/imdb/q_train_imdb_table_01 -o ./neurbench/datasets/dynamic_datasets/imdb/q_train_queryshift_table_01.txt

python qpre.py -I ./neurbench/datasets/dynamic_datasets/imdb/q_train_imdb_predicates_01 -o ./neurbench/datasets/dynamic_datasets/imdb/q_train_queryshift_predicates_01.txt
```

### Download database

```bash
# 1. download the PostgreSQL 13.1  into the host
wget https://ftp.postgresql.org/pub/source/v13.1/postgresql-13.1.tar.bz2
tar -xvf postgresql-13.1.tar.bz2

cd ./
cp ./Lero/0001-init-lero.patch .
cd .//postgr esql-13.1
# 2. update code
git apply ../0001-init-lero.patch
```

### Config running env

```bash
cd ./Lero
# build and run the containers
docker build -t lero_img:latest .

docker run --name lero_docker \
-e POSTGRES_PASSWORD=123 \
-v .//neurbench:/code/neurbench \
-v .//postgresql-13.1:/code/postgresql-13.1 \
-v .//neurbench/datasets:/datasets/ \
-p 5432:5432 \
--shm-size="10gb" \
-d lero_img:latest
```

1. Install conda/python

Check the readme in main folder.

create envs

```bash
conda create -n ai4db python=3.8

source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai4db

pip3 install scikit-learn numpy jobliby
pip3 install torch==1.5.0
pip install psycopg2-binary
```

2. Compile the DB

```bash
docker exec -it lero_docker bash


# 1. install PostgreSQL
cd /code/postgresql-13.1
./configure
make
make install

# 2. create user etc
useradd -m -s /bin/bash postgres  # Create the postgres user
mkdir /usr/local/pgsql/data       # Create a directory for the data
chown postgres:postgres /usr/local/pgsql/data  # Change ownership to the postgres user

# 3. Initialize the Database Cluster
su - postgres
/usr/local/pgsql/bin/initdb -D /usr/local/pgsql/data

# 4. modify the configuration of PostgreSQL in postgresql.conf
cp /code/neurbench/baseline/Lero/postgresql.conf /usr/local/pgsql/data/postgresql.conf
cp /code/neurbench/baseline/Lero/pg_hba.conf /usr/local/pgsql/data/pg_hba.conf

# 5. start db
/usr/local/pgsql/bin/pg_ctl -D /usr/local/pgsql/data -l logfile start

# 6 add to sys path.
echo 'export PATH=/usr/local/pgsql/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

3. Test installation.

```bash
/usr/local/pgsql/bin/psql -h 0.0.0.0

# update password
/usr/local/pgsql/bin/psql
ALTER USER postgres PASSWORD '123';
```

### Run the server before train and inference.

Note, in the first time or running, change`ModelPath = ./reproduce/imdb_pw` in the `server.conf`.

Later we run the `train_mode.py` , which will call the subprocess to trigger the `train.py`, and this will save the model to somewhere.

Then we stop the server, change the `ModelPat` in the config, then re-run the server.
Then we perform `test.py`

```bash
cd /code/neurbench/baseline/Lero/lero
nohup python -u server.py > ./run_backend_log_for_test_table_01.txt 2>&1 &
```

### Query Shifts (IMDB)

Train

We run via scripts

```bash
chmod +x ./schedule_query_shift_imdb.sh
nohup ./schedule_query_shift_imdb.sh > schedule_query_shift_imdb.log 2>&1 &


chmod +x ./schedule_query_shift_imdb_table_predict.sh
nohup ./schedule_query_shift_imdb_table_predict.sh > schedule_query_shift_imdb_table_predict.log 2>&1 &
```



```bash
# query shift join
python train_model.py --query_path q_train_join_lero_merge.txt --test_query_path q_test_small_set.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_imdb_ori_train_queryshift_01.log --model_prefix imdb_ori_query_shift_model_01 --topK 3

# query shift join
python train_model.py --query_path q_train_join_lero_merge.txt --test_query_path q_test_small_set.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_imdb_ori_train_queryshift_03.log --model_prefix imdb_ori_query_shift_model_03 --topK 3

# query shift join
python train_model.py --query_path q_train_join_lero_merge.txt --test_query_path q_test_small_set.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_imdb_ori_train_queryshift_05.log --model_prefix imdb_ori_query_shift_model_05 --topK 3
```

Test

```bash
# join
rm -r log
ModelPath = imdb_ori_query_shift_model_01
python test.py --query_path imdb_queries/q_test_small_set.txt --output_query_latency_file q_test_imdb_query_shift_join_01.test

rm -r log
ModelPath = imdb_ori_query_shift_model_03
python test.py --query_path imdb_queries/q_test_small_set.txt --output_query_latency_file q_test_imdb_query_shift_join_03.test

rm -r log
ModelPath = imdb_ori_query_shift_model_05
python test.py --query_path imdb_queries/q_test_small_set.txt --output_query_latency_file q_test_imdb_query_shift_join_05.test
```

Other pattern shfits

```bash
rm -r log
ModelPath = imdb_ori_query_shift_model_predict_01_3
python test.py --query_path imdb_queries/q_test_small_set.txt --output_query_latency_file q_test_imdb_query_shift_predict_01.test


rm -r log
ModelPath = imdb_ori_query_shift_model_table_01_3
python test.py --query_path imdb_queries/q_test_small_set.txt --output_query_latency_file q_test_imdb_query_shift_table_01.test
```



### Data Shifts (IMDB)

```bash
# train model
python train_model.py --query_path q_train_small_set.txt --test_query_path q_test_small_set.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_imdb_ori_train.log --model_prefix imdb_ori_model --topK 3

# test model
# update config
DB = "imdb_ori"
rm -r log
python test.py --query_path imdb_queries/q_test_small_set.txt --output_query_latency_file q_test_small_set_imdb_ori.test

DB = "imdb_01v2"
rm -r log
python test.py --query_path imdb_queries/q_test_small_set.txt --output_query_latency_file q_test_small_set_imdb_01.test

DB = "imdb_05v2"
rm -r log
python test.py --query_path imdb_queries/q_test_small_set.txt --output_query_latency_file q_test_small_set_imdb_05.test

DB = "imdb_07v2"
rm -r log
python test.py --query_path imdb_queries/q_test_small_set.txt --output_query_latency_file q_test_small_set_imdb_07.test
```

### Data Shifts (TPCH)

Train

```bash
cd /code/neurbench/baseline/Lero/lero

# output_query_latency_file: the final executed plan will be output to this file
# model_prefix: prefix of model name
# topK: the number of plans that can be explored by each query

python train_model.py --query_path q_train_0_merge.txt --test_query_path q_test_0_merge.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch.log --model_prefix tpch_test_model --topK 3


# output_query_latency_file: the final executed plan will be output to this file
# model_prefix: prefix of model name
# topK: the number of plans that can be explored by each query
python train_model.py --query_path tpch_train.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch.log --model_prefix tpch_test_model --topK 3
```

Test

Update `test_script/config.py`, default PG results:

```bash
# execute the script
DB = "tpch"
rm -r log
python train_model.py --query_path q_test_0_merge.txt --algo pg --output_query_latency_file pg_tpch_test0.log
python run_pg_only.py --dbname tpch


DB = "tpch1"
rm -r log
python train_model.py --query_path q_test_0_merge.txt --algo pg --output_query_latency_file pg_tpch_test1.test
python run_pg_only.py --dbname tpch1

DB = "tpch2"
rm -r log
python train_model.py --query_path q_test_0_merge.txt --algo pg --output_query_latency_file pg_tpch_test2.test
python run_pg_only.py --dbname tpch2

DB = "tpch3"
rm -r log
python train_model.py --query_path q_test_0_merge.txt --algo pg --output_query_latency_file pg_tpch_test3.test
python run_pg_only.py --dbname tpch3
```

lero

```bash
# execute the script
DB = "tpch"
rm -r log
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_0_merge.test

DB = "tpch1"
rm -r log
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_1_merge.test

DB = "tpch2"
rm -r log
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_2_merge.test

DB = "tpch3"
rm -r log
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_3_merge.test
````

### Query Shifts (TPCH)

Train

```bash
# query shift join
python train_model.py --query_path q_train_join_lero_merge.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch_join.log --model_prefix tpch_test_model_join --topK 3

# query shift predicates
python train_model.py --query_path q_train_predicates_lero_merge.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch_predicates.log --model_prefix tpch_test_model_predicates --topK 3

# query shift table
python train_model.py --query_path q_train_table_lero_merge.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch_tables.log --model_prefix tpch_test_model_tables --topK 3
```

Test

```bash
# join
rm -r log
ModelPath = tpch_test_model_join_4
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_join_merge.test

# predicates
rm -r log
ModelPath = tpch_test_model_predicates_4
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_predicates_merge.test

# table
rm -r log
ModelPath = tpch_test_model_tables_4
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_table_merge.test
```

### Micro benchmark

#### For lero

Train and test d= 0.1

```bash
# query shift join
python train_model.py --query_path q_train_join_lero_merge_01.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch_join_01.log --model_prefix tpch_test_model_join_01 --topK 3

# query shift predicates
python train_model.py --query_path q_train_predicates_lero_merge_01.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch_predicates_01.log --model_prefix tpch_test_model_predicates_01 --topK 3


# join
rm -r log
ModelPath = tpch_test_model_join_01_4
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_join_merge_01.test

# predicates
rm -r log
ModelPath = tpch_test_model_predicates_01_4
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_predicates_merge_01.test
```

Train and test d=0.3

```bash
# query shift join
python train_model.py --query_path q_train_join_lero_merge_03.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch_join_03.log --model_prefix tpch_test_model_join_03 --topK 3

# query shift predicates
python train_model.py --query_path q_train_predicates_lero_merge_03.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch_predicates_03.log --model_prefix tpch_test_model_predicates_03 --topK 3


# join
rm -r log
ModelPath = tpch_test_model_join_03_4
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_join_merge_03.test

# predicates
rm -r log
ModelPath = tpch_test_model_predicates_03_4
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_predicates_merge_03.test
```

Train and test d=0.7

```bash
# query shift join
python train_model.py --query_path q_train_join_lero_merge_07.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch_join_07.log --model_prefix tpch_test_model_join_07 --topK 3

# query shift predicates
python train_model.py --query_path q_train_predicates_lero_merge_07.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch_predicates_07.log --model_prefix tpch_test_model_predicates_07 --topK 3


# join
rm -r log
ModelPath = tpch_test_model_join_07_4
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_join_merge_07.test

# predicates
rm -r log
ModelPath = tpch_test_model_predicates_07_4
python test.py --query_path q_test_0_merge.txt --output_query_latency_file q_test_predicates_merge_07.test
```

## HyperQO

Config envs

```bash
# in docker
pip install tqdm
pip install numpy==1.24.1
pip install psycopg2-binary
pip install psqlparse
```

Prepare the training input file, (fixed format)

```bash
python3 neurbench_workload_gene.py --input_folder ./neurbench/datasets/dynamic_datasets/imdb/q_train_1/joins --output_json query_join__train.json

python3 neurbench_workload_gene.py --input_folder ./neurbench/datasets/dynamic_datasets/imdb/q_train_1/predicates --output_json query_predicates__train.json

python3 neurbench_workload_gene.py --input_folder ./neurbench/datasets/dynamic_datasets/imdb/q_train_1/tables --output_json query_tables__train.json

# test
python3 neurbench_workload_gene.py --input_folder ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 --output_json query_join__test.json

python3 neurbench_workload_gene.py --input_folder ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 --output_json query_predicates__test.json

python3 neurbench_workload_gene.py --input_folder ./neurbench/datasets/dynamic_datasets/imdb/q_test_0 --output_json query_tables__test.json

python3 neurbench_workload_gene.py --input_folder ./neurbench/datasets/dynamic_datasets/imdb/q_train_imdb_join_03 --output_json query_sample_join_03__train.json

python3 neurbench_workload_gene.py --input_folder ./neurbench/datasets/dynamic_datasets/imdb/q_train_imdb_join_05 --output_json query_sample_join_05__train.json
```

Here we sample and divide the 108 query into train and test, we use leave one out sampling

```bash
python3 neurbench_workload_gene.py --input_folder ./neurbench/datasets/dynamic_datasets/imdb/q_train_sample_80% --output_json query_sample_train.json

python3 neurbench_workload_gene.py --input_folder ./neurbench/datasets/dynamic_datasets/imdb/q_test_sample_20% --output_json query_sample_test.json
```



Schedule to run

```bash
# in our panda server
conda activate balsa

chmod +x ./schedule_datashift_v2.sh
nohup ./schedule_datashift_v2.sh > schedule_datashift_v2_output.log 2>&1 &
nohup ./schedule_queryshift.sh > schedule_queryshift.log 2>&1 &

# this is the ori dist evaluation
nohup python run_mcts.py --query_file workload/query_join__train.json --train_database imdb_ori --test_database imdb_ori > data_v2_test_ori.log 2>&1 &
```

View logs at

```

```



## NEO

### Env Config

Env

```bash
pip install absl-py
pip install pytorch_lightning
pip install tensorboard
```

Env, this code treate the pg_executor as a lib

```bash
pip install -e pg_executor
```

Activate

```
conda activate balsa
```

### Debug

```bash
# train
python run.py --run Neo_JOB_DEBUG --local
python run.py --run Neo_JOB_DEBUG --local

# test
python3 test_model.py --run Neo_JOB_DEBUG --model_checkpoint  ./lqo_ml_perspective/balsa/wandb/offline-run-20250219_200200-1lotrfye/files/checkpoint.pt
```

View results

```
cd ./lqo_ml_perspective/balsa/logs
```

### Query Shifts(JOB)

Train with join shift with d = 03

```bash
# run jobs in backend
tmux new -s nr
tmux a
tmux + B , then D to quite

# train once and test multipe times. Run twice !!
rm data/initial_policy_data.pkl
python run.py --run Neo_DB2_SMALL_SET_JOIN_SHIFT_Train03 --local
python run.py --run Neo_DB2_SMALL_SET_JOIN_SHIFT_Train03 --local
# Note : choose (3) for the wandb
```

```bash
# train once and test multipe times. Run twice !!
rm data/initial_policy_data.pkl
python run.py --run Neo_DB2_SMALL_SET_JOIN_SHIFT_Train05 --local
python run.py --run Neo_DB2_SMALL_SET_JOIN_SHIFT_Train05 --local
# Note : choose (3) for the wandb
```

Test

```bash
python3 test_model.py --run Neo_DB2_SMALL_SET_JOIN_SHIFT_Train03 --model_checkpoint  ./lqo_ml_perspective/balsa/wandb/offline-run-20250225_221813-wqv11v1p/files/checkpoint.pt

python3 test_model.py --run Neo_DB2_SMALL_SET_JOIN_SHIFT_Train05 --model_checkpoint ./lqo_ml_perspective/balsa/wandb/offline-run-20250226_160155-3qzc8o35/files/checkpoint.pt
```

Clean up

```bash
rm -rf tensorboard_logs/  wandb/ data/ runs/
mkdir logs
```

### Query Shifts(TPCH)

Train

```bash
# run jobs in backend
tmux new -s nr
tmux a
tmux + B , then D to quite


python run.py --run Neo_JOB_JOIN --local
python run.py --run Neo_JOB_PRE --local
python run.py --run Neo_JOB_TABLE --local
```

Test

```bash
python3 test_model.py --run Neo_JOB_JOIN --model_checkpoint  ./neurbench/lqo_ml_perspective/balsa/wandb/offline-run-20241016_203421-8fo30r6c/files/checkpoint.pt
```

Clean up

```bash
rm -rf tensorboard_logs/  wandb/ data/ runs/
mkdir logs
```

### Data Shifts

Train

```bash
# run jobs in backend
tmux new -s nr
tmux a
tmux + B , then D to quite

# train once and test multipe times. Run twice !!
python run.py --run Neo_DB2_SMALL_SET_Train --local
python run.py --run Neo_DB2_SMALL_SET_Train --local
# Note : choose (3) for the wandb
```

Test

1. Change the `dbName`  in the lib `balsa/pg_executor/pg_executor/pg_executor.py`
2. Then run `pip install -e pg_executor`
3. Then run the following code

```bash
python3 test_model.py --run Neo_DB2_SMALL_SET_Train --model_checkpoint ./lqo_ml_perspective/balsa/wandb/offline-run-20250219_213742-35prg91g/files/checkpoint.pt
```

### Save tmux logs

```
tmux capture-pane -J -S - -E - -p > output.log
```



## PARQO

env

```
conda create -n parqo python=3.8
```





