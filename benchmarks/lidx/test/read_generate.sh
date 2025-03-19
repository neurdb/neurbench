# imdb

# --------------- lookup only --------------- 
./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_00/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_00/insert_keys_uint64 \
    10000000 


./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_01/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_01/insert_keys_uint64 \
    10000000 



./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_03/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_03/insert_keys_uint64 \
    10000000 


./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_05/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_05/insert_keys_uint64 \
    10000000 





#  ------------ range query (scope 100) (mixed with lookup)------------
./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_00/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_00/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 100

./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_01/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_01/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 100

./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_03/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_03/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 100


./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_05/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_05/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 100



#  ------------ range query  (scope 50) (mixed with Lookup) ------------
./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_00/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_00/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 50

./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_01/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_01/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 50

./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_03/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_03/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 50
    

./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_05/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_05/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 50




#  ------------ range query  (scope 25) (mixed with Lookup) ------------
./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_00/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_00/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 25

./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_01/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_01/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 25

./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_03/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_03/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 25
    

./build/drift_generate \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_05/init_keys_uint64 \
    /users/lingze/neurbench/data/workload/imdb_4M_uint64_drift_05/insert_keys_uint64 \
    10000000 \
    -s 0.5 \
    -r 25