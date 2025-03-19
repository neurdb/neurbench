# -----------------fixed read workload generation -----------------
./build/drift_generate \
    /data/lingze/workload/fb_200M_uint64_drift_01/init_keys_uint64 \
    /data/lingze/workload/fb_200M_uint64_drift_01/insert_keys_uint64 \
    200000000 \
    --fix

./build/drift_generate \
    /data/lingze/workload/fb_200M_uint64_drift_03/init_keys_uint64 \
    /data/lingze/workload/fb_200M_uint64_drift_03/insert_keys_uint64 \
    200000000 \
    --fix

./build/drift_generate \
    /data/lingze/workload/fb_200M_uint64_drift_05/init_keys_uint64 \
    /data/lingze/workload/fb_200M_uint64_drift_05/insert_keys_uint64 \
    200000000 \
    --fix

./build/drift_generate \
    /data/lingze/workload/fb_200M_uint64_drift_07/init_keys_uint64 \
    /data/lingze/workload/fb_200M_uint64_drift_07/insert_keys_uint64 \
    200000000 \
    --fix



# osm
./build/drift_generate \
    /data/lingze/workload/osm_cellids_200M_uint64_drift_01/init_keys_uint64 \
    /data/lingze/workload/osm_cellids_200M_uint64_drift_01/insert_keys_uint64 \
    200000000 \
    --fix


./build/drift_generate \
    /data/lingze/workload/osm_cellids_200M_uint64_drift_03/init_keys_uint64 \
    /data/lingze/workload/osm_cellids_200M_uint64_drift_03/insert_keys_uint64 \
    200000000 \
    --fix

./build/drift_generate \
    /data/lingze/workload/osm_cellids_200M_uint64_drift_05/init_keys_uint64 \
    /data/lingze/workload/osm_cellids_200M_uint64_drift_05/insert_keys_uint64 \
    200000000 \
    --fix

./build/drift_generate \
    /data/lingze/workload/osm_cellids_200M_uint64_drift_07/init_keys_uint64 \
    /data/lingze/workload/osm_cellids_200M_uint64_drift_07/insert_keys_uint64 \
    200000000 \
    --fix


# books
./build/drift_generate \
    /data/lingze/workload/books_200M_uint64_drift_01/init_keys_uint64 \
    /data/lingze/workload/books_200M_uint64_drift_01/insert_keys_uint64\
    200000000 \
    --fix

# 0.3
./build/drift_generate \
    /data/lingze/workload/books_200M_uint64_drift_03/init_keys_uint64 \
    /data/lingze/workload/books_200M_uint64_drift_03/insert_keys_uint64\
    200000000 \
    --fix
# 0.5
./build/drift_generate \
    /data/lingze/workload/books_200M_uint64_drift_05/init_keys_uint64 \
    /data/lingze/workload/books_200M_uint64_drift_05/insert_keys_uint64\
    200000000 \
    --fix

# 0.7
./build/drift_generate \
    /data/lingze/workload/books_200M_uint64_drift_07/init_keys_uint64 \
    /data/lingze/workload/books_200M_uint64_drift_07/insert_keys_uint64\
    200000000 \
    --fix



# wiki_ts
./build/drift_generate \
    /data/lingze/workload/wiki_ts_200M_uint64_drift_01/init_keys_uint64 \
    /data/lingze/workload/wiki_ts_200M_uint64_drift_01/insert_keys_uint64\
    200000000 \
    --fix

# 0.3
./build/drift_generate \
    /data/lingze/workload/wiki_ts_200M_uint64_drift_03/init_keys_uint64 \
    /data/lingze/workload/wiki_ts_200M_uint64_drift_03/insert_keys_uint64\
    200000000 \
    --fix
# 0.5
./build/drift_generate \
    /data/lingze/workload/wiki_ts_200M_uint64_drift_05/init_keys_uint64 \
    /data/lingze/workload/wiki_ts_200M_uint64_drift_05/insert_keys_uint64\
    200000000 \
    --fix

# 0.7
./build/drift_generate \
    /data/lingze/workload/wiki_ts_200M_uint64_drift_07/init_keys_uint64 \
    /data/lingze/workload/wiki_ts_200M_uint64_drift_07/insert_keys_uint64\
    200000000 \
    --fix

