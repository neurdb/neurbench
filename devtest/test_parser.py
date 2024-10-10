from parser import sql_parser, table_parser

CREATE_TABLE_PATH = "testdata/tpch-ddl.sql"
PREPROCESSED_QUERIES_PATH = "testdata/tpch-pp.sql"


if __name__ == "__main__":
    with open(PREPROCESSED_QUERIES_PATH, "r") as f:
        sqls = f.read().split(";")

    table_names, _, table_attr_types_map, short_name_full_name_map = table_parser.get_all_table_attr_infos(
        CREATE_TABLE_PATH
    )
    print(table_names)
    print(table_attr_types_map)
    print(short_name_full_name_map)

    db_data_info = table_parser.retrieve_db_info(
        table_names,
        table_attr_types_map,
        short_name_full_name_map,
        "/Users/kevin/project_python/AI4QueryOptimizer/AI4QueryOptimizer/datasets/dynamic_datasets/tpch/data_0_csv",
        "testdata/tables_info.json",
    )

    base_queries_info = sql_parser.parse_queries_on_batch(sqls, db_data_info)
