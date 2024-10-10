from parser import sql_parser, table_parser

CREATE_TABLE_PATH = "testdata/tpch-ddl.sql"
PREPROCESSED_QUERIES_PATH = "testdata/tpch-pp.sql"


if __name__ == "__main__":
    with open(PREPROCESSED_QUERIES_PATH, "r") as f:
        sqls = f.read().split(";")

    table_names, _, table_attr_types_map = table_parser.get_all_table_attr_infos(
        CREATE_TABLE_PATH
    )
    print(table_names)
    print(table_attr_types_map)

    db_data_info = table_parser.retrieve_db_info(
        table_names,
        table_attr_types_map,
        "../tpch-kit/dbgen/1g_csv",
        "testdata/tables_info.json",
    )

    base_queries_info = sql_parser.parse_queries_on_batch(sqls, db_data_info)
