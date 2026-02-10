import os
from expert_router.encoder import Sql2VecEmbeddingV2, load_db_info_json
from parser.sql_parser import update_table_aliases_with_psqlparse
from common import IMDBConfig, StackConfig
from db.pg_conn import PostgresConnector
from arg_utils import shared_args
from parser.table_parser import get_all_table_attr_infos, get_index_table, retrieve_db_info


def dataset_profiling(args, all_paths: shared_args.WorkloadPaths):

    """
    summarize some statstics based on the orignal dataset
    """

    # 1. get columns name, type from create sql.
    table_names, _, table_attr_types_map, table_attr_map = get_all_table_attr_infos(
        all_paths.create_table_file_path)
    index_table_map = get_index_table(all_paths.create_index_file_path)

    # 2. retrieval db info from current database
    # Assumptions, known table size, attribute name, etc
    db_data_info = retrieve_db_info(table_names,
                                    table_attr_types_map,
                                    table_attr_map,
                                    index_table_map,
                                    args.whole_dataset_dir,
                                    all_paths.tables_info_path,
                                    all_paths.base_query_file_path)
    return db_data_info




def generate_imdb_encoding():
    cfg = IMDBConfig
    db_profile_res = load_db_info_json(cfg.DB_INFO_DICT)
    sql_vec = Sql2VecEmbeddingV2(db_profile_res=db_profile_res,
                                 checkpoint_file=cfg.EMBED_FILE)

    all_queries = cfg.get_all_queries()
    for query_id in all_queries:
        try:
            sql = cfg.load_sql_query(query_id)
        except Exception as e:
            print(e)
            continue
        print("processing query ", query_id)
        sql_vec.encode_query(query_id, sql, train_database=cfg.DB_NAME)
        print("\n\n")


def _collect_stack_data_info(db_name, all_queries):
    pg_runner = PostgresConnector(db_name)

    db_info = pg_runner.parse_database_schema()

    update_table_aliases_with_psqlparse(
        db_info,
        all_queries,
        output_file=StackConfig.DB_INFO_DICT)


def generate_stack_encoding():
    cfg = StackConfig

    all_queries = cfg.get_all_queries()

    if not os.path.exists(cfg.DB_INFO_DICT):
        _collect_stack_data_info(db_name=cfg.CONFIG.DB_NAME, all_queries=all_queries)

    db_profile_res = load_db_info_json(cfg.DB_INFO_DICT)
    sql_vec = Sql2VecEmbeddingV2(db_profile_res=db_profile_res, checkpoint_file=cfg.EMBED_FILE)

    for query_id in all_queries:
        print("processing query ", query_id)
        sql = cfg.load_sql_query(query_id)
        sql_vec.encode_query(query_id, sql, train_database=cfg.CONFIG.DB_NAME)


if __name__ == "__main__":
    generate_imdb_encoding()
    # generate_stack_encoding()
