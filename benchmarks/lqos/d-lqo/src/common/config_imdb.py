import os
from common.base_config import BaseConfig


class IMDBConfig(BaseConfig):
    DB_NAME = "imdb_1_gen"
    
    ALL_METHODS = ["balsa", "bao", "lero", "pg"]
    
    # ALL_METHODS = ['HintPlanSel', 'Balsa', 'Neo', 'PostgreSQL', 'JoinOrder']
    FIXED_LABEL_MAPPING = {m: i for i, m in enumerate(ALL_METHODS)}
    EXECUTION_TIME_OUT = 360000.0

    CSV = './datasets/imdb_collection.csv'
    QUERY_DIR = "./datasets/origin_datasets/imdb/"

    DB_INFO_DICT = "./experiment/datasets/ori_table_info.json"

    TRAIN_TEST = f"./experiment/datasets/workload_data_train_test/{DB_NAME}"
    EMBED_FILE = f"./experiment/result_data/query_encodings_embedding_v2_{DB_NAME}.json"
    TRAIN_TEST_ONLINE = "./experiment/datasets/workload_data_train_test_online_mix"
    TRAIN_TEST_ONLINE_CONVARIATE = "./experiment/datasets/workload_data_train_test_online_mix_convariate"
    TESTPATH = f"./experiment/datasets/workload_data_test/{DB_NAME}"

    SENSITIVE_EXPERT_DICT = {
        # 'base_query_split_1': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
        # 'base_query_split_2': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
        'base_query_split_3': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
        'leave_one_out_split_1': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
        # 'leave_one_out_split_2': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
        # 'leave_one_out_split_3': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
        'random_split_1': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
        # 'random_split_2': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
        'random_split_3': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
    }

    JOIN_COUNTS = {
        '3a': 3, '3b': 3, '3c': 3,
        '4a': 4, '4b': 4, '4c': 4, '2a': 4, '2b': 4, '2c': 4, '2d': 4,
        '5a': 4, '5b': 4, '5c': 4, '6a': 4, '6b': 4, '6c': 4, '6d': 4, '6e': 4, '6f': 4,
        '32a': 5, '32b': 5, '1a': 5, '1b': 5, '1c': 5, '1d': 5,
        '10a': 6, '10b': 6, '10c': 6, '17a': 6, '17b': 6, '17c': 6, '17d': 6, '17e': 6, '17f': 6,
        '8a': 6, '8b': 6, '8c': 6, '8d': 6, '18a': 6, '18b': 6, '18c': 6,
        '12a': 7, '12b': 7, '12c': 7, '13a': 7, '13b': 7, '13c': 7, '13d': 7,
        '14a': 7, '14b': 7, '14c': 7, '9a': 7, '9b': 7, '9c': 7, '9d': 7,
        '11a': 8, '11b': 8, '11c': 8, '11d': 8, '15a': 8, '15b': 8, '15c': 8, '15d': 8,
        '16a': 8, '16b': 8, '16c': 8, '16d': 8, '7a': 8, '7b': 8, '7c': 8,
        '21a': 9, '21b': 9, '21c': 9,
        '22a': 10, '22b': 10, '22c': 10, '22d': 10, '25a': 10, '25b': 10, '25c': 10,
        '31a': 10, '31b': 10, '31c': 10,
        '20a': 11, '20b': 11, '20c': 11, '23a': 11, '23b': 11, '23c': 11,
        '30a': 11, '30b': 11, '30c': 11,
        '19a': 12, '19b': 12, '19c': 12, '19d': 12,
        '24a': 13, '24b': 13, '26a': 13, '26b': 13, '26c': 13,
        '27a': 14, '27b': 14, '27c': 14, '28a': 14, '28b': 14, '28c': 14,
        '33a': 14, '33b': 14, '33c': 14,
        '29a': 16, '29b': 16, '29c': 16
    }

    TEST_QUERIES = {
        'base_query_split_1': ['02a', '02b', '02c', '02d', '07a', '07b', '07c', '15a', '15b', '15c', '15d', '24a',
                               '24b',
                               '25a', '25b', '25c', '31a', '31b', '31c'],
        'base_query_split_2': ['13a', '13b', '13c', '13d', '15a', '15b', '15c', '15d', '20a', '20b', '20c', '26a',
                               '26b',
                               '26c', '29a', '29b', '29c', '30a', '30b', '30c', '33a', '33b', '33c'],
        'base_query_split_3': ['01a', '01b', '01c', '01d', '05a', '05b', '05c', '12a', '12b', '12c', '17a', '17b',
                               '17c',
                               '17d', '17e', '17f', '22a', '22b', '22c', '22d', '27a', '27b', '27c', '28a', '28b',
                               '28c'],
        'leave_one_out_split_1': ['01c', '02a', '03b', '04a', '05a', '06b', '07c', '08c', '09c', '10b', '11b', '12c',
                                  '13b',
                                  '14a', '15b', '16c', '17c', '18b', '19a', '20c', '21c', '22b', '23b', '24a', '25a',
                                  '26c',
                                  '27c', '28a', '29b', '30a', '31b', '32b', '33c'],
        'leave_one_out_split_2': ['01d', '02d', '03a', '04b', '05c', '06d', '07a', '08c', '09c', '10a', '11a', '12a',
                                  '13d',
                                  '14b', '15b', '16a', '17f', '18a', '19d', '20a', '21b', '22c', '23b', '24b', '25a',
                                  '26a',
                                  '27b', '28c', '29a', '30b', '31a', '32b', '33b'],
        'leave_one_out_split_3': ['01c', '02d', '03b', '04a', '05c', '06d', '07b', '08a', '09a', '10c', '11d', '12a',
                                  '13a',
                                  '14b', '15a', '16d', '17b', '18b', '19d', '20b', '21a', '22a', '23b', '24a', '25b',
                                  '26a',
                                  '27a', '28b', '29c', '30a', '31a', '32a', '33c'],
        'random_split_1': ['01c', '02c', '04b', '04c', '05c', '06a', '06c', '06e', '08b', '08c', '09c', '11d', '15a',
                           '17b',
                           '17e', '18b', '20a', '21a', '25c', '28b', '32b', '33a'],
        'random_split_2': ['01a', '04c', '05c', '06c', '06d', '07b', '08c', '10a', '11a', '11d', '13c', '13d', '15d',
                           '16a',
                           '17b', '19a', '20a', '22b', '25b', '29b', '31a', '32b'],
        'random_split_3': ['02a', '03b', '06d', '09b', '10b', '11b', '11c', '13c', '13d', '16b', '18c', '19c', '21c',
                           '22a',
                           '22d', '26a', '26b', '27c', '28a', '28c', '30a', '33c'],
    }

    id2aliasname = {
        0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k', 9: 'lt',
        10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt',
        18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3',
        26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1',
        34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'
    }
    aliasname2id = {
        'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39, 'kt2': 32,
        'miidx': 18, 'it': 16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8,
        'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an1': 37, 'at': 19,
        'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12,
        'cc': 22, 'cct2': 24, 'an': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33
    }

    @staticmethod
    def load_sql_query(query_ident: str) -> str:
        sql_filename = IMDBConfig.ident_to_sql_filename(query_ident)
        filepath = os.path.join(IMDBConfig.QUERY_DIR, 'join-order-benchmark', sql_filename)
        with open(filepath) as f:
            return f.read().strip().replace("\n", " ")

    @staticmethod
    def get_all_queries():
        path = os.path.join(IMDBConfig.QUERY_DIR, 'join-order-benchmark')
        return sorted([q[:-4] for q in os.listdir(path) if q.endswith('.sql')])


def GetAllTableNumRows(rel_names, cursor=None):
    """Ask PG how many number of rows each rel in rel_names has.

    Returns:
      A dict, {rel name: # rows}.
    """

    CACHE = {
        'aka_name': 901343,
        'aka_title': 361472,
        'cast_info': 36244344,
        'char_name': 3140339,
        'comp_cast_type': 4,
        'company_name': 234997,
        'company_type': 4,
        'complete_cast': 135086,
        'info_type': 113,
        'keyword': 134170,
        'kind_type': 7,
        'link_type': 18,
        'movie_companies': 2609129,
        'movie_info': 14835720,
        'movie_info_idx': 1380035,
        'movie_keyword': 4523930,
        'movie_link': 29997,
        'name': 4167491,
        'person_info': 2963664,
        'role_type': 12,
        'title': 2528312,

        'account': 13863748,
        'tag_question': 36883820,
        'site': 173,
        'question': 12631974,
        'badge': 51232233,
        'so_user': 21097404,
        'tag': 186770,
        'comment': 103557557,
        'answer': 6343509,
        'post_link': 2264333,
    }

    d = {}
    for rel_name in rel_names:
        if rel_name in CACHE:
            # Kind of slow to ask PG for this.  For some reason it doesn't
            # immediately return from catalog but instead seems to do scans.
            d[rel_name] = CACHE[rel_name]
            continue

        sql = 'SELECT count(*) FROM {};'.format(rel_name)
        print('Issue:', sql)
        cursor.execute(sql)
        num_rows = cursor.fetchall()[0][0]
        print(num_rows)
        d[rel_name] = num_rows
    return d
