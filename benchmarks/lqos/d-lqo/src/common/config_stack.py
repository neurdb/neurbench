# config/stack_config.py
import os
from common.base_config import BaseConfig


class StackConfig(BaseConfig):
    DB_NAME = "stack"
    # ALL_METHODS = ['Bao', 'Balsa', 'Neo', 'PostgreSQL', 'HybridQO', 'LEON']
    ALL_METHODS = ['HintPlanSel', 'Balsa', 'Neo', 'PostgreSQL', 'JoinOrder', 'LEON']
    FIXED_LABEL_MAPPING = {m: i for i, m in enumerate(ALL_METHODS)}
    EXECUTION_TIME_OUT = 3 * 60 * 1000.0

    CSV = './datasets/stack_collection.csv'
    DB_INFO_DICT = "./experiment_result/datasets/ori_table_info_stack.json"
    QUERY_DIR = "./datasets/origin_datasets/stack/stack_queries"

    TRAIN_TEST = "./experiment_result/datasets/workload_data_train_test_stack"
    EMBED_FILE = "./experiment_result/result_data/query_encodings_embedding_v2_stack.json"
    TESTPATH = "./experiment_result/datasets/workload_data_test_stack"
    TRAIN_TEST_ONLINE = None
    TRAIN_TEST_ONLINE_CONVARIATE = None

    SENSITIVE_EXPERT_DICT = {
        'base_query_split_1': ['Bao', 'Balsa', 'Neo', 'PostgreSQL', "LEON"],
        'leave_one_out_split_1': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
        'random_split_1': [ele for ele in ALL_METHODS if ele != "PostgreSQL"],
    }

    TEST_QUERIES = {
        "base_query_split_1":
            ['q13__13ad1b8c6bea4fda1892b9fa82cc1ceb9ceb85fc', 'q13__1ddcc8650e17b292bc7344902baffc90c5ae5761',
             'q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2', 'q13__a091adce62743b65c04532e98e8ff3d7e546ea77',
             'q13__a3d03772d880754fc4e150d82908757477ae2186', 'q13__add0df9dccb2790c14508e19c9e0deb79fad6ea2',
             'q13__d383cd5b4aee7d3f73508e2a1fe5f6d0f7dd42a2', 'q13__d4707be2adfdbc842f42acb1fc16e3a43faf7474',
             'q2__q2-001',
             'q2__q2-012', 'q2__q2-032', 'q2__q2-035', 'q2__q2-050', 'q2__q2-081', 'q2__q2-094', 'q2__q2-098',
             'q7__q7-034',
             'q7__q7-036', 'q7__q7-047', 'q7__q7-077', 'q7__q7-082', 'q7__q7-085', 'q7__q7-095', 'q7__q7-099'],

        "leave_one_out_split_1":
            ['q11__6c5cba419c5b7b02d431aeb5e766d775d812967a', 'q12__547c6bf1994c9b2ba82a7ae32f4b051beabf46fd',
             'q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2', 'q14__5e4835cd72aaa2d7be15b2a5ffa2e66156b3656f',
             'q15__543ab3f730e494a69e3d15e59675f491544cb15d', 'q16__b1a96cd48ba297dd93bce73c27b491069ad7449f',
             'q1__q1-035',
             'q2__q2-032', 'q3__q3-043', 'q4__q4-041', 'q5__q5-041', 'q6__q6-060', 'q7__q7-047', 'q8__q8-046'],

        "random_split_1":
            ['q11__6c5cba419c5b7b02d431aeb5e766d775d812967a', 'q11__c1ae2a992cde4ea2c4922d852df22043254b4f84',
             'q12__55de941e8497cfeeb93d3f8f2d7a18489e0e6c32', 'q14__63c0776f1727638316b966fe748df7cc585a335b',
             'q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94', 'q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6',
             'q14__b49361f85785200ed6ec1f2eec357b7598c9e564', 'q15__3e37e62655ceaebc14e79edad518e5710752f51d',
             'q15__543ab3f730e494a69e3d15e59675f491544cb15d', 'q15__b8ddf65b0c0c7867a9b560e571d457fec410715c',
             'q15__d5546c01928a687eb1f54e9f8eb4e1aff68fc381', 'q16__1e863562a79ca1f7754c759ebab6a2addda0bde8',
             'q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637', 'q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa',
             'q1__q1-031', 'q1__q1-035', 'q4__q4-042', 'q4__q4-064', 'q4__q4-089', 'q5__q5-032', 'q6__q6-060',
             'q6__q6-064', 'q7__q7-099']
    }

    @staticmethod
    def load_sql_query(query_ident: str) -> str:
        path = os.path.join(StackConfig.QUERY_DIR, f"{query_ident}.sql")
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('--')]
            return " ".join(lines)

    @staticmethod
    def get_all_queries():
        return sorted([f.replace('.sql', '') for f in os.listdir(StackConfig.QUERY_DIR) if f.endswith('.sql')])
