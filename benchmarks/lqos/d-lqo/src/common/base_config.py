import torch


class BaseConfig:

    # Whitelist of valid join and scan operation combinations for Postgres, empirically determined.
    # Note: Combinations involving 'Seq Scan' require the scan to be on a "small" table.
    NestedLoop_WHITE_LIST = {
        ('Nested Loop', 'Index Scan'),
        ('Nested Loop', 'Seq Scan'),
        ('Nested Loop', 'Index Only Scan'),
        ('Hash Join', 'Index Scan'),
        ('Hash Join', 'Index Only Scan'),
        ('Merge Join', 'Index Scan'),
        ('Seq Scan', 'Index Scan'),
        ('Seq Scan', 'Nested Loop'),
        ('Seq Scan', 'Index Only Scan'),
        ('Index Scan', 'Index Scan'),
        ('Index Scan', 'Seq Scan'),
    }

    JOIN_ORDER_BENCHMARK_JOIN_GRAPH = {
        'aka_title': ['title'],
        'char_name': ['cast_info'],
        'role_type': ['cast_info'],
        'comp_cast_type': ['complete_cast'],
        'movie_link': ['title', 'link_type'] + [
            'complete_cast', 'aka_title', 'movie_link', 'cast_info',
            'movie_companies', 'movie_keyword', 'movie_info_idx', 'movie_info',
            'kind_type'
        ],  # movie_link.id linked to title.id which are both primary keys
        'link_type': ['movie_link'],
        'cast_info': ['char_name', 'role_type', 'title', 'aka_name'],
        'complete_cast': ['comp_cast_type', 'title'],
        'title': [
            'complete_cast', 'aka_title', 'movie_link', 'cast_info',
            'movie_companies', 'movie_keyword', 'movie_info_idx', 'movie_info',
            'kind_type'
        ],
        'aka_name': ['cast_info', 'name'],
        'movie_companies': ['title', 'company_name', 'company_type'],
        'kind_type': ['title'],
        'name': ['aka_name', 'person_info'] +
                ['cast_info'],  # name.id linked to aka_name.id which are both primary keys
        'company_type': ['movie_companies'],
        'movie_keyword': ['title', 'keyword'],
        'movie_info': ['title', 'info_type'],
        'person_info': ['name', 'info_type'],
        'info_type': ['movie_info', 'person_info', 'movie_info_idx'],
        'company_name': ['movie_companies'],
        'keyword': ['movie_keyword'],
        'movie_info_idx': ['title', 'info_type'],
    }

    STACK_JOIN_GRAPH = {
        'account': ['so_user'],
        'answer': ['site', 'so_user', 'question'],
        'badge': ['site', 'so_user'],
        'comment': ['site'],
        'post_link': ['site', 'question'],
        'question': ['answer', 'post_link', 'tag_question', 'site', 'so_user'],
        'site': ['site', 'answer', 'badge', 'comment', 'post_link', 'question', 'so_user', 'tag', 'tag_question'],
        'so_user': ['account', 'answer', 'badge', 'question'],
        'tag': ['site', 'tag_question'],
        'tag_question': ['site', 'tag', 'question'],
    }

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    LOG_DIR = "./experiment_result/logs"
    RESULT_DATA_BASE = "./experiment_result/result_data/"

    HINTSETS = [
        ["enable_hashjoin", "enable_indexscan", "enable_mergejoin", "enable_nestloop", "enable_seqscan",
         "enable_indexonlyscan"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_indexscan", "enable_mergejoin", "enable_seqscan"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_nestloop", "enable_seqscan"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_seqscan"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_indexscan", "enable_nestloop", "enable_seqscan"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_mergejoin", "enable_nestloop"],
        ["enable_hashjoin", "enable_indexscan", "enable_mergejoin", "enable_nestloop"],
        ["enable_indexonlyscan", "enable_mergejoin", "enable_nestloop"],
        ["enable_hashjoin", "enable_indexonlyscan"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_indexscan", "enable_nestloop"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_indexscan", "enable_seqscan"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_mergejoin", "enable_nestloop", "enable_seqscan"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_mergejoin", "enable_seqscan"],
        ["enable_hashjoin", "enable_indexscan", "enable_nestloop"],
        ["enable_indexscan", "enable_nestloop"],
        ["enable_indexscan", "enable_mergejoin", "enable_nestloop", "enable_seqscan"],
        ["enable_indexonlyscan", "enable_indexscan", "enable_nestloop"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_indexscan", "enable_mergejoin", "enable_nestloop"],
        ["enable_indexscan", "enable_mergejoin", "enable_nestloop"],
        ["enable_indexonlyscan", "enable_mergejoin", "enable_nestloop", "enable_seqscan"],
        ["enable_indexonlyscan", "enable_indexscan", "enable_nestloop", "enable_seqscan"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_indexscan", "enable_mergejoin"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_mergejoin"],
        ["enable_hashjoin", "enable_indexscan", "enable_nestloop", "enable_seqscan"],
        ["enable_hashjoin", "enable_indexscan"],
        ["enable_hashjoin", "enable_indexonlyscan", "enable_nestloop"],
    ]

    @staticmethod
    # --- Utility Functions ---
    def query_file_to_ident(file_name: str) -> str:
        """Convert query filename to identifier (e.g., '1a.sql' -> '01a')."""
        ident = file_name.split('.sql')[0]
        return f"{ident[:-1].zfill(2)}{ident[-1]}"

    # --- Utility Functions ---
    @staticmethod
    def query_file_to_ident_stack(file_name: str) -> str:
        """Convert query filename to identifier (e.g., '1a.sql' -> '01a')."""
        ident = file_name.split('.sql')[0]
        return ident

    @staticmethod
    def ident_to_sql_filename(query_ident: str) -> str:
        """Convert query identifier back to SQL filename (e.g., '01a' -> '1a.sql')."""
        num = str(int(query_ident[:-1]))  # Remove leading zeros and get number
        letter = query_ident[-1]
        return f"{num}{letter}.sql"
