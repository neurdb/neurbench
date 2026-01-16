import argparse
import traceback
import random

from utils import *
import os
import socket
from config import *
from multiprocessing import Pool

class PolicyEntity:
    def __init__(self, score) -> None:
        self.score = score

    def get_score(self):
        return self.score


class CardinalityGuidedEntity(PolicyEntity):
    def __init__(self, score, card_str) -> None:
        super().__init__(score)
        self.card_str = card_str


class PgHelper():
    def __init__(self, queries, output_query_latency_file) -> None:
        self.queries = queries
        self.output_query_latency_file = output_query_latency_file

    def start(self, pool_num):
        pool = Pool(pool_num)
        print("---------------- starts PgHelper ----------------")
        for fp, q in self.queries:
            pool.apply_async(do_run_query, args=(q, fp, [], self.output_query_latency_file, True, None, None))
        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()


class LeroHelper():
    def __init__(self, queries, query_num_per_chunk, output_query_latency_file,
                test_queries, model_prefix, topK, training_style="lero", continue_model=None,
                start_chunk_idx=0, model_idx_offset=0) -> None:
        self.queries = queries
        self.query_num_per_chunk = query_num_per_chunk
        self.output_query_latency_file = output_query_latency_file
        self.test_queries = test_queries
        self.model_prefix = model_prefix
        self.topK = topK
        self.training_style = training_style  # "lero" (cardinality-guided) or "bao" (hint-based)
        self.continue_model = continue_model  # Existing model to continue training from
        self.start_chunk_idx = start_chunk_idx  # Resume from this chunk index
        self.model_idx_offset = model_idx_offset  # Offset for model naming when queries are trimmed
        self.lero_server_path = LERO_SERVER_PATH
        self.lero_card_file_path = os.path.join(LERO_SERVER_PATH, LERO_DUMP_CARD_FILE)
        self._ALL_OPTIONS = [
            "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
            "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"
        ]
        self.failed_queries = []  # Track failed queries
        print(f"Training style: {self.training_style}")
        if self.continue_model:
            print(f"Continue training from: {self.continue_model}")
        if self.start_chunk_idx > 0:
            print(f"Resuming from chunk index: {self.start_chunk_idx}")

    @staticmethod
    def find_last_chunk_from_models(model_prefix, lero_server_path):
        """Find the last completed chunk index by looking at existing model directories"""
        import glob
        # Model names are like: {model_prefix}_0, {model_prefix}_1, etc.
        pattern = os.path.join(lero_server_path, model_prefix + "_*")
        model_dirs = glob.glob(pattern)

        if not model_dirs:
            return None

        max_idx = -1
        for model_dir in model_dirs:
            # Extract the chunk index from the directory name
            basename = os.path.basename(model_dir)
            try:
                # model_prefix might contain underscores, so we take the last part after the prefix
                suffix = basename[len(model_prefix) + 1:]  # +1 for the underscore
                idx = int(suffix)
                if idx > max_idx:
                    max_idx = idx
            except (ValueError, IndexError):
                continue

        return max_idx if max_idx >= 0 else None

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _arm_idx_to_hints(self, arm_idx):
        hints = []
        for option in self._ALL_OPTIONS:
            hints.append(f"SET {option} TO off")

        if arm_idx == 0:
            for option in self._ALL_OPTIONS:
                hints.append(f"SET {option} TO on")
        elif arm_idx == 1:
            hints.append("SET enable_hashjoin TO on")
            hints.append("SET enable_indexonlyscan TO on")
            hints.append("SET enable_indexscan TO on")
            hints.append("SET enable_mergejoin TO on")
            hints.append("SET enable_seqscan TO on")
        elif arm_idx == 2:
            hints.append("SET enable_hashjoin TO on")
            hints.append("SET enable_indexonlyscan TO on")
            hints.append("SET enable_nestloop TO on")
            hints.append("SET enable_seqscan TO on")
        elif arm_idx == 3:
            hints.append("SET enable_hashjoin TO on")
            hints.append("SET enable_indexonlyscan TO on")
            hints.append("SET enable_seqscan TO on")
        elif arm_idx == 4:
            hints.append("SET enable_hashjoin TO on")
            hints.append("SET enable_indexonlyscan TO on")
            hints.append("SET enable_indexscan TO on")
            hints.append("SET enable_nestloop TO on")
            hints.append("SET enable_seqscan TO on")
        else:
            raise Exception("Only supports the first 5 arms")
        return hints

    def run_pairwise_with_hints(self, q, fp, run_args, output_query_latency_file, exploratory_query_latency_file, pool):
        print("---------------- run_pairwise_with_hints (SEQUENTIAL MODE) ----------------")
        try:
            # First run with default settings (all options on)
            default_hints = self._arm_idx_to_hints(0)
            default_run_args = run_args + default_hints
            do_run_query(q, fp, default_run_args, output_query_latency_file, True, None, None)

            # Then run with different hint combinations
            for arm_idx in range(1, 5):  # Try arms 1-4
                hints = self._arm_idx_to_hints(arm_idx)
                current_run_args = run_args + hints
                do_run_query(q, fp, current_run_args, exploratory_query_latency_file, True, None, None)

        except Exception as e:
            print("Running sql error", q, e)

    def start(self, pool_num):
        lero_chunks = list(self.chunks(self.queries, self.query_num_per_chunk))
        total_chunks = len(lero_chunks)

        run_args = self.get_run_args()
        print(f"---------------- starts LeroHelper (SEQUENTIAL MODE, style={self.training_style}) ----------------")
        print(f"Total chunks: {total_chunks}, starting from chunk {self.start_chunk_idx}")

        for c_idx, chunk in enumerate(lero_chunks):
            # Skip already completed chunks when resuming
            if c_idx < self.start_chunk_idx:
                print(f"Skipping chunk {c_idx} (already completed)")
                continue

            print(f"Processing chunk {c_idx}/{total_chunks-1}...")
            for fp, q in chunk:
                try:
                    if self.training_style == "bao":
                        # Bao-style: explore plans by disabling different operators
                        self.run_pairwise_with_hints(q, fp, run_args, self.output_query_latency_file,
                                                     self.output_query_latency_file + "_exploratory", None)
                    else:
                        # Original Lero: explore plans by modifying cardinality estimates
                        self.run_pairwise(q, fp, run_args, self.output_query_latency_file,
                                          self.output_query_latency_file + "_exploratory", None)
                except Exception as e:
                    print(f"[ERROR] Query {fp} failed: {str(e)[:200]}")
                    self.failed_queries.append((fp, str(e)))
                    # If database crashed, wait for recovery
                    if is_db_crash_error(e):
                        print(f"[DB Crash] Detected crash during query {fp}, waiting for recovery...")
                        if wait_for_db_recovery():
                            print(f"[DB Crash] Database recovered, continuing with next query...")
                        else:
                            print(f"[DB Crash] Database did not recover, stopping training")
                            break
                    # Continue with next query

            model_idx = c_idx + self.model_idx_offset  # Apply offset for correct model naming
            model_name = self.model_prefix + "_" + str(model_idx)
            self.retrain(model_name)
            print(f"Chunk {c_idx}/{total_chunks-1} completed, model saved: {model_name} (idx={model_idx})")

            # todo: skip the teting for each train
            # self.test_benchmark(self.output_query_latency_file + "_" + model_name)

        # print error query summary
        if self.failed_queries:
            print("\n" + "="*60)
            print(f"Training completed with {len(self.failed_queries)} failed queries:")
            for fp, err in self.failed_queries:
                print(f"  - {fp}: {err[:100]}")
            print("="*60 + "\n")

    def retrain(self, model_name):
        training_data_file = self.output_query_latency_file + ".training"
        create_training_file(training_data_file, self.output_query_latency_file, self.output_query_latency_file + "_exploratory")
        print("retrain Lero model:", model_name, "with file", training_data_file)

        # Create directory for training history
        history_dir = os.path.join(os.path.dirname(self.output_query_latency_file), "training_history")
        os.makedirs(history_dir, exist_ok=True)

        # Use GPU for training (removed CUDA_VISIBLE_DEVICES="" to enable GPU)
        # cmd_str = "cd " + self.lero_server_path + " && CUDA_VISIBLE_DEVICES=\"\" python3.8 train.py" \
        cmd_str = "cd " + self.lero_server_path + " && python3.8 train.py" \
                                                + " --training_data " + os.path.abspath(training_data_file) \
                                                + " --model_name " + model_name \
                                                + " --training_type 1" \
                                                + " --history_file " + os.path.join(history_dir, f"{model_name}_history.json")

        # Add pretrain model if continuing from existing model
        if self.continue_model:
            cmd_str += " --pretrain_model_name " + self.continue_model
            print("Continuing training from pretrain model:", self.continue_model)

        print("run cmd:", cmd_str)
        os.system(cmd_str)

        self.load_model(model_name)
        return model_name

    def load_model(self, model_name):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        json_str = json.dumps({"msg_type":"load", "model_path": os.path.abspath(LERO_SERVER_PATH + model_name)})
        print("load_model", json_str)

        s.sendall(bytes(json_str + "*LERO_END*", "utf-8"))
        reply_json = s.recv(1024)
        s.close()
        # print(reply_json)
        os.system("sync")

    def test_benchmark(self, output_file):
        run_args = self.get_run_args()
        for (fp, q) in self.test_queries:
            do_run_query(q, fp, run_args, output_file, True, None, None)

    def get_run_args(self):
        run_args = []
        run_args.append("SET enable_lero TO True")
        return run_args

    def get_card_test_args(self, card_file_name):
        run_args = []
        run_args.append("SET lero_joinest_fname TO '" + card_file_name + "'")
        return run_args

    def run_pairwise(self, q, fp, run_args, output_query_latency_file, exploratory_query_latency_file, pool):
        print(f"---------------- run_pairwise {fp} (SEQUENTIAL MODE) ----------------")
        try:
            explain_query(q, run_args)
        except Exception as e:
            print(f"Running sql error (explain) {fp}: {e}")
            # If database crashed, re-raise to let start() handle it
            if is_db_crash_error(e):
                raise e
            return  # Skip this query for other errors

        try:
            policy_entities = []
            with open(self.lero_card_file_path, 'r') as f:
                lines = f.readlines()
                lines = [line.strip().split(";") for line in lines]
                for line in lines:
                    policy_entities.append(CardinalityGuidedEntity(float(line[1]), line[0]))

            policy_entities = sorted(policy_entities, key=lambda x: x.get_score())
            policy_entities = policy_entities[:self.topK]

            i = 0
            for entity in policy_entities:
                if isinstance(entity, CardinalityGuidedEntity):
                    card_str = "\n".join(entity.card_str.strip().split(" "))
                    # ensure that the cardinality file will not be changed during planning
                    card_file_name = "lero_" + fp + "_" + str(i) + ".txt"
                    card_file_path = os.path.join(PG_DB_PATH, card_file_name)
                    with open(card_file_path, "w") as card_file:
                        card_file.write(card_str)

                    output_file = output_query_latency_file if i == 0 else exploratory_query_latency_file
                    do_run_query(q, fp, self.get_card_test_args(card_file_name), output_file, True, None, None)
                    i += 1
        except Exception as e:
            print(f"Running sql error (run_pairwise) {fp}: {e}")
            traceback.print_exc()
            # If database crashed, re-raise to let start() handle it
            if is_db_crash_error(e):
                raise e

    def predict(self, plan):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LERO_SERVER_HOST, LERO_SERVER_PORT))
        s.sendall(bytes(json.dumps({"msg_type":"predict", "Plan":plan}) + "*LERO_END*", "utf-8"))
        reply_json = json.loads(s.recv(1024))
        assert reply_json['msg_type'] == 'succ'
        s.close()
        print(reply_json)
        os.system("sync")
        return reply_json['latency']

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--query_path",
                        metavar="PATH",
                        help="Load the queries")
    parser.add_argument("--test_query_path",
                        metavar="PATH",
                        help="Load the test queries")
    parser.add_argument("--algo", type=str)
    parser.add_argument("--query_num_per_chunk", type=int)
    parser.add_argument("--output_query_latency_file", metavar="PATH")
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--pool_num", type=int)
    parser.add_argument("--topK", type=int)
    parser.add_argument("--training_style", type=str, default="lero",
                        choices=["lero", "bao"],
                        help="Training style: 'lero' (cardinality-guided) or 'bao' (hint-based)")
    parser.add_argument("--min_queries", type=int, default=None,
                        help="Minimum number of training queries (will sample with replacement if needed). Default: no sampling")
    parser.add_argument("--continue_data_file", type=str, default=None,
                        help="Existing data file to append to (for continue training)")
    parser.add_argument("--continue_model", type=str, default=None,
                        help="Existing model path to continue training from")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved progress (continue from last completed chunk)")
    args = parser.parse_args()

    query_path = args.query_path
    print("Load queries from ", query_path)

    # For resume mode, check if sampled queries file exists
    sampled_queries_file = args.output_query_latency_file + ".sampled_queries" if args.output_query_latency_file else None

    queries = []
    loaded_from_sampled_file = False
    if args.resume and sampled_queries_file and os.path.exists(sampled_queries_file):
        # Load previously sampled queries for consistent resume
        print(f"Loading previously sampled queries from {sampled_queries_file}")
        with open(sampled_queries_file, 'r') as f:
            for line in f.readlines():
                arr = line.strip().split(SEP)
                queries.append((arr[0], arr[1]))
        print(f"Loaded {len(queries)} sampled queries for resume")
        loaded_from_sampled_file = True
    else:
        with open(query_path, 'r') as f:
            for line in f.readlines():
                arr = line.strip().split(SEP)
                queries.append((arr[0], arr[1]))
        print("Read", len(queries), "training queries.")

        # Sample queries if needed (like BAO does)
        min_queries = args.min_queries
        if min_queries is not None and len(queries) < min_queries:
            # Use fixed seed for reproducible sampling (important for resume)
            SAMPLE_SEED = 42
            random.seed(SAMPLE_SEED)
            print(f"Sampling queries from {len(queries)} to {min_queries} (with replacement, seed={SAMPLE_SEED})")
            queries = random.choices(queries, k=min_queries)
            print(f"After sampling: {len(queries)} training queries")

            # Save sampled queries for resume
            if sampled_queries_file:
                with open(sampled_queries_file, 'w') as f:
                    for qname, qsql in queries:
                        f.write(f"{qname}{SEP}{qsql}\n")
                print(f"Saved sampled queries to {sampled_queries_file}")
        else:
            print(f"Using original {len(queries)} queries (no sampling)")

    output_query_latency_file = args.output_query_latency_file
    print("output_query_latency_file:", output_query_latency_file)

    pool_num = 1
    if args.pool_num:
        pool_num = args.pool_num
    print("pool_num:", pool_num)

    ALGO_LIST = ["lero", "pg"]
    algo = "lero"
    if args.algo:
        assert args.algo.lower() in ALGO_LIST
        algo = args.algo.lower()
    print("algo:", algo)

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    if algo == "pg":
        helper = PgHelper(queries, output_query_latency_file)
        helper.start(pool_num)
    else:
        test_queries = []
        if args.test_query_path is not None:
            with open(args.test_query_path, 'r') as f:
                for line in f.readlines():
                    arr = line.strip().split(SEP)
                    test_queries.append((arr[0], arr[1]))
        print("Read", len(test_queries), "test queries.")

        query_num_per_chunk = args.query_num_per_chunk
        print("query_num_per_chunk:", query_num_per_chunk)

        model_prefix = None
        if args.model_prefix:
            model_prefix = args.model_prefix
        print("model_prefix:", model_prefix)

        topK = 5
        if args.topK is not None:
            topK = args.topK
        print("topK", topK)

        training_style = args.training_style
        print("training_style:", training_style)

        continue_model = args.continue_model
        if continue_model:
            print("continue_model:", continue_model)

        # Check for resume mode - find last chunk from existing models
        start_chunk_idx = 0
        model_idx_offset = 0
        if args.resume and model_prefix:
            last_chunk = LeroHelper.find_last_chunk_from_models(model_prefix, LERO_SERVER_PATH)
            if last_chunk is not None:
                start_chunk_idx = last_chunk + 1  # Start from next chunk
                print(f"Found existing models up to chunk {last_chunk}, resuming from chunk {start_chunk_idx}")

                # If we didn't load from sampled file, trim queries and use offset for model naming
                if not loaded_from_sampled_file:
                    skip_queries = start_chunk_idx * query_num_per_chunk
                    if skip_queries < len(queries):
                        print(f"No sampled queries file found. Trimming first {skip_queries} queries (already processed).")
                        queries = queries[skip_queries:]
                        model_idx_offset = start_chunk_idx  # Use offset for correct model naming
                        start_chunk_idx = 0  # Reset since queries are trimmed
                        print(f"Remaining queries: {len(queries)}, model naming starts from {model_idx_offset}")
                    else:
                        print(f"All queries already processed (skip={skip_queries}, total={len(queries)}). Nothing to do.")
                        queries = []
            else:
                print("No existing models found, starting from beginning")

        helper = LeroHelper(queries, query_num_per_chunk, output_query_latency_file, test_queries,
                           model_prefix, topK, training_style, continue_model, start_chunk_idx, model_idx_offset)
        helper.start(pool_num)
