from experiments import *

######################### NRBench (NB) Revision #########################


######################### 1. JOB query on multiple datasets #########################

full_query_test_query_glob_job = [
    '1a.sql', '1b.sql', '1c.sql', '1d.sql', '2a.sql', '2b.sql', '2c.sql', '2d.sql',
    '3a.sql', '3b.sql', '3c.sql', '4a.sql', '4b.sql', '4c.sql', '5a.sql', '5b.sql', '5c.sql',
    '6a.sql', '6b.sql', '6c.sql', '6d.sql', '6e.sql', '6f.sql',
    '7a.sql', '7b.sql', '7c.sql', '8a.sql', '8b.sql', '8c.sql', '8d.sql',
    '9a.sql', '9b.sql', '9c.sql', '9d.sql', '10a.sql', '10b.sql', '10c.sql',
    '11a.sql', '11b.sql', '11c.sql', '11d.sql', '12a.sql', '12b.sql', '12c.sql',
    '13a.sql', '13b.sql', '13c.sql', '13d.sql', '14a.sql', '14b.sql', '14c.sql',
    '15a.sql', '15b.sql', '15c.sql', '15d.sql', '16a.sql', '16b.sql', '16c.sql', '16d.sql',
    '17a.sql', '17b.sql', '17c.sql', '17d.sql', '17e.sql', '17f.sql',
    '18a.sql', '18b.sql', '18c.sql', '19a.sql', '19b.sql', '19c.sql', '19d.sql',
    '20a.sql', '20b.sql', '20c.sql', '21a.sql', '21b.sql', '21c.sql',
    '22a.sql', '22b.sql', '22c.sql', '22d.sql', '23a.sql', '23b.sql', '23c.sql',
    '24a.sql', '24b.sql', '25a.sql', '25b.sql', '25c.sql',
    '26a.sql', '26b.sql', '26c.sql', '27a.sql', '27b.sql', '27c.sql',
    '28a.sql', '28b.sql', '28c.sql', '29a.sql', '29b.sql', '29c.sql',
    '30a.sql', '30b.sql', '30c.sql', '31a.sql', '31b.sql', '31c.sql',
    '32a.sql', '32b.sql', '33a.sql', '33b.sql', '33c.sql'
]
empty_test_query_glob_job = ['1a.sql']
# Query directory - use path relative to docker mount point
full_query_query_dir_job = '/app/AI4QueryOptimizer/neurbench/queries/job'

current_used_query_dir_job = full_query_query_dir_job

# Model save directory
MODEL_SAVE_BASE = '/app/AI4QueryOptimizer/neurbench/balsa_logs_all'
current_used_test_query_glob_in_train_job = empty_test_query_glob_job
current_used_test_query_glob_in_test_job = full_query_test_query_glob_job
# originally, it is 100 for job, 50 for stack datasets
current_val_iters = 10


@balsa.params_registry.Register
class NB_Balsa_train_imdb_ori_job_datashift(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_train_job
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = MODEL_SAVE_BASE
        p.model_prefix = 'balsa_imdb_ori_job_full'
        return p


@balsa.params_registry.Register
class NB_Balsa_train_imdb_01v2_job_datashift(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_train_job
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = MODEL_SAVE_BASE
        p.model_prefix = 'balsa_imdb_01v2_job_full'
        return p

@balsa.params_registry.Register
class NB_Balsa_train_imdb_03v2_job_datashift(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_03v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_train_job
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = MODEL_SAVE_BASE
        p.model_prefix = 'balsa_imdb_03v2_job_full'
        return p

@balsa.params_registry.Register
class NB_Balsa_train_imdb_05v2_job_datashift(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_train_job
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = MODEL_SAVE_BASE
        p.model_prefix = 'balsa_imdb_05v2_job_full'
        return p


@balsa.params_registry.Register
class NB_Balsa_train_imdb_17v2_job_datashift(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_train_job
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = MODEL_SAVE_BASE
        p.model_prefix = 'balsa_imdb_17v2_job_full'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_ori_job_datashift(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_train_job
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = MODEL_SAVE_BASE
        p.model_prefix = 'neo_imdb_ori_job_full'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_01v2_job_datashift(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_train_job
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = MODEL_SAVE_BASE
        p.model_prefix = 'neo_imdb_01v2_job_full'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_05v2_job_datashift(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_train_job
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = MODEL_SAVE_BASE
        p.model_prefix = 'neo_imdb_05v2_job_full'
        return p


@balsa.params_registry.Register
class NB_Neo_train_imdb_17v2_job_datashift(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_train_job
        p.validate_every_n_epochs = 200
        p.val_iters = current_val_iters
        p.model_save_path = MODEL_SAVE_BASE
        p.model_prefix = 'neo_imdb_17v2_job_full'
        return p


######## JOB test classes #########


@balsa.params_registry.Register
class NB_Balsa_test_imdb_ori_job_datashift(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_test_job
        return p


@balsa.params_registry.Register
class NB_Balsa_test_imdb_01v2_job_datashift(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_test_job
        return p


@balsa.params_registry.Register
class NB_Balsa_test_imdb_05v2_job_datashift(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_test_job
        return p


@balsa.params_registry.Register
class NB_Balsa_test_imdb_17v2_job_datashift(Balsa_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_test_job
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_ori_job_datashift(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_ori'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_test_job
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_01v2_job_datashift(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_01v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_test_job
        return p

@balsa.params_registry.Register
class NB_Neo_test_imdb_03v2_job_datashift(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_03v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_test_job
        return p

@balsa.params_registry.Register
class NB_Neo_test_imdb_05v2_job_datashift(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_05v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_test_job
        return p

@balsa.params_registry.Register
class NB_Neo_test_imdb_17_job_datashift(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_test_job
        return p


@balsa.params_registry.Register
class NB_Neo_test_imdb_17v2_job_datashift(Neo_JOB_EvaluationBase):
    def Params(self):
        p = super().Params()
        p.db = 'imdb_17v2'
        p.query_dir = current_used_query_dir_job
        p.test_query_glob = current_used_test_query_glob_in_test_job
        return p
