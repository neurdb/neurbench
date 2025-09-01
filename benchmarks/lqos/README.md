This part of the test code is cloned and updated based on the codebase used in the paper "Is Your Learned Query Optimizer Behaving As You Expect? A Machine Learning Perspective", VLDB 2024.

# Setup

This repository includes a variety of learned query optimizer methods and ways to train and evaluate them. Please refer to the individual `README.md` files in the `docker` subdirectory on how to setup each individual method.


Currently, we use the configuration file provided in `conf` to setup PostgreSQL. If you want to change the configuration, you can directly edit the file before installing PostgreSQL.


## Methods

Information about setting up and running all the methods included in this codebase can be found in the corresponding READMEs of the docker directories.

> **Note:** The PostgreSQL baseline is run from the Bao codebase using the `bao/run_test_queries.py` with the parameter `--use_postgres`.


# Citations

Since we include the code bases from recent publications, please make sure to also include their citations. We thank the authors of the previous work for making their research available:

>Marcus, Ryan, et al. "**Neo: A Learned Query Optimizer.**" Proceedings of the VLDB Endowment 12.11.

>Marcus, Ryan, et al. "**Bao: Making learned query optimization practical.**" Proceedings of the 2021 International Conference on Management of Data. 2021.

>Yang, Zongheng, et al. "**Balsa: Learning a Query Optimizer Without Expert Demonstrations.**" Proceedings of the 2022 International Conference on Management of Data. 2022.

<!-- >Chen, Xu, et al. "**LEON: a new framework for ml-aided query optimization.**" Proceedings of the VLDB Endowment 16.9 (2023): 2261-2273. -->

>Yu, Xiang, et al. "**Cost-based or learning-based? A hybrid query optimizer for query plan selection.**" Proceedings of the VLDB Endowment 15.13 (2022): 3924-3936.

Additionally, we use the Join Order Benchmark published by Leis et al.:

>Leis, Viktor, et al. "**How good are query optimizers, really?.**" Proceedings of the VLDB Endowment 9.3 (2015): 204-215.

And the STACK benchmark published by Marcus et al.:

>Marcus, Ryan, et al. "**Bao: Making learned query optimization practical.**" Proceedings of the 2021 International Conference on Management of Data. 2021.
