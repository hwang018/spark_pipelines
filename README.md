# spark_pipelines
A collection of general tools for general binary classification tasks using spark.

Structure of this spark pipeline tool:
lib/
1. categorical_handler.py : provides common methods to encode categorical columns.
2. feature_selection.py : provides 4 types of feature selection: hard_code_remover, chisquare test, model based selection..
3. imbalance_handler.py : contains 3 functions: random down sampler, smote oversampler, overall driver for any sampling desired.
4. data_explore.py : computes basic stats on num and cat columns.
5. util.py : contains utility functions for both spark and pandas df.
6. plot_metrics.py : plot modelling results in multiple setups, training/validation...
7. modelling.py : contains data science part, spark ml and sklearn toolkits.
8. logger.py : for logging purposes.
