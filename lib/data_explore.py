import pyspark.mllib.stat as st
import numpy as np

def get_basic_features_stats(sdf,num_cols,cat_cols):
    '''
    get basic stats for all features
    generate mean, variance for each numerical features
    generate top categories for each cat features
    input: 
    * spark df
    * num_cols, cat_cols: list of str
    '''
    #basic stats for numerical cols
    numeric_rdd = sdf.select(num_cols).rdd.map(lambda row: [e for e in row])

    mllib_stats = st.Statistics.colStats(numeric_rdd)

    for col, m, v in zip(num_cols,mllib_stats.mean(),mllib_stats.variance()):
        print('{0}: \t{1:.2f} \t {2:.2f}'.format(col, m, np.sqrt(v)))

    #basic stats for categorical cols
    categorical_rdd = dataset.select(cat_cols).rdd.map(lambda row: [e for e in row])

    for i, col in enumerate(cat_cols):
        agg = categorical_rdd.groupBy(lambda row: row[i]).map(lambda row: (row[0], len(row[1])))
        print(col, sorted(agg.collect(),key=lambda el: el[1],reverse=True))
    
    return None