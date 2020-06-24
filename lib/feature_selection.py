import pyspark.mllib.stat as st
import pyspark.mllib.linalg as ln
import numpy as np
import random

'''
contains 5 types of feature removal
1.hard-coded feature remover
2.remove features that's not to the standard (e.g. too few/much info), no variation... too much missing values...
3.using model to select features, lgbm or lasso feature selection
'''

####################### type 1: hard-coded feature remover #################

def hard_coded_feature_remover(sdf,config):
    '''
    find features to remove, according to input config
    input:sparkdf,config (contains what names of features to remove)
    output:selected_features,dropped features,selected columns spark dataframe
    '''
    cols_nm_remove = config.cols_remove
    must_keep_cols = config.must_keep_cols
    input_cols = sdf.columns
    
    retain_cols = []
    removed_cols = []
    
    for col in input_cols:
        for rm in cols_nm_remove:
            if rm.lower() in col.lower() and col not in must_keep_cols:
                #remove this column
                removed_cols.append(col)
            else:
                retain_cols.append(col)
                
    retained_sdf = sdf.select(*retain_cols)
    
    return retain_cols,removed_cols,retained_sdf

####################### type 2: remove categorical cols that fails chi-square test #################

def chi_square_cat_feature_selector(sdf,target_col,cat_cols,chi_square_thres=0.05):
    '''
    chi-square test on cat cols, find irrelevant cat cols
    input:
        * sdf: spark df
        * target_col: the prediction target column
        * cat_cols: the cat cols for chi-square test
        * chi_square_thres: critical value chosen for chi-square test, default 0.05
    output:
        * chi_square_drop_cols: cat cols that are independent of the target
    '''
    chi_square_drop_cols = []
    
    #simple chi-squared test to filter out columns that are meaningless to target
    for cat in cat_cols[1:]:
        agg = sdf.groupby(target_col).pivot(cat).count()
        agg_rdd = agg.rdd.map(lambda row: (row[1:])) \
          .flatMap(lambda row:[0 if e == None else e for e in row]).collect()

        row_length = len(agg.collect()[0]) - 1
        agg = ln.Matrices.dense(row_length, 2, agg_rdd)
        test = st.Statistics.chiSqTest(agg)
        
        if round(test.pValue, 4) >= chi_square_thres:
            #independent col
            chi_square_drop_cols.append(cat)
    print("cols to drop after chi-square test:%s"%chi_square_drop_cols)
    
    return chi_square_drop_cols


####################### type 3: remove too high cardinality cat cols #################

def cat_col_cardinality_test(spark_df, cat_columns, mini=2, maxi=100):
    """
    desc:  The coverage test for categorical features, which make sure the number of categorical levels for categorical featues to be
    larger than or equal to mini, and smaller than or equal to maxi.
    inputs:
        * spark_df: the input spark dataframe to be checked.
        * cat_columns (list of str): the list of categorical column names to be checked.
        * mini (int, optional): the minimum number of categorical levels defined for categorical features.
        * maxi (int, optional): the maximum number of categorical levels defined for categorical features.
    returns:
        * final_count_df: the pandas dataframe which store the number of categorical levels for categorical featues.
        * no_info_col (list of str): the list of column names with number of categorical levels less than mini.
        * high_nums_col (list of str): the list of column names with number of categorical levels larger than maxi.
    """
    print("Start the count computation for categorical features...")
    print("The no. of categorical features: {0}".format(str(len(cat_columns))))

    final_count_df = pd.DataFrame()
    count =1

    for col in cat_columns:
        count_df = spark_df.agg(countDistinct(col).alias("count")).toPandas()
        count_df.index = [col]
        if final_count_df.empty:
            final_count_df = count_df.copy()
        else:
            final_count_df = final_count_df.append(count_df)
        
        del count_df
        gc.collect()
        count +=1

    no_info_df = final_count_df[final_count_df['count']<mini]
    no_use_tuple = [(x, y) for x, y in zip(list(no_info_df.index), list(no_info_df['count']))]

    high_nums_df = final_count_df[final_count_df['count']>maxi]
    high_nums_tuple = [(x, y) for x, y in zip(list(high_nums_df.index), list(high_nums_df['count']))]

    no_info_col = no_info_df.index.values.tolist()
    high_nums_col = high_nums_df.index.values.tolist()

    return final_count_df, no_info_col, high_nums_col

####################### type 4: remove highly correlated num cols #################

def num_cols_correlation_test(sdf,num_cols,corr_thres,must_keep_cols=[]):
    '''
    input: spark df, threshold of correlation
    pearson correlation used
    '''
    numeric_rdd = sdf.select(num_cols).rdd.map(lambda row: [e for e in row])
    #get pearson correlation matrix
    corrs = st.Statistics.corr(numeric_rdd)
    
    #find correlated pairs
    correlated_pairs = []
    
    for i, el in enumerate(corrs > corr_thres):
        correlated = [(num_cols[j], corrs[i][j]) for j, e in enumerate(el) if e == 1.0 and j != i]
        if len(correlated) > 0:
            for e in correlated:
                correlated_pairs.append(sorted([num_cols[i], e[0]]))
    
    correlated = [','.join(a) for a in correlated_pairs]
    res = list(set(correlated))
    res_list = [a.split(',') for a in res]
    
    #now have all unique pair of correlated features
    #just choose randomly
    col_to_drop = []
    
    for pair in res_list:
        target = random.choice(pair)
        if target in must_keep_cols:
            continue
        else:
            col_to_drop.append(target)
            
    return col_to_drop
    
####################### type 5: model based feature selection #################
