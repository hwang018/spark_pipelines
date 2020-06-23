#contains 4 types of feature removal
#1.hard-coded feature remover
#2.remove features that's not to the standard (e.g. too few/much info), no variation... too much missing values...
#3.check pearson correlation, and chi-squared feature selection
#4.using model to select features, RF or lasso feature selection

#start:

#1.hard-coded feature remover
#input:sparkdf,config (contains what names of features to remove)
#output:selected_features,dropped features,selected columns spark dataframe

def hard_coded_feature_remover(sdf,config):
    '''
    find features to remove, according to input config
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

#2.remove features that's not to the standard (e.g. too few/much info), no variation... too much missing values...
# for categorical columns, do cardinality test, for numerical features, calculate correlation

def cat_col_cardinality_test(sdf,config):
    '''
    input: spark df, config.max_cat, config.min_cat
    for categorical columns, cardinality must fall in min max range to be retained
    if not specified, will not have lower/upper bound
    '''

def num_cols_correlation_test(sdf,config):
    '''
    input: spark df, config.corr_thres
    if above corr_thres, column will be excluded
    consider using chi-square test on cat features
    '''
    
    
#multi-colinearlity test, drop highly correlated features, using 2 metrics, pearson correlation
#and chi-square 


    
