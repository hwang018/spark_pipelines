import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from distutils.version import LooseVersion
from pyspark.ml.feature import StandardScaler,MinMaxScaler
from pyspark.ml.feature import StringIndexer, VectorAssembler


#contains a few types of categorical feature encoding
#1.one-hot encoding, by using OneHotEstimator
#2.label encoding, just by using stringIndexer, ok for ordinal but not for norminal features

########################### high cardinality, group tails ###########################

def get_retained_categories(sdf,cat_cols,max_cat):
    '''
    for each cat col, find all items count,
    take top max_cat values and return the categories per col that are retained
    input: *spark df, cat_cols to investigate, max_cat allowed (cardinality)
    output: *res, a dict storing each cat col's retained items
    '''
    #basic stats for categorical cols
    categorical_rdd = sdf.select(cat_cols).rdd.map(lambda row: [e for e in row])

    res = {}
    for i, col in enumerate(cat_cols):
        agg = categorical_rdd.groupBy(lambda row: row[i]).map(lambda row: (row[0], len(row[1])))
        
        item_count = sorted(agg.collect(),key=lambda el: el[1],reverse=True)
        
        counts_all = [c[0] for c in item_count]
        
        kept_cat = []
        
        if len(item_count) > max_cat:
            #too many categories in this cat col, need to group tails to others
            kept_tuples = item_count[:max_cat]
            #store top frequent items
            kept_cat = [c[0] for c in kept_tuples]
            res[col] = kept_cat
        else:
            res[col] = counts_all
    return res


def get_cat_col_cardinality(sdf,cat_cols):
    '''
    generate top categories for each cat features
    input: 
    * spark df
    * num_cols, cat_cols: list of str
    func is sql.functions
    '''
    #cat_info_dict to store index of cat feature and its cardinality, for modelling input
    cat_info_dict = {}

    print('generating cardinality map for cat cols')
    for i, col in enumerate(cat_cols):
        cat_info_dict[i] = sdf.select(func.countDistinct(col).alias("distinct_count_%s"%col)).collect()[0][0]
    
    return cat_info_dict

########################### string indexer ###########################

def str_index_cat_cols(sdf,cat_cols,cat_index_suffix):
    '''
    only to stringIndex cols (per item frequency), no encoding applied
    input:
        * sdf: spark df
        * cat_cols: cat cols to be string indexed
        * cat_cols_affix: output affix to indexed cat cols
        * stages: input stages (from any previous stages)
    output:
        * stages: modified stages for spark pipeline
    '''
    stages = []
    
    for categoricalCol in cat_cols:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + cat_index_suffix)
        stages+=[stringIndexer]
        
    return stages

########################### OHE encoder ###########################

def one_hot_encode_cat_cols(sdf,cat_cols,cat_encode_suffix):
    '''
    perform one hot encoding for cat_cols 
    input:
    * sdf: spark df
    * cat_cols: categorical columns already str indexed
    output:
    * stages
    '''
    stages = []

    for categoricalCol in cat_cols:
        # Category Indexing with StringIndexer, will encode to numerical according to frequency, highest frequency will be encoded to 0
        # when applying this stringIndexer onto another dataset and encounter missing encoded value, we can throw exception or setHandleInvalid(“skip”)
        # like indexer.fit(df1).setHandleInvalid("skip").transform(df2), will remove all rows unable to encode    
        # no indexing applied
        # stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")

        # Use OneHotEncoder to convert categorical variables into binary SparseVectors，
        # binary sparse vectors like (2,[0],[1.0]) means a vector of length 2 with 1.0 at position 0 and 0 elsewhere.
        # spark OHE will automatically drop the last category, you can force it not to drop by dropLast=False
        # it omits the final category to break the correlation between features

        # column is already indexed, with suffix _index as default
        if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
            from pyspark.ml.feature import OneHotEncoderEstimator
            encoder = OneHotEncoderEstimator(inputCols=[categoricalCol], outputCols=[categoricalCol + cat_encode_suffix])
        else:
            from pyspark.ml.feature import OneHotEncoder
            encoder = OneHotEncoder(inputCols=[categoricalCol], outputCols=[categoricalCol + cat_encode_suffix])
        # Add stages.  These are not run here, but will run all at once later on.
        stages += [encoder]
            
    return stages


def assemble_into_features_OHE(sdf,num_cols,cat_cols,cat_index_suffix,cat_encode_suffix):
    '''
    assemble all features into vector
    cat_cols with suffix
    num cols
    input:
    * processed cat cols affix
    '''
    # to combine all the feature columns into a single vector column. 
    # This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.
    # Transform all features into a vector using VectorAssembler
    
    # empty stage holder
    stages_ = []
    
    # perform str indexing
    str_ind_stages = str_index_cat_cols(sdf,cat_cols,cat_index_suffix)

    stages_ += str_ind_stages
    
    indexed_catcols = [a+cat_index_suffix for a in cat_cols]
    # perform OHE encoding
    
    ohe_stages = one_hot_encode_cat_cols(sdf,indexed_catcols,cat_encode_suffix)
    stages_ += ohe_stages
    
    encoded_catcols = [a+cat_encode_suffix for a in indexed_catcols]
    
    assemblerInputs = num_cols+encoded_catcols
    
    #VectorAssembler only applied to numerical or transformed categorical columns
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages_ += [assembler]

    # then we apply scaling on the vectorized features, 2 additional params are:
    # withStd: True by default. Scales the data to unit standard deviation.
    # withMean: False by default. Centers the data with mean before scaling.
    # scaler = StandardScaler(inputCol="features", outputCol="scaled_features",withMean=True)
    #scaler = MinMaxScaler(min=0, max=1, inputCol='features', outputCol='features_minmax')

    #stages += [scaler] 
    return stages_

def assemble_into_features_RF(sdf,num_cols,cat_cols,cat_index_suffix):
    '''
    assemble all features into vector
    without encoding, just label indexed
    input:
    * processed cat cols affix
    '''
    stages_ = []
    # perform str indexing
    str_ind_stages = str_index_cat_cols(sdf,cat_cols,cat_index_suffix)
    stages_ += str_ind_stages
    # save new names for indexed cat cols
    indexed_catcols = [a+cat_index_suffix for a in cat_cols]
    
    assemblerInputs = num_cols+indexed_catcols
    
    #VectorAssembler only applied to numerical or transformed categorical columns
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages_ += [assembler]

    return stages_