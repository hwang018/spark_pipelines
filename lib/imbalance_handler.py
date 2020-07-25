import random
import numpy as np
from sklearn import neighbors
from pyspark.sql import Row
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col,when,concat,substring,lit,udf,lower,sum as ps_sum,count as ps_count,row_number
from pyspark.sql.window import *

############################## random down sampling ##########################

def spark_df_down_sampling(sdf, desired_major_to_minor_ratio, label_col, major_class_val = 0, seed = 52):
    """
    Downsample majority class to get desired major to minor ratio, only accepts binary classification
    inputs:
        * sdf: spark df before feature selection
        * desired_major_to_minor_ratio (int)
        * label_col: col name for label in spark df
        * major_class_val (int): The label for majority class. 0 for majority, 1 for minority by default
        * seed: for random function
    output:
        * downsampled_spark_df: the spark df after downsampling majority class.
    """
    # current distribution of 2 classes, 0 for major, 1 for minor by default
    minor_class_val = 1 - major_class_val
    class_count = dict(sdf.groupBy(col(label_col)).count().collect())
    
    # current ratio is upper bound to desired_major_to_minor_ratio
    current_maj_to_min_ratio = int(float(class_count[major_class_val])/float(class_count[minor_class_val]))

    # check validity in desired ratio
    if current_maj_to_min_ratio > desired_major_to_minor_ratio:
        # need to apply downsample
        desired_maj_samples =  desired_major_to_minor_ratio * class_count[minor_class_val]
        # set seed
        np.random.seed(seed)
        w = Window().orderBy(label_col)
        # index to differentiate pos/neg samples, all rows to be indexed
        sdf = sdf.withColumn("randIndex", when(sdf[label_col] == major_class_val, row_number().over(w)).otherwise(-1))

        selected_sample_index = np.random.choice(class_count[major_class_val], desired_maj_samples, replace=False).tolist()

        sdf_sampled = sdf.filter(sdf['randIndex'].isin(selected_sample_index) | (sdf['randIndex'] < 0)).drop('randIndex')

        print("After downsampling \"{0}\": label distribution is {1}".format(label_col,sdf_sampled.groupBy(col(label_col)).count().collect()))

        return sdf_sampled
    else:
        # provided desired ratio is too large and exceed total number of majority rows
        print("Desired ratio is too large and no downsampling performed, return input dataframe.")
        return sdf
    
############################## smote up sampling ##########################

def __smote_single_query():
    '''
    single step to generate up to k synthetic samples per minority point
    using spark lsh algo to find nearest k points, randomly choose one neighbour and create one synthetic sample
    keep track of stringIndexed columns (these are original cat cols)
    '''
    
    
def spark_df_smote_sampling(sdf, desired_major_to_minor_ratio, num_col_indices, str_col_indices, label_col, major_class_val = 0, seed = 52):
    #WIP: main difficulty: doing smote on categorical columns
    return None
