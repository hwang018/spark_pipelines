#####
##### util.py stores all small tools in the pipeline process
##### 
import findspark
findspark.init()
import pyspark    
import pandas as pd
import numpy as np
import pickle
import gc
import json
import os

from imblearn.over_sampling import SMOTE

import pyspark.sql.functions as F
from pyspark.sql.functions import col, countDistinct, when, row_number
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.sql import SparkSession,SQLContext

#create some entry points
sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

def equivalent_type(f):
    '''
    can add more spark sql types like bigint ...
    '''
    if f == 'datetime64[ns]': return DateType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)

# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
        struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)