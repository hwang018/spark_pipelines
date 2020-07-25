import findspark
#spark path using default value
findspark.init()
import pyspark
import pickle
import gc
import json
import os

import numpy as np
import pandas as pd
import datetime 
import datetime as dt
from datetime import datetime,timedelta,date
from datetime import date

import dateutil
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse

from imblearn.over_sampling import SMOTE

import pyspark.sql.functions as F
from pyspark.sql.functions import col, countDistinct, when, row_number
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.sql import SparkSession,SQLContext

######################## convert pandas to spark df ###############################

def equivalent_type(f):
    '''
    add more spark sql types like bigint ...
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

def pandas_to_spark(sqlcontext,pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
        struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlcontext.createDataFrame(pandas_df, p_schema)

######################## convert spark df to pandas df, faster than toPandas() ############

def _map_to_pandas(rdds):
    """ Needs to be here due to pickling issues """
    return [pd.DataFrame(list(rdds))]

def toPandas(df, n_partitions=None):
    """
    Returns the contents of 'df' as a local 'pandas.DataFrame' in a speedy fashion. The DataFrame is
    repartitioned if 'n_partitions' is passed.
    :param df:              pyspark.sql.DataFrame
    :param n_partitions:    int or None
    :return:                pandas.DataFrame
    """
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand

######################## get num and cat features spark df ###############################

def get_num_cat_feat(input_spark_df, exclude_list=[]):
    """
    desc: return cat and num features list from a spark df, a step before any encoding on cat features
    inputs:
        * input_spark_df: the input spark dataframe to be checked.
        * exclude_list (list of str): the excluded column name list, which will be excluded for the categorization.
    output:
        * numeric_columns (list of str): the list of numeric column names.
        * string_columns (list of str): the list of categorical column names.
    """
    timestamp_columns = [item[0] for item in input_spark_df.dtypes if item[1].lower().startswith(('time', 'date'))]

    # categorize the remaining columns into categorical and numeric columns
    string_columns = [item[0] for item in input_spark_df.dtypes if item[1].lower().startswith('string') \
                                and item[0] not in exclude_list+timestamp_columns]
    
    numeric_columns = [item[0] for item in input_spark_df.dtypes if item[1].lower().startswith(('big', 'dec', 'doub','int', 'float')) \
                                and item[0] not in exclude_list+timestamp_columns]
    
    # check whether all the columns are covered
    all_cols = timestamp_columns + string_columns + numeric_columns + exclude_list

    if len(set(all_cols)) == len(input_spark_df.columns):
        print("All columns are been covered.")
    elif len(set(all_cols)) < len(input_spark_df.columns):
        not_handle_list = list(set(input_spark_df.columns)-set(all_cols))
        print("Not all columns are covered. The columns missed out: {0}".format(not_handle_list))
    else:
        mistake_list = list(set(all_cols) - set(input_spark_df.columns))
        print("The columns been hardcoded wrongly: {0}".format(mistake_list))

    return numeric_columns, string_columns

######################## datetime related ###############################

def pandasdate_maker(intdate):
    '''
    Parses an integer date like 20180819 and return its pandas date format
    Dependency: datetime lib
    Input: integer date, like 20180819
    Output: pandas date format '2018-08-19'
    '''
    strdate = str(intdate)[0:8]
    pd_date = pd.to_datetime(strdate, format='%Y%m%d', errors='coerce')
    return pd_date

def relative_days(intwkend,deltadays = -7):
    try:
        strend = datetime.strptime(str(intwkend)[0:8], "%Y%m%d") #string to date 
        startday = strend + timedelta(deltadays) # date - days
        intstartday = int(startday.strftime('%Y%m%d'))
    except:
        intstartday = float('NaN')
    return intstartday

def last_day(d, day_name):
    '''
    All are helper functions to generate proper sectionalized periods
    Get the last weekday, given current date and which desired last weekday
    '''
    days_of_week = ['sunday','monday','tuesday','wednesday',
                        'thursday','friday','saturday']
    target_day = days_of_week.index(day_name.lower())
    delta_day = target_day - d.isoweekday()
    if delta_day >= 0: delta_day -= 7 # go back 7 days
    return d + timedelta(days=delta_day)

def get_date_pairs(start_date_int,end_date_int):
    '''
    given start and end dates (int), return pairs of dates in pandas format
    '''
    #Make integer dates in pandas format
    start_period = pandasdate_maker(start_date_int)
    end_period = pandasdate_maker(end_date_int)
    weekly_segmented_dates = pd.date_range(start_period, end_period, freq='W')

    #list of starting date, by default it's every Sunday of a week
    list_of_weekly_dates_start = list(weekly_segmented_dates.to_pydatetime())
    #list of ending date, by default it's every Saturday of a week
    list_of_weekly_dates_end =  [x+ timedelta(6) for x in list_of_weekly_dates_start]

    i = 0
    date_pairs = []
    print("loaded date pairs:")
    while i < len(list_of_weekly_dates_start):
        date_pair = (int(str(list_of_weekly_dates_start[i])[:10].replace("-", "")),\
                     int(str(list_of_weekly_dates_end[i])[:10].replace("-", "")))
        date_pairs.append(date_pair)
        print("Week"+ str(i+1) + ": " + str(list_of_weekly_dates_start[i])[:10].replace("-", "") + ',' \
              + str(list_of_weekly_dates_end[i])[:10].replace("-", ""))
        i+=1
        
    return date_pairs

def week_periods_generator(rundate,number_of_periods=4):
    '''
    Small wrapper to generate the list of periods for constructing sql
    '''
    #input the current/rundate and convert into pandas date
    current_rundate = pandasdate_maker(rundate) #convert into pandas date
    #use helper func to get current date's last weekday
    last_saturday = last_day(current_rundate,'saturday')
    last_saturday_int = int(str(last_saturday)[:10].replace("-",""))
    
    #get the list of date pairs, from far away (deltadays serves as a dynamic cap of the periods)
    #and only taking the latest number_of_periods weeks
    list_of_datepairs = get_date_pairs(relative_days(intwkend=last_saturday_int,deltadays=-8*number_of_periods),last_saturday_int)
    list_of_datepairs_final = list_of_datepairs[-number_of_periods:]
    
    return list_of_datepairs_final


def period_days_generator(datetuple):
    '''
    convenience function to generate a list of continuous days
    within a given period [sdate,edate]
    '''
    sdate = datetuple[0]
    edate = datetuple[1]
    syyyy = int(str(sdate)[0:4])
    smm = int(str(sdate)[4:6])
    sdd = int(str(sdate)[6:])

    eyyyy = int(str(edate)[0:4])
    emm = int(str(edate)[4:6])
    edd = int(str(edate)[6:])

    sdate = date(syyyy, smm, sdd)   # start date
    edate = date(eyyyy, emm, edd)   # end date

    delta = edate - sdate       # as timedelta

    days_list = []

    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        days_list.append(int(str(day).replace('-', '')))

    return days_list

def get_month(yyyyMM):
    """
    desc: The function to get the month from yyyyMM.
    inputs:
        * yyyyMM (str): the month/year, e.g. "201809"
    returns:
        * month (str): the month, e.g. "9" for yyyyMM = "201809"
    """
    if yyyyMM[-2] == '0':
        month = yyyyMM[-1]
    else:
        month = yyyyMM[-2:]
    return month

def get_prev_month(yyyyMM):
    """
    desc: The function to get the previous month from yyyyMM.
    inputs:
        * yyyyMM (str): the month/year, e.g. "201809"
    returns:
        * prev_month_output (str): the month/year, e.g. "201808" for yyyyMM = "201809"
    """
    prev_month = int(yyyyMM[4:])-1
    if prev_month == 0:
        prev_month = 12
        year = str(int(yyyyMM[:4])-1)
    else:
        year = yyyyMM[:4]

    prev_month = str(prev_month)
    if len(prev_month) == 1:
        prev_month = '0'+prev_month

    prev_month_output = year + prev_month
    return prev_month_output