#contains a few types of categorical feature encoding
#1.one-hot encoding, by using OneHotEstimator
#2.label encoding, just by using stringIndexer, ok for ordinal but not for norminal features
#3.


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

