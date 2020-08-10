'''
contains 2 types of tree based classification model from spark ml
1.Random Forest
2.Gradient Boosted Trees
Both comes with param tuning using grid search
'''
from pyspark.ml.classification import GBTClassifier,RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import lib.categorical_handler as ctgy
from pyspark.ml import Pipeline

##################################### RF, with tuning process ################################

def train_rf_param_tuning(params,train_df,test_df,num_cols,cat_cols,cv_Folds = 5):
    # label encoding on cat cols, assemble, creating col feature
    rf_tune_grid = params['RF']
    print("received rf param grid: %s"%rf_tune_grid)
    
    stages_rf = ctgy.assemble_into_features_RF(train_df,num_cols,cat_cols,'_index')

    partialPipeline = Pipeline().setStages(stages_rf) # rf stages involves only label encoding
    
    pipelineModel = partialPipeline.fit(train_df)

    print("transforming train df")
    train_df_transformed = pipelineModel.transform(train_df)
    
    print("transforming test df")
    test_df_transformed = pipelineModel.transform(test_df)
    
    # Create an initial RandomForest model.
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.maxDepth, rf_tune_grid['maxDepth'])
                 .addGrid(rf.maxBins, rf_tune_grid['maxBins'])
                 .addGrid(rf.numTrees, rf_tune_grid['numTrees'])
                 .build())

    # Evaluate model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

    cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=cv_Folds)
    
    print("fitting cv rf models on transformed train df")
    cvModel = cv.fit(train_df_transformed)
    bestModel = cvModel.bestModel

    test_predictions = bestModel.transform(test_df_transformed)
    print('best model performance on test set:')
    areaUnderROC = evaluator.setMetricName("areaUnderROC").evaluate(test_predictions)
    areaUnderPR = evaluator.setMetricName("areaUnderPR").evaluate(test_predictions)
    print('AUC: %s, AUPR: %s'%(areaUnderROC,areaUnderPR))
    print('return best rf model')
    
    return bestModel


##################################### Gradient Boosted Trees, with tuning process ################################

def train_gbt_param_tuning(params,train_df,test_df,num_cols,cat_cols,cv_Folds = 5):
    # label encoding on cat cols, assemble, creating col feature
    gbt_tune_grid = params['GBT']
    print("received gbt param grid: %s"%gbt_tune_grid)
    
    # using the same encoding and assembler stages as rf
    stages_gbt = ctgy.assemble_into_features_RF(train_df,num_cols,cat_cols,'_index')

    partialPipeline = Pipeline().setStages(stages_gbt) # gbt stages involves only label encoding
    
    pipelineModel = partialPipeline.fit(train_df)

    print("transforming train df")
    train_df_transformed = pipelineModel.transform(train_df)
    
    print("transforming test df")
    test_df_transformed = pipelineModel.transform(test_df)
    
    # Create an initial RandomForest model.
    gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

    paramGrid = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, gbt_tune_grid['maxDepth'])
                 .addGrid(gbt.maxBins, gbt_tune_grid['maxBins'])
                 .build())

    # Evaluate model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

    cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=cv_Folds)
    
    print("fitting cv gbt models")
    cvModel = cv.fit(train_df_transformed)
    bestModel = cvModel.bestModel

    test_predictions = bestModel.transform(test_df_transformed)
    print('best model performance on test set:')
    areaUnderROC = evaluator.setMetricName("areaUnderROC").evaluate(test_predictions)
    areaUnderPR = evaluator.setMetricName("areaUnderPR").evaluate(test_predictions)
    print('AUC: %s, AUPR: %s'%(areaUnderROC,areaUnderPR))
    print('return best gbt model')
    
    return bestModel