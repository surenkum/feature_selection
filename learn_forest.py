import utils as ut
import pdb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def learn_forest(filepath):
    # Learning a forest for predicting PMN
    # Please choose PMN, MAC, LDH, TP
    feature = 'PMN'
    # For cross-validation, we can exclude certain studies from training. For
    # example to exclude Pauluhun,2010 use
    #author_exclude = ['Pauluhn, J.',2010]
    #author_exclude = ['Shvedova, A. et al.',2005]
    # To use the entire training data, pass author_exclude as None
    author_exclude = None
    
    # Getting training input and output
    (train_inp,train_out,test_inp,test_out,feature_names) = ut.prepare_data_rf(filepath,\
            feature,author_exclude)

    # Training
    # Imputing all the NaN values
    estimator = Pipeline([("imputer",Imputer(strategy="mean")),
        ("forest",ExtraTreesRegressor(random_state=0))])
    estimator.fit(train_inp,train_out)

    # Testing the model against validation if it exists or else calculating
    # error on the training input itself
    # See if have some test samples
    if test_out.shape[0]>0:
        predict_test = estimator.predict(test_inp)
        # Estimating MSE score
        score = mean_squared_error(test_out,predict_test)
        print "MSE error for ",feature," after excluding ",author_exclude, "is : ",score
    else:
        predict_test = estimator.predict(train_inp)
        # Estimating MSE score
        score = mean_squared_error(train_out,predict_test)
        print "MSE error for ",feature," with all points in the model is : ",score
    
    # Plotting feature importance
    feature_importance = estimator._final_estimator.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance for feature '+feature)
    plt.show()

if __name__=="__main__":
    filepath = './data/Carbon_Nanotube_Pulmonary_Toxicity_Data_Set_20120313.xls'
    data = learn_forest(filepath)
