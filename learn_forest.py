import utils as ut
import pdb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error

def learn_forest(filepath):
    # Learning a forest for predicting PMN
    # Please choose PMN, MAC, LDH, TP
    feature = 'PMN'
    # For cross-validation, we can exclude certain studies from training. For
    # example to exclude Pauluhun,2010 use
    #author_exclude = ['Pauluhn, J.',2010]
    # To use the entire training data, pass author_exclude as None
    author_exclude = ['Shvedova, A. et al.',2005]
    
    # Getting training input and output
    (train_inp,train_out,test_inp,test_out) = ut.prepare_data_rf(filepath,\
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

if __name__=="__main__":
    filepath = './data/Carbon_Nanotube_Pulmonary_Toxicity_Data_Set_20120313.xls'
    data = learn_forest(filepath)
