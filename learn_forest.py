import utils as ut
import pdb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.tree import export_graphviz

'''
Main function that learns a random forest based on toxicity data
Inputs: filepath: path to .xls file 
        toxicity: Type of toxicity we are analyzing, defaults to CNT
'''
def learn_forest(filepath,toxicity="CNT"):
    # Learning a forest for predicting PMN
    # Please choose PMN, MAC, LDH, TP, TCC (Total Cell Count -- only for AgNP)
    features = ['PMN','MAC','LDH','TP','TCC']
    for feature in features:
        # For cross-validation, we can exclude certain studies from training. For
        # example to exclude Pauluhun,2010 use
        # To use the entire training data, pass author_exclude as None
        author_exclude = None#[['Seiffert J',2015],['Silva R',2015]]#None
        particle_exclude = [{'Particle Type (1=basic, 2 = citratecapped, 3 = PVPcapped)':1}]
        
        # Getting training input and output
        (train_inp,train_out,test_inp,test_out,feature_names) = ut.prepare_data_rf(filepath,\
                feature,author_exclude,toxicity = toxicity,
                other_excludes = particle_exclude)

        # Get median values for plotting dose response curves
        median_vals = get_median(train_inp)

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
        # Exporting the learned graph
        feature_string = np.array(['Particle Type','Mean Diameter, nm','Exposure Mode',
                'Rat Species','Mean Animal Mass, g','Sex','Surface Area (m^2/g)',
                'Mass Conc. (ug/m^3)','Exp. Hours','Total Dose (ug/kg)',
                'Post Exp. (days)'])
        print "original feature names ",feature_names
        print "replaced feature names ",feature_string
        # Increase font size for plots
        matplotlib.rcParams.update({'font.size': 12})

        # Print all the estimators
        for ind,em in enumerate(estimator._final_estimator.estimators_):
            export_graphviz(em,out_file="tree"+str(ind)+".dot",feature_names = feature_string)
        
        # Plotting feature importance
        feature_importance = estimator._final_estimator.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, feature_string[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance for feature '+feature)
        plt.show()

def get_median(train_inp):
    # Getting the median values of the entire data
    median_vals = np.zeros((train_inp.shape[1],))
    for i in range(train_inp.shape[1]):
        median_vals[i] = np.median(train_inp[~np.isnan(train_inp[:,i]),i])
        pdb.set_trace()

if __name__=="__main__":
    # filepath = './data/Carbon_Nanotube_Pulmonary_Toxicity_Data_Set_20120313.xls'
    filepath = './data/Toxicity Measurements -- Meta Analysis.xlsx'
    toxicity = "AgNP" # Use "CNT" for analyzing Carbon Nano toxicity
    data = learn_forest(filepath,toxicity)
