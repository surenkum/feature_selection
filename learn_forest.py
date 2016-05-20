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
        print "Processing the data to estimate ",feature
        # For cross-validation, we can exclude certain studies from training. For
        # example to exclude Pauluhun,2010 use
        # To use the entire training data, pass author_exclude as None
        author_exclude = None#[['Seiffert J',2015],['Silva R',2015]]#None
        particle_exclude = None#[{'Particle Type (1=basic, 2 = citratecapped, 3 = PVPcapped)':1}]
        
        # Getting training input and output
        (train_inp,train_out,test_inp,test_out,feature_names) = ut.prepare_data_rf(filepath,\
                feature,author_exclude,toxicity = toxicity,
                other_excludes = particle_exclude)

        # Get median values for plotting dose response curves
        (median_vals, min_vals, max_vals) = get_median_min_max(train_inp)

        # Training
        # Imputing all the NaN values
        estimator = Pipeline([("imputer",Imputer(strategy="mean")),
            ("forest",ExtraTreesRegressor(random_state=0))])
        estimator.fit(train_inp,train_out)

        # Plotting risk-contour curves
        print feature_names
        feature_indexes = [1,7]
        plot_risk_contour(estimator,median_vals,min_vals,max_vals,\
                feature_indexes,feature_names,feature)

        # Plotting dose-response curves
        # Testing for nano-particle size
        feature_index = 1
        feature_vals = [20,100] # Passing 20 and 100 nanometers
        plot_dose_response(estimator,median_vals,min_vals, max_vals, \
                feature_index,feature_vals,feature_names,feature)

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

# Get the median, minimum and maximum values of all the input dimensions
def get_median_min_max(train_inp):
    # Getting the median values of the entire data
    median_vals = np.zeros((train_inp.shape[1],))
    min_vals = np.zeros((train_inp.shape[1],))
    max_vals = np.zeros((train_inp.shape[1],))
    for i in range(train_inp.shape[1]):
        median_vals[i] = np.median(train_inp[~np.isnan(train_inp[:,i]),i])
        min_vals[i] = np.nanmin(train_inp[:,i])
        max_vals[i] = np.nanmax(train_inp[:,i])
    return (median_vals,min_vals,max_vals)

'''
Requires the estimator, median values, two different feature indexes 
to plot risk contours curves
'''
def plot_risk_contour(estimator,median_vals,min_vals,max_vals,\
        feature_indexes,feature_names,target_feature):
    assert (len(feature_indexes)==2), "Need 2 feature indexes to plot risk contour"
    # Plotting all the output values from the curve
    # Divide the minimum and maximum values in 20 points range
    origin = 'lower'
    cmap = plt.cm.get_cmap("rainbow")
    cmap.set_under("magenta")
    cmap.set_over("yellow")
    num_points  = 20
    x = np.arange(min_vals[feature_indexes[0]],max_vals[feature_indexes[0]],\
            (max_vals[feature_indexes[0]]-min_vals[feature_indexes[0]])/(num_points*1.0))
    y = np.arange(min_vals[feature_indexes[1]],max_vals[feature_indexes[1]],\
            (max_vals[feature_indexes[1]]-min_vals[feature_indexes[1]])/(num_points*1.0))
    X,Y = np.meshgrid(x,y)
    Z = np.zeros((X.shape))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Get current input values
            inp_feature = median_vals
            inp_feature[feature_indexes[0]] = X[i,j] # fill-in first feature
            inp_feature[feature_indexes[1]] = Y[i,j] # fill-in second feature
            Z[i,j] = estimator.predict(inp_feature.reshape(1,-1))[0]
    # Plotting the contour
    plt.figure()
    CS = plt.contourf(X, Y, Z, 10,
                  #[-1, -0.1, 0, 0.1],
                  #alpha=0.5,
                  cmap=cmap,
                  origin=origin)
    plt.xlabel(feature_names[feature_indexes[0]])
    plt.ylabel(feature_names[feature_indexes[1]])
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel(target_feature)
    plt.show()

'''
Requires the estimator, median values, two different feature indexes 
to plot dose response curves
'''
def plot_dose_response(estimator,median_vals,min_vals, max_vals, \
        feature_index,feature_vals,feature_names,target_feature):
    # Plotting all the output values from the curve
    # Divide the minimum and maximum values in 20 points range
    # Total dose is 9th index 
    num_points  = 20
    if (abs(min_vals[9]-max_vals[9])<2):
        x = np.arange(100,1000,900.0/20)
    else:
        x = np.arange(min_vals[9],max_vals[9],\
            (max_vals[9]-min_vals[9])/(num_points*1.0))
    plot_response = np.zeros((x.shape[0],len(feature_vals)))
    for i in range(x.shape[0]):
        for j in range(len(feature_vals)):
            # Get current input values
            inp_feature = median_vals
            inp_feature[9] = x[i] # 9th index is total dose 
            inp_feature[feature_index] = feature_vals[j] # fill-in second feature
            plot_response[i,j] = estimator.predict(inp_feature.reshape(1,-1))[0]
    # Plotting the contour
    plt.figure()
    colors = ['r','g','b','y','k']
    for j in range(len(feature_vals)):
        plt.plot(x,plot_response[:,j],linewidth=3.0,color=colors[j])
    plt.xlabel('Total Dose')
    plt.ylabel(target_feature)
    plt.show()


if __name__=="__main__":
    # filepath = './data/Carbon_Nanotube_Pulmonary_Toxicity_Data_Set_20120313.xls'
    filepath = './data/Toxicity Measurements -- Meta Analysis.xlsx'
    toxicity = "AgNP" # Use "CNT" for analyzing Carbon Nano toxicity
    data = learn_forest(filepath,toxicity)
