# Utility file for reading the excel file containing all the toxicity data
import pandas as pd
import xlrd
import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
##
# @brief Read all the toxicity data from excel file
#
# @param filepath - Full filepath of excel file
#
# @return - Input data and output data frames as Panda dataframes
def import_data(filepath):
    # Read the file located at filepath
    cnt_data = pd.read_excel(filepath,4)
    # Ignore all the particle species that are not carbon
    cnt_data = cnt_data.loc[cnt_data['Particle Species']=='Carbon']

    # Construct the input output data for models
    # Refer to ``A_MetaAnalysis_of_Carbon_Nanotube_Pulmonary_Toxicity_Studies --
    # How_Physical_Dimensions_and_Impurities_Affect_the_Toxicity_of_Carbon_Nanotubes"

    # Input: size and shape,impurities and exposure characteristics
    # Output: PMN, MAC, LAH, TP -- before sampling -- we store both mean and SD
    # We will also store the Author and publication data columns for Model
    # Validation
    size_shape_data = cnt_data[['Config. (1=SW, 2=MW)',
            'min length','length median, nm','max length','min dia',
            'diameter median, nm','max dia', 'MMAD, nm','SA m2/g','Purity']]
    # Converting the data from percentage dose to total and 24 hours average
    # dose
    impurities_wt = cnt_data[['%wt Oxidized C', '% wt Co','% wt Al', 
            '%wt Fe', '%wt Cu', '%wt Cr', '%wt Ni']]
    # Also multiplying by 10^4 to convert from uq/kg and % wt to pg/kg
    impurities_total  = np.multiply(impurities_wt.values,\
            10000*cnt_data[['Total Dose (ug/kg)']].values)
    # Recreating a dataframe from these values
    impurities_total_df = pd.DataFrame(impurities_total,columns=['C Total',\
            'Co Total','Al Total','Fe Total','Cu Total','Cr Total','Ni Total'])
    impurities_24_hr_dose  = np.multiply(impurities_wt.values,\
            10000*cnt_data[['Avg 24-hr Dose (ug/kg)']].values)
    # Recreating a dataframe from these values
    impurities_24_df = pd.DataFrame(impurities_total,columns=['C 24 Hour',\
            'Co 24 Hour','Al 24 Hour','Fe 24 Hour','Cu 24 Hour','Cr 24 Hour','Ni 24 Hour'])
    # Getting exposure characteristics
    exposure = cnt_data[['Exp. Hrs. ','Exp. Per. (hrs)',
            'animal (1=rats, 2=mice)',
            'species (1=sprague-dawley, 2=wistar, 3=C57BL/6, 4=ICR, 5=Crl:CD(SD)IGS BR, 6=BALB/cAnNCrl)',
            'sex (1=male, 2=female)','mean animal mass, g','Post Exp. (days)',
            'Exp. Mode (1=inhalation, 2=instillation, 3=aspiration)',
            'mass conc. (mg/m3)','Total Dose (ug/kg)','Avg 24-hr Dose (ug/kg)',
            'Total Dose (m2/kg)', 'Avg 24-hr Dose (m2/kg)']]
    # Auxiliary data
    aux_data = cnt_data[['Author(s)', 'Year', 'No. of Subjects (N)']]
    # Merging all the input dataframes again
    input_data = pd.concat([aux_data,size_shape_data,impurities_total_df,\
            impurities_24_df,exposure],axis=1)

    # Outputs: PMN, MAC,LDH and TP -- we only use fold of control data as
    # specified by the Authors in the paper
    # In order to obtain multiple columns of data, we will sample num_samples times the
    # number of animals
    output_data = cnt_data[['BAL Neutrophils (fold of control)',\
            'BAL Neutrophils (fold of control) SD','BAL Macrophages (fold of control)'
            ,'BAL Macrophages (fold of control) SD','BAL LDH (fold of control)',
            'BAL LDH (fold of control) SD','BAL Total Protein (fold of control)',
            'BAL Total Protein (fold of control) SD']]

    return (input_data,output_data)

##
# @brief Prepare data from an excel file that can be sent for learning a random
# forest
#
# @param filepath - Full path of excel file containing data
# @param output_feature - What feature output value are we trying to predict,
# possible values are PMN,MAC,LDH and TP
# @param author_exclude - The authors that should be excluded from the learning
# part and just be tested against
# @param num_sample - Number of samples to be drawn from output value Gaussian
# distribution for one sample animal. Default is 100 - same as the paper
#
# @return Input data and output data for random forest learning 
def prepare_data_rf(filepath,output_feature='PMN',author_exclude=None,num_sample=100):

    # Outputs from this function
    train_inp = [];train_out = [];test_inp = [];test_out = []
    # Read the excel file and get all the input data
    (input_data,output_data) = import_data(filepath)

    # In case we are trying to replicate Table IV of the paper, where a part of
    # the data is used for testing and the excluded part is for testing
    if author_exclude is not None:
        exclude_ind = (input_data['Author(s)'].values==author_exclude[0])*\
        (input_data['Year'].values == author_exclude[1])
    else:
        exclude_ind = np.full((input_data.shape[0],),False,dtype=np.bool_)

    # Training Data
    print "Training Data"
    (train_inp,train_out) = sample_input_output(input_data.loc[~exclude_ind,:],\
            output_data.loc[~exclude_ind,:],output_feature,num_sample)
    print "Validation Data"
    # Testing Data
    (test_inp,test_out) = sample_input_output(input_data.loc[exclude_ind,:],\
            output_data.loc[exclude_ind,:],output_feature,num_sample)

    return (train_inp,train_out,test_inp,test_out)

def sample_input_output(input_data,output_data,output_feature,num_sample):
    # Sample input data based on the number of samples required for each test
    # subject
    input_data_sampled = input_data.loc[np.repeat(input_data.index.values,\
            num_sample*input_data['No. of Subjects (N)'].values)]
    
    # For input values, we simply replicated the values. However,
    # we need to sample the output values from a truncated Gaussian distribution
    # Selecting the relevant column of output
    if output_feature=='PMN':
        feat_index = 0
    elif output_feature=='MAC':
        feat_index = 1
    elif output_feature=='LDH':
        feat_index = 2
    else:
        # Implies it is TP
        feat_index = 3

    output_data_sampled =output_sampler(output_data.iloc[:,2*feat_index:2*feat_index+2].values,\
            input_data['No. of Subjects (N)'].values,num_sample)
    
    # Removing the number of test subjects values and reindexing
    input_data_sampled = input_data_sampled.drop("No. of Subjects (N)",axis=1).reset_index(drop=True)

    # Finally removing the names and year and just returning numpy arrays for
    # training RF models
    # We also drop all the input values where the output value is NaN or in
    # other words that data is useless for the current learning problem
    input_data_sampled =input_data_sampled.iloc[~np.isnan(output_data_sampled[:,0]),2:].values
    # Because we are only learning a tree for a single output value
    output_data_sampled = output_data_sampled[~np.isnan(output_data_sampled[:,0]),0]

    return (input_data_sampled,output_data_sampled)

def output_sampler(output_data,num_subjects,num_sample):
    output_data_sampled = np.zeros((np.sum(num_subjects)*num_sample,\
            output_data.shape[1]/2))
    mean_nan = np.zeros((output_data_sampled.shape[1],))
    sd_nan = np.zeros((output_data_sampled.shape[1],))
    # Going over all the rows in output_data and sampling
    for i in range(output_data.shape[0]):
        # Sampling for each output value
        for j in range(output_data_sampled.shape[1]):
            # Core sampling part: for each test subjects- sample
            # num_subjects*num_samples from a constrained normal distribution

            # Lets check for data validity in input
            # if mean value is not present --> That experiments is invalid
            if np.isnan(output_data[i,2*j]):
                mean_nan[j] = mean_nan[j]+1
                # Just saying that all the current samples are NaN (not a number)
                #print "For index ",i," and output index ",j
                #print "Mean value is not a number, generating NaN samples only"
                samples = np.full((num_subjects[i]*num_sample),np.nan)
            # If SD value is not given, lets assume it to be a very small value
            elif np.isnan(output_data[i,2*j+1]):
                sd_nan[j] = sd_nan[j]+1
                #print "For index ",i," and output index ",j
                #print "Standard deviation value not given, sampling with almost 0 SD"
                samples = sample_truncated(output_data[i,2*j],1e-5,\
                    num_subjects[i]*num_sample)
            else:
                # This mean both mean and SD are provided
                samples = sample_truncated(output_data[i,2*j],output_data[i,2*j+1],\
                    num_subjects[i]*num_sample)
            # Filling in all the values
            output_data_sampled[np.sum(num_subjects[:i])*num_sample:np.sum(num_subjects[:i+1])*num_sample,j]\
                    =samples
    # Printing summary stats
    print "Output data summary: Total Not a Number (NaN) samples"
    print "Total number of studies: ",output_data.shape[0]
    print "index is 0=PMN,1=MAC,2=LDH,3=TP:"
    print "mean NaN: ", mean_nan
    print "SD NaN: ", sd_nan
    return output_data_sampled

# Sample num_samples for a truncated Gaussian distribution with lower bound
# being 0 and upper bound being +infinity
def sample_truncated(mu,sigma,num_sample):
    lower, upper = 0, np.inf
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs(num_sample)

# To prove the point that sampling from a truncated distribution does not induce
# bias in sampling
def demo_samples():
    lower, upper = 0, np.inf
    mu, sigma = 2.75, 1.55
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    N = stats.norm(loc=mu, scale=sigma)

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].hist(X.rvs(10000), normed=True)
    ax[0].set_title('Truncated Gaussian Distribution Samples')
    ax[1].hist(N.rvs(10000), normed=True)
    ax[1].set_title('Gaussian Distribution Samples')
    # Sample 10000 points and make all the negative points 0
    samples = N.rvs(10000)
    modified_samples = np.zeros(samples.shape)
    for i in range(samples.shape[0]):
        if samples[i]<0:
            modified_samples[i] = 0
        else:
            modified_samples[i] = samples[i]

    ax[2].hist(modified_samples, normed=True)
    ax[2].set_title('Gaussian Distribution Samples with moving negative values strategy')
    plt.show()

if __name__=="__main__":
    filepath = './data/Carbon_Nanotube_Pulmonary_Toxicity_Data_Set_20120313.xls'
    data = prepare_data_rf(filepath)
