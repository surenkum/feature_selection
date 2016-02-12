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
# @param num_sample - Number of samples to be drawn from output value Gaussian
# distribution for one sample animal. Default is 100 - same as the paper
#
# @return Input data and output data for random forest learning 
def prepare_data_rf(filepath,num_sample=100):
    # Read the excel file and get all the input data
    (input_data,output_data) = import_data(filepath)

    # Sample input data based on the number of samples required for each test
    # subject
    input_data_sampled = input_data.loc[np.repeat(input_data.index.values,\
            num_sample*input_data['No. of Subjects (N)'].values)]
    # Removing the number of test subjects values and reindexing
    input_data_sampled = input_data_sampled.drop("No. of Subjects (N)",axis=1).reset_index(drop=True)

    # For input values, we simply replicated the values. However,
    # we need to sample the output values from a truncated Gaussian distribution
    output_data_sampled = output_sampler(output_data,num_sample)
    pdb.set_trace()


def output_sampler(output_data,num_sample):
    output_data_sampled = np.zeros(output_data.shape)

    return output_data_sampled

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
