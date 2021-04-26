#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse
import re
import ast

## Input parameters + path to directory and files ## 
directory_name = 'Planck2018thetad' # Name of your directory
path_name = '/Users/Kim/Desktop/MontePythonCluster/chains/' # Path to your MontePython directory
data_name = 'Planck2018thetad_2d_100*theta_d-100*theta_s.dat' # Name of your .dat file
param_list = ['100*theta_s','100*theta_d'] # Name of the variables of interest. See names in 'cov_list' below.
# The first parameter will be the x-variable in the contour plot. 
# Now you are ready to run the code. The following two parameters can be adjusted if one wishes to multiply 
# the entries of the covariance matrix by a scalar or generate more/less random sample points from a 2D Gaussian. 
factor = 1 # Used to multiply the entries of the covariance matrix to better match contours.
size = 100000 # no. of generated points from 2D Gaussian with covariance matrix from .covmat and mu's from .h_info 

# Path to files
cov_file = path_name + directory_name + '/' + directory_name +  '.covmat' # path to .covmat file
h_file = path_name + directory_name + '/' + directory_name + '.h_info' 
dat_file = path_name + '/' + directory_name + '/plots/' + data_name

## Creating covariance matrix ## 

# Loading cov_file
with open(cov_file, 'r') as index_cov:
    index_line = index_cov.readline()  # first line gives the parameter names
# Removing symbols/text from cells
cov_list = [elem.strip() for elem in index_line[1:].split(',')]

# Creating covariance matrix 
covFull = np.loadtxt(cov_file)
variable_indices = [cov_list.index(p) for p in param_list]
cov = covFull[np.ix_(variable_indices, variable_indices)]

## Selecting mu's from .h_info file ##
with open(h_file, 'r') as index_h:
    index_h = index_h.readlines(1)
    
# Removing unwanted symbols/text from parameter names
index_h = [i.replace(' param names\t:', '') for i in index_h] 
index_h = [i.replace('\n', '') for i in index_h]
index_h = [i.replace('\\', '') for i in index_h]
index_h = [i.replace('{', '') for i in index_h]
index_h = [i.replace('}', '') for i in index_h]
string_h = " ".join(str(x) for x in index_h) # converting to string to seperate spacing with commas
string_h = re.sub("\s+", ",", string_h.strip())
index_h = string_h.split(",")

with open(h_file, 'r') as h_list:
    lines = h_list.readlines()   
mu_indices = [index_h.index(p) for p in param_list]
mean_row = lines[3] 
# The mean row is located at row 3 in the h_info file. If one wishes to use best-fit values instead, change 3 to 2
mean_row = mean_row.split()
mean_row = [m.replace('mean', '') for m in mean_row] 
mean_row = [m.replace('\t', '') for m in mean_row] 
mean_row = [m.replace('\n', '') for m in mean_row] 
mean_row = [m.replace(':', '') for m in mean_row] 
mean_row = [m.replace('', '') for m in mean_row] 
string_mean = " ".join(str(x) for x in mean_row) # converting to string to seperate spacing with commas
string_mean = re.sub("\s+", ",", string_mean.strip())
mean_row = string_mean.split(",")
mu_list = [mean_row[x] for x in mu_indices]
mu_list = (('[%s]' % ', '.join(map(str, mu_list))))
mu_list = ast.literal_eval(mu_list)
mu = tuple(mu_list)

## 2D Gaussian plot ## 
contour = pd.read_csv(dat_file, sep=' ') # Converts .dat to csv.file
contour.rename(columns = {'#':'para1_col','contour':'para2_col'},inplace = True)
# The first row in the file has entries '#' and 'contour'
i = contour[(contour.para1_col == '#')].index 
# locates line in the file which says "# contour for confidence level 0.6826000000000001"
contour = contour.drop(i) # removes that line
contour['para1_col'] = contour['para1_col'].map(lambda x: str(x)[:-1]) # removes \t from first column

para1_col = contour.para1_col
para2_col = contour.para2_col
para1 =  pd.to_numeric(para1_col, errors='coerce') # convert to float object
para2 =  pd.to_numeric(para2_col, errors='coerce')

def error_ellipse(ax, xc, yc, cov, sigma=1, facecolor='none', **kwargs):
    '''
    Plot an error ellipse contour over your data.
    Inputs:
    ax : matplotlib Axes() object
    xc : x-coordinate of ellipse center
    yc : x-coordinate of ellipse center
    cov : covariance matrix
    sigma : # sigma to plot (default 1)
    additional kwargs passed to matplotlib.patches.Ellipse()
    '''
    w, v = np.linalg.eigh(cov) # assumes symmetric matrix
    order = w.argsort()[::-1]
    w, v = w[order], v[:,order]
    theta = np.degrees(np.arctan2(*v[:,0][::-1]))
    ellipse = Ellipse(xy=(xc,yc),
                    width=2.*sigma*np.sqrt(w[0]),
                    height=2.*sigma*np.sqrt(w[1]),
                    angle=theta, **kwargs)
    ellipse.set_facecolor('none')
    ax.add_artist(ellipse)
    
if __name__ == '__main__':
    # Generate random points
    points = np.random.multivariate_normal(
            mean=mu, cov=cov, size=size)
    x, y = points.T
    cov = np.cov(x,y, rowvar=False)
    # Plot the raw points
    fig, ax = plt.subplots(1,1)
    ax.scatter(x, y, s=1, color='b')
    # Plot one and two sigma error ellipses
    error_ellipse(ax, np.mean(x), np.mean(y), cov, sigma=1, ec='orange')
    error_ellipse(ax, np.mean(x), np.mean(y), cov, sigma=2, ec='orange')
    ax.scatter(mu[0], mu[1], c='orange', s=1, label='Generated Contour') # center of ellipse    
    if variable_indices[0]<variable_indices[1]:
        ax.plot(para1,para2, 'o', markersize=1, color='black', label='Planck 2018 contour')
    else: 
        ax.plot(para2,para1, 'o', markersize=1, color='black', label='Planck 2018 contour')
    ax.set_xlabel(param_list[0])
    ax.set_ylabel(param_list[1])
    ax.legend()
    plt.show()
