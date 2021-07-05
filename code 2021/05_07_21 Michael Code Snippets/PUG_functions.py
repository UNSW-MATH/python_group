# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:43:26 2021
@author: mphem

Some functions presented during PUG on 05/07/2021

"""

# %% ----------------------------------------------------------
# Import packages

import numpy as np
import datetime as dt
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import h5py 
from scipy.io import loadmat
from scipy import stats
from scipy.io import savemat

# %% ----------------------------------------------------------
# Time functions

# Converting from datetime to Numpy datetime64
def to_date64(TIME):
    
    # convert to numpy
    if 'xarray' in str(type(TIME)):
        TIME = np.array(TIME)
    # if only one time, easier to convert using numpy
    if np.size(TIME) == 1: 
        TIME = np.datetime64(TIME)
    else:
        t = []
        for nt in range(len(TIME)):
            o = TIME[nt]
            if '64' not in str(type(o)):
                # both lines work
                t.append(np.datetime64(o.strftime("%Y-%m-%dT%H:%M:%S")))
                # t.append(np.datetime64(o))
            else:
                t.append(o)
        TIME = np.array(t)
        
    return TIME
    
def to_datetime(TIME):
    if 'xarray' in str(type(TIME)):
        TIME = np.array(TIME)    
    if np.size(TIME) == 1:
        TIME = TIME.tolist()
    else: 
        t = []
        # Check that input is xarray data array
        # if 'xarray' not in str(type(TIME)):
        #     TIME = xr.DataArray(TIME)
        for nt in range(len(TIME)):
            o = TIME[nt]
            if '64' in str(type(o)):
                t_str = str(np.array(o))
                if len(t_str) > 10:
                    yr = int(t_str[0:4])
                    mn = int(t_str[5:7])
                    dy = int(t_str[8:10])
                    hr = int(t_str[11:13])
                    mins = int(t_str[14:16])
                    secs = int(t_str[17:19])
                    t.append(dt.datetime(yr,mn,dy,hr,mins,secs))
                if len(t_str) == 10:
                    yr = int(t_str[0:4])
                    mn = int(t_str[5:7])
                    dy = int(t_str[8:10])                
                    t.append(dt.datetime(yr,mn,dy))
                if len(t_str) == 7:
                    yr = int(t_str[0:4])
                    mn = int(t_str[5:7])                
                    t.append(dt.datetime(yr,mn,1))
                if len(t_str) == 4:
                    t.append(dt.datetime(yr,1,1))
        TIME = np.array(t) 

    return TIME

# %% ---------------------------------------------------------

def time_range(start,end,res,time_format):
    """
    start / end = can either be integer years, or numpy
                  datetime64/datetime dates (don't mix)
    res = 'monthly','daily','yearly'
    time_format = 'np64' or 'datetime'
    
    """
    if 'int' not in str(type(start)):
        if '64' not in str(type(start)): 
            start = np.datetime64(start)
            end = np.datetime64(end)    
        if 'monthly' in res:
                time = np.arange(start,end,np.timedelta64(1, 'M'),
                                 dtype='datetime64[M]')
        if 'daily' in res:
            time = np.arange(start, end, np.timedelta64(1, 'D'),  
                             dtype='datetime64[D]')       
        if 'yearly' in res:
            time = np.arange(start, end, np.timedelta64(1, 'Y'),  
                             dtype='datetime64[Y]') 
        time = np.array(time)
    else:     

        if 'monthly' in res:
            time = np.arange(np.datetime64(str(start) + '-01-01'), 
                             np.datetime64(str(end) + '-01-01'),  
                             np.timedelta64(1, 'M'),  
                             dtype='datetime64[M]')
        if 'daily' in res:
            time = np.arange(np.datetime64(str(start) + '-01-01'), 
                             np.datetime64(str(end) + '-01-01'),  
                             np.timedelta64(1, 'D'),  
                             dtype='datetime64[D]')   
        if 'yearly' in res:
            time = np.arange(np.datetime64(str(start)), 
                             np.datetime64(str(end)),  
                             np.timedelta64(1, 'Y'),  
                             dtype='datetime64[Y]')   
            
    if 'np64' not in time_format:
        time = to_datetime(np.array(time))
    
    return time

# %% ---------------------------------------------------------
    
# Python version of MATLAB's datevec function  
def datevec(TIME):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    TIME : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """
    
    # If not a datetime64, convert from dt.datetime
    TIME = to_date64(TIME)
    # allocate output 
    out = np.empty(TIME.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [TIME.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (TIME - D).astype("m8[h]") # hour
    out[..., 4] = (TIME - h).astype("m8[m]") # minute
    out[..., 5] = (TIME - m).astype("m8[s]") # second
    out[..., 6] = (TIME - s).astype("m8[us]") # microsecond
    
    yr = out[:,0]; mn = out[:,1]; dy = out[:,2]; hr = out[:,3]; 
    yday = []
    for n in range(len(yr)):
        yday.append(dt.date(yr[n], mn[n], dy[n]).timetuple().tm_yday)

    return yr, mn, dy, hr, yday

# %% ---------------------------------------------------------
# binning functions

# %% ---------------------------------------------------------
def bin_over_time(start_yr,end_yr,TIME,VARIABLE,res):
    """
    start / end = integer year
    TIME = Time array 
    VARIABLE = Variable array
    res = 'monthly','daily','yearly'
    
    """
    
    # convert TIME and VARIABLE to numpy array
    TIME = np.array(TIME)
    VARIABLE = np.array(VARIABLE)    

    # create time grid
    if 'daily' in res:
        time_grid_lower = time_range(start_yr,end_yr,'daily','np64')
        time_grid_upper = time_grid_lower + np.timedelta64(1, 'D') 
        time_grid = time_grid_lower + np.timedelta64(12, 'h')
    if 'monthly' in res:
        time_grid = time_range(start_yr,end_yr,'monthly','np64')     
    if 'yearly' in res:
        time_grid = time_range(start_yr,end_yr,'yearly','np64')      
    # bin data
    bin_t = []; bin_med = []; bin_mean = []; bin_SD = []; bin_sum = []
    if 'daily' in res:
        for n_day in range(len(time_grid)):
            c = [(TIME > time_grid_lower[n_day]) &
                 (TIME <= time_grid_upper[n_day])]
            bin_t.append(time_grid[n_day])
            bin_med.append(np.nanmedian(VARIABLE[c]))
            bin_mean.append(np.nanmean(VARIABLE[c]))
            bin_SD.append(np.nanstd(VARIABLE[c]))
            bin_sum.append(len(VARIABLE[c]))
    if 'monthly' in res:    
        # get years, months, days 
        yr, mn, _, _, _ = datevec(TIME)
        unique_yrs = np.sort(np.unique(yr))   
        for n_yr in range(len(unique_yrs)):
            for n_mn in range(1,13):
                c = [(yr == unique_yrs[n_yr]) & (mn == n_mn)]
                bin_t.append(dt.datetime(unique_yrs[n_yr],n_mn,1))
                bin_med.append(np.nanmedian(VARIABLE[c]))
                bin_mean.append(np.nanmean(VARIABLE[c]))
                bin_SD.append(np.nanstd(VARIABLE[c]))
                bin_sum.append(len(VARIABLE[c]))
    if 'yearly' in res:    
        # get years, months, days 
        yr, _, _, _, _ = datevec(TIME)
        unique_yrs = np.sort(np.unique(yr))   
        for n_yr in range(len(unique_yrs)):
            c = yr == unique_yrs[n_yr]
            bin_t.append(dt.datetime(unique_yrs[n_yr],1,1))
            bin_med.append(np.nanmedian(VARIABLE[c]))
            bin_mean.append(np.nanmean(VARIABLE[c]))
            bin_SD.append(np.nanstd(VARIABLE[c]))
            bin_sum.append(len(VARIABLE[c]))
    # save binned output as a class
    class bin_class:
        TIME = np.array(bin_t)
        MEDIAN = np.array(bin_med)
        MEAN = np.array(bin_mean)
        SD = np.array(bin_SD)
        POINTS_IN_BIN = np.array(bin_sum)
 
    return bin_class

# %% ---------------------------------------------------------

def bin_profile(VARIABLE,DEPTH,TIME,BINS,BIN_M,METHOD):
    
    DEPTH = np.array(DEPTH); 
    VARIABLE = np.array(VARIABLE)
    TIME = np.array(TIME)
    
    # get time information if seasons selected
    if 'all' not in METHOD:
        _, mn, _, _, _ = datevec(TIME)
    
    # define lists
    bin_mean = []
    bin_SD = []
    bin_median = []
    bin_D = []
    bin_n = []
    bin_V_points = []
    bin_V_points_D = []
    
    
    if 'get' in BINS:
        H, edges = np.histogramdd(DEPTH,int(np.round(np.nanmax(DEPTH))/2))
        H = (H - np.nanmin(H)) / (np.nanmax(H)-np.nanmin(H))
        # get normalised temporary bin sums in vertical depth
        edges = np.squeeze(edges); edges = edges[0:-1]
        BINS = edges[H > 0.4]
        last_BIN = BINS[-1]
        # remove bins that are too close to one another (3 m)
        diff_BINS = np.diff(BINS)
        f_diff_BINS = np.where(diff_BINS >= 3)
        BINS = BINS[f_diff_BINS]
        BINS = np.append(np.int32(BINS),np.int32(last_BIN))

    # bin data
    for n_b in range(len(BINS)):
        if 'all' in METHOD:
            c = [(DEPTH >= BINS[n_b] - BIN_M) & 
                 (DEPTH < BINS[n_b] + BIN_M)]
        if 'summer' in METHOD:
            c = np.squeeze([(DEPTH >= BINS[n_b] - BIN_M) & 
                 (DEPTH < BINS[n_b] + BIN_M) & 
                 [(mn == 11) | (mn == 12) | (mn == 1)]])
        if 'autumn' in METHOD:
            c = np.squeeze([(DEPTH >= BINS[n_b] - BIN_M) & 
                 (DEPTH < BINS[n_b] + BIN_M) & 
                 [(mn == 2) | (mn == 3) | (mn == 4)]])            
        if 'winter' in METHOD:
            c = np.squeeze([(DEPTH >= BINS[n_b] - BIN_M) & 
                 (DEPTH < BINS[n_b] + BIN_M) & 
                 [(mn == 5) | (mn == 6) | (mn == 7)]])  
        if 'spring' in METHOD:
            c = np.squeeze([(DEPTH >= BINS[n_b] - BIN_M) & 
                 (DEPTH < BINS[n_b] + BIN_M) & 
                 [(mn == 8) | (mn == 9) | (mn == 10)]])               
        bin_mean.append(float(np.nanmean(VARIABLE[c])))
        bin_median.append(float(np.nanmedian(VARIABLE[c])))
        bin_D.append(int(BINS[n_b]))
        bin_SD.append(float(np.nanstd(VARIABLE[c])))
        bin_n.append(int(np.size(VARIABLE[c])))
        bin_V_points.append(VARIABLE[c])
        bin_V_points_D.append(
            np.squeeze(np.ones((1,bin_n[n_b]),dtype=float) * BINS[n_b]))

    # save binned output as a class
    class bin_class:
        MEDIAN = np.array(bin_median)
        MEAN = np.array(bin_mean)
        SD = np.array(bin_SD)
        DEPTH = np.array(bin_D)
        POINTS_IN_BIN = bin_V_points
        DEPTH_POINTS_IN_BIN = bin_V_points_D

    return bin_class
    
# %% ---------------------------------------------------------
# MATLAB functions

# %% ------------------------------------------------------------
# datenum

def datetime2matlabdn(python_datetime):
   mdn = python_datetime + dt.timedelta(days = 366)
   frac_seconds = (python_datetime-dt.datetime(
       python_datetime.year,python_datetime.month,
       python_datetime.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
   frac_microseconds = python_datetime.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
   return mdn.toordinal() + frac_seconds + frac_microseconds   

def matlabdn2datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    time = []
    for n in range(len(datenum)):
        dn = np.float(datenum[n])
        days = dn % 1
        d = dt.datetime.fromordinal(int(dn)) \
           + dt.timedelta(days=days) \
           - dt.timedelta(days=366)
        time.append(d)
        
    return time
          
# %% ------------------------------------------------------------
# loading in MATLAB data

class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

def load_MATfile(filename):
    
    try:
        # if not 'v7.3'
        data = loadmat(filename)
    except:
        # if 'v7.3'
        data = h5py.File(filename,'r') 
    # convert dictionary to class
    if 'dict' in str(type(data)):
        # if not 'v7.3'
        data = Dict2Obj(data)
    else:
        # if 'v7.3'
        keys = list(data.keys())
        keys = keys[2:-1]
        # data = Dict2Obj(data) 
        data_dict = {}
        for n_keys in range(len(keys)):
            d = np.array(eval('data.get("' + keys[n_keys] + '")'))
            data_dict[keys[n_keys]] = []    
            data_dict[keys[n_keys]].append(d)

    return data


