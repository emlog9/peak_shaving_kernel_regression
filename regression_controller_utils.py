# Core Python libraries
import os
import time
import math
import ast
import random as rnd

# Numerical / Data libraries
import numpy as np
import pandas as pd

# Optimization libraries
import gurobipy as gp
from gurobipy import GRB

# SciPy utilities
import scipy.io
from scipy.io import savemat
from scipy import interpolate
from scipy.special import expit, logit

# Machine Learning / scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# File handling and iteration utilities
from glob import glob
from itertools import product



#------------------------------------------------------------------------------------------------
# Find optimal min SoC reserve and peak-shaving target for historical training data
#------------------------------------------------------------------------------------------------
def training_data_generation(E_batt, P_batt, E_min, e0, eta, D, Ts, mins_in_peak, delta=1e-6):
    
    """   
    Optimize battery dispatch for peak shaving given historical demand.
    
    Parameters
    ----------
    E_batt: Battery energy capacity (MWh)
    P_batt: Battery power limit (MW)
    E_min: Minimum allowed SoC (MWh)
    e0: Initial state of charge (SoC) (MWh)
    eta: Battery round-trip efficiency
    D: Demand profile (MW)
    Ts: Timestep duration (hours)
    mins_in_peak: Duration of peak demand window for peak shaving (minutes)
    delta: Small tie-breaking penalty on SoC

    Returns
    -------
    net_demand_opt: Optimized net demand after battery dispatch (MW)
    power: Battery power (discharge - charge) at each timestep (MW)
    e_opt: SoC trajectory (MWh)
    d_opt: Optimized discharge power (MW)
    q_opt: Optimized charge power (MW)
    p_value: Peak demand value after optimization (MW)
    """

    T = len(D)  # number of timesteps
    model = gp.Model("battery_optimization")

    # Variables
    d = model.addVars(T, lb=0, ub=P_batt, name="discharge")  # discharge power (MW)
    q = model.addVars(T, lb=0, ub=P_batt, name="charge")     # charge power (MW)
    e = model.addVars(T, lb=E_min, ub=E_batt, name="SoC")    # SoC (MWh)
    p = model.addVar(vtype=GRB.CONTINUOUS, name="peak_variable") # Peak demand variable (MW)

    # Objective function
    model.setObjective(gp.quicksum(1e-6*d[t] + delta*e[t] for t in range(T)) + 10000*p, GRB.MINIMIZE)


    # Constraints
    for t in range(T):
        # Ensure peak variable is above net demand at each timestep
        model.addConstr(p >= D[t] - d[t] + q[t])
        # SoC dynamics
        if t == 0:
            model.addConstr(e[t] == e0 - d[t]/eta*Ts + q[t]*eta*Ts)
        else:
            model.addConstr(e[t] == e[t-1] - d[t]/eta*Ts + q[t]*eta*Ts)
            

    model.Params.OutputFlag = 0

    # Solve
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print("Optimization was not successful")
        print("Gurobi status:", model.status)
        return None

    # Extract results
    d_opt = np.array([d[t].X for t in range(T)])
    q_opt = np.array([q[t].X for t in range(T)])
    e_opt = np.array([e[t].X for t in range(T)])

    d_opt[d_opt < 1e-6] = 0
    q_opt[q_opt < 1e-6] = 0

    power = d_opt - q_opt 
    net_demand_opt = D-power

    p_value = model.getVarByName("peak_variable").X

    return net_demand_opt, power, e_opt, d_opt, q_opt, p_value


#------------------------------------------------------------------------------------------------
# Peak charge calculator
#------------------------------------------------------------------------------------------------
def calculate_peak_demand_charge(demand_profile, timestamps, Ts, mins_in_peak, std_pricing = True, verbose=False):
    """
    Calculates monthly peak demand charges for ConEd large business customers.
    
    Can compute charges for standard or TOU pricing. Prints a detailed monthly breakdown if verbose=True.

    Parameters
    ----------
    demand_profile: Demand profile (MW)
    timestamps: Corresponding datetime timestamps
    Ts: Timestep duration (h)
    mins_in_peak: Duration of peak demand window for peak shaving (minutes)
    std_pricing: If True, uses standard flat-rate pricing
    verbose: If True, prints detailed monthly peak charge breakdown (default: False)

    Returns
    -------
    total_peak_charge: Total annual/monthly peak demand charge ($)
    monthly_breakdown: List of monthly peak demand charges ($)
    """

    # Initialize series and rolling window
    demand_series = pd.Series(demand_profile, index=pd.DatetimeIndex(timestamps))
    window_size = int(mins_in_peak / 60 / Ts) 
    total_peak_charge = 0.0
    monthly_breakdown = []
    months = demand_series.index.to_period('M').unique()
    
    if verbose:
        print("--- Peak Demand Price Breakdown ---")
        
    # Loop through each month
    for month_period in months:
        month_data = demand_series[demand_series.index.to_period('M') == month_period]
        month_num = month_data.index[0].month
    
        # Rolling avg
        rolling_avg = month_data.rolling(window=window_size,min_periods=1).mean()

        # Summer months (Jun-Sep)
        if 6 <= month_num <= 9:
            if std_pricing:
                # If standard rate
                rates = {'all_hours': 42.80 * 1000} # $/MW
                periods = {'all_hours': rolling_avg}
            else:
                # Define periods and rates for TOU pricing
                rates = {'daytime_peak': 12.77 * 1000,  # 8 a.m.–6 p.m., M–F
                         'on_peak': 27.40 * 1000,       # 8 a.m.–10 p.m., M–F
                         'all_hours': 26.20 * 1000      # All hours, all days
                        }
                periods = {'daytime_peak': rolling_avg[(rolling_avg.index.weekday <= 4) & 
                                                (rolling_avg.index.hour >= 8) & (rolling_avg.index.hour < 18)],
                           'on_peak': rolling_avg[(rolling_avg.index.weekday <= 4) & 
                                                (rolling_avg.index.hour >= 8) & (rolling_avg.index.hour < 22)],
                           'all_hours': rolling_avg}

        # Non-summer (other) months (Oct–May):
        else:
            if std_pricing:
                rates = {'all_hours': 33.50 * 1000} # $/MW
                periods = {'all_hours': rolling_avg}
            else:
                rates = {
                    'on_peak': 17.74 * 1000,   # 8 a.m.–10 p.m., M–F
                    'all_hours': 7.51 * 1000   # all hours
                }
                periods = {
                    'on_peak': rolling_avg[(rolling_avg.index.weekday <= 4) & 
                                           (rolling_avg.index.hour >= 8) & (rolling_avg.index.hour < 22)],
                    'all_hours': rolling_avg}
                
        # Compute monthly peak charge
        month_charge = 0.0
        for label, period_data in periods.items():
            if len(period_data) == 0:
                continue
            peak_value = period_data.max()
            rate = rates[label]
            charge = peak_value * rate
            month_charge += charge
        
            if verbose:
                peak_time = period_data.idxmax()
                print(f"Month {month_num:02d} | {label:12s} | Peak {peak_value:.4f} MW @ {peak_time.strftime('%Y-%m-%d %H:%M')} | Rate ${rate/1000:.2f}/kW | Cost ${charge:,.2f}")

        
        monthly_breakdown.append(month_charge)
        total_peak_charge += month_charge
        
    if verbose:
        print("--------------------------------------------------")
    
    return total_peak_charge, monthly_breakdown


#------------------------------------------------------------------------------------------------
# Compute Total Bill
#------------------------------------------------------------------------------------------------
def calculate_total_bill(demand_profile, timestamps, Ts, mins_in_peak, monthly_charge, RTP, std_pricing = True, verbose=False):

    """
    Calculates the total electricity bill including energy, peak demand, and monthly customer charges.
    
    Parameters
    ----------
    demand_profile: Demand profile (MW)
    timestamps: Corresponding datetime timestamps
    Ts: Timestep size (h)
    mins_in_peak: Duration of peak demand window for peak shaving (minutes)
    monthly_charge: Fixed monthly customer charge ($)
    RTP: Real-time electricity prices ($/MWh)
    std_pricing: If True, use standard peak pricing (default: True)
    verbose: If True, prints detailed bill breakdown (default: False)

    Returns
    -------
    total_bill: Total electricity bill ($)
    """

    # Convert demand profile to Pandas Series
    demand_series = pd.Series(demand_profile, index=pd.DatetimeIndex(timestamps))
    
    # Energy charge
    energy_charge = np.sum(demand_series.values * RTP * Ts) # prices given in $/MWh

    # Monthly customer charges
    months = demand_series.index.to_period('M').unique()
    monthly_customer_charge = monthly_charge * len(months)

    # Peak demand charges
    peak_charge, peak_charge_by_month = calculate_peak_demand_charge(demand_profile, timestamps, Ts, mins_in_peak, std_pricing=std_pricing, verbose=verbose)

    # Total bill
    total_bill = energy_charge + monthly_customer_charge + peak_charge

    if verbose:
        print(f"Energy Charge: ${energy_charge:,.2f}")
        print(f"Peak Charge:   ${peak_charge:,.2f}")
        print(f"Cumulative Monthly Fees:   ${monthly_customer_charge:,.2f}")
        print(f"TOTAL BILL:    ${total_bill:,.2f}\n")

    return total_bill


#------------------------------------------------------------------------------------------------
# Kernel Regression
#------------------------------------------------------------------------------------------------
def kernel_regression(D, D_hist, minSoC_hist, p_hist, T, e0, sigma, K, alpha_SoC,
                          sin_time_hist, cos_time_hist, sin_time_test, cos_time_test):

    """
    Kernel regression using a Gaussian kernel over K nearest neighbors - predicts SoC and peak demand target for peak shaving.

    Parameters
    ----------
    D: Demand (test data) profile (MW)
    D_hist: Historical (training) demand (MW)
    minSoC_hist: Historical minimum SoC values corresponding to D_hist (MWh)
    p_hist: Historical net demand values corresponding to D_hist (MW)
    T: Lookback window size (timesteps)
    e0: Initial SoC value (MWh)
    sigma: Gaussian kernel bandwidth
    K: Number of nearest neighbors
    alpha_SoC: Confidence level for SoC prediction (0 < alpha_SoC <= 1)
    sin_time_hist, cos_time_hist: Sine/cosine encoded time features for historical data
    sin_time_test, cos_time_test: Sine/cosine encoded time features for test/current data

    Returns
    -------
    e_pred_conf: SoC prediction at confidence level (MWh)
    p_pred : Peak demand prediction (MW)
    indices: Indices of nearest neighbors in historical data
    weights: Gaussian weights assigned to neighbors
    """

    M = len(D) # total timesteps in current sequence
    N = len(D_hist) # total historical timesteps

    # Initialize predictions
    e_pred = np.zeros(M)
    e_pred_conf = np.zeros(M)
    p_pred = np.zeros(M)
    p_pred_conf = np.zeros(M)

    # Initialize first T timesteps
    e_pred[:T] = e0
    e_pred_conf[:T] = e0
    p_pred[:T] = np.mean(p_hist[:T])
    p_pred_conf[:T] = np.mean(p_hist[:T])

    # Indices where regression is applied
    valid_idx = np.arange(T, M)

    # -----------------------------
    # Prepare feature vectors
    # ----------------------------
    
    # Current features: demand history + time features
    D_current_features = (np.array([D[t-T:t] for t in valid_idx]))
    time_current = (np.column_stack([sin_time_test[valid_idx], cos_time_test[valid_idx]]))
    X_current = np.hstack([D_current_features, time_current])

    # Historical features
    D_past_features = (np.array([D_hist[s-T:s] for s in range(T, N)]))
    time_past = np.column_stack([sin_time_hist[T:], cos_time_hist[T:]])
    X_past = np.hstack([D_past_features, time_past])
    e_past = minSoC_hist[T:]
    p_past = p_hist[T:]

    # -----------------------------
    # Fit KNN and compute Gaussian weights
    # -----------------------------
    knn = NearestNeighbors(n_neighbors=K, algorithm='auto', metric='euclidean')
    knn.fit(X_past)
    dists, indices = knn.kneighbors(X_current)

    # Gaussian kernel with lookback window normalization
    weights = np.exp(-dists**2 / (2 * T * sigma**2))
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    weights /= row_sums

    tolerance = 1e-6
    if abs(np.sum(weights, axis=1).min() - 1) > tolerance:
        print(f"Weights do not sum to 1. Check parameters.")
        return None, None, None, None

    # -----------------------------
    # Predict SoC and peak demand
    # -----------------------------
    for i, t in enumerate(valid_idx):
        # SoC confidence estimate
        sorted_idx = np.argsort(e_past[indices[i]])
        sorted_SoC = e_past[indices[i]][sorted_idx]
        sorted_weights = weights[i][sorted_idx]
        cdf = np.cumsum(sorted_weights)
        k = np.searchsorted(cdf, alpha_SoC)

        if k == 0:
            SoC_at_conf = sorted_SoC[0]
        else:
            cdf_low = cdf[k-1]
            cdf_high = cdf[k]
            SoC_low = sorted_SoC[k-1]
            SoC_high = sorted_SoC[k]
            SoC_at_conf = SoC_low + (alpha_SoC - cdf_low) / (cdf_high - cdf_low) * (SoC_high - SoC_low)

        e_pred_conf[t] = SoC_at_conf
        e_pred[t] = np.sum(weights[i] * e_past[indices[i]])

        # Peak demand confidence estimate
        sorted_idx_p = np.argsort(p_past[indices[i]])
        sorted_p = p_past[indices[i]][sorted_idx_p]
        sorted_w_p = weights[i][sorted_idx_p]
        cdf_p = np.cumsum(sorted_w_p)

        kp = np.searchsorted(cdf_p, alpha_SoC)

        if kp == 0:
            p_conf = sorted_p[0]
        else:
            cdf_low, cdf_high = cdf_p[kp-1], cdf_p[kp]
            p_low, p_high = sorted_p[kp-1], sorted_p[kp]
            p_conf = p_low + (alpha_SoC - cdf_low) / (cdf_high - cdf_low) * (p_high - p_low)

        p_pred_conf[t] = p_conf      
        p_pred[t] = np.sum(weights[i] * p_past[indices[i]])
    
    return e_pred_conf[T:], p_pred[T:], indices, weights


#------------------------------------------------------------------------------------------------
# Real-Time Controller
#------------------------------------------------------------------------------------------------
def real_time_control(D, SoC_reserve, Ts, eta, p_init, p_t_series, P_batt, E_batt, e0):
    """
    Implements real-time battery control for peak shaving.

    Parameters
    ----------
    D: Demand profile(MW)
    SoC_reserve: SoC reserve profile from kernel regression (MWh)
    Ts: Timestep size (h)
    eta: Battery efficiency (0 < eta <= 1)
    p_init: Initial peak shaving target (MW)
    p_t_series: Kernel regression predicted peak targets over the timeframe (MW)
    P_batt: Battery power limit (MW)
    E_batt: Battery energy capacity (MWh)
    e0: Initial SoC (MWh)

    Returns
    -------
    net_demand: Net demand after battery dispatch (MW)
    q: Battery charging power after real-time adjustment (MW)
    d: Battery discharging power after real-time adjustment (MW)
    SoC: SoC trajectory after real-time adjustment (MWh)
    p_current: Final peak demand target after real-time adjustment (MW)
    """
    
    T = len(SoC_reserve)
    d = np.zeros(T)          # Discharge array
    q = np.zeros(T)          # Charge array
    SoC = np.zeros(T+1)      # SoC array
    net_demand = np.zeros(T) # Net demand array
    SoC[0] = e0

    p_current = p_init

    for t in range(T):        

        # Predicted peak demand from kernel regression
        p_pred_t = p_t_series[t]
        # Update active peak target
        p_current = max(p_current, p_pred_t)
        
        # Difference between target SoC and current SoC
        delta_SoC = SoC_reserve[t] - SoC[t]

        # Decide discharge
        if D[t] > p_current:
            # Discharge if demand exceeds current peak target
            d[t] = max(0, min(max(-(delta_SoC/Ts)*eta, D[t]-p_current), SoC[t]*eta / Ts, P_batt)) 
            # respect peak demand target and power limit, cannot exceed stored energy
            q[t] = 0
        

        # Decide charge
        elif SoC[t] < SoC_reserve[t] and D[t] <= p_current:
                q[t] = max(0, min((SoC_reserve[t] - SoC[t])/(eta * Ts),(E_batt-SoC[t])/(Ts*eta), P_batt, p_current-D[t]))
                # try to reach target SoC, cannot exceed battery energy or power limits, cannot exceed peak target
                d[t] = 0
            
        else:
            # No action
            q[t] = 0
            d[t] = 0

        net_demand[t] = D[t] + q[t] - d[t] # Net demand after battery action
        SoC[t+1] = SoC[t] - d[t]/eta*Ts + q[t]*eta*Ts

        # Update peak demand target if net demand exceeds it
        if net_demand[t] > p_current:
            p_current = net_demand[t]

    return net_demand, q, d, SoC[1:], p_current


#------------------------------------------------------------------------------------------------
# Kernel Regression Peak Shaving Pipeline
# Executes daily kernel regression predictions and real-time control, continuously updates training dataset
#------------------------------------------------------------------------------------------------
def kernel_regression_controller(test_df: pd.DataFrame, train_df: pd.DataFrame, training_net_demand: np.ndarray, 
                                 all_training_demand: np.ndarray, training_SoC: np.ndarray, summer_months: list,
                                 params: dict, E_batt: float, P_batt: float, E_min: float, e0: float, eta: float,
                                 Ts: float, mins_in_peak: int, verbose=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, 
                                 np.ndarray, np.ndarray, np.ndarray, dict]:

    """
    Runs the full peak-shaving pipeline: performs kernel regression using historical demand/SoC/max daily net demand,
    generates SoC reserve trajectory and peak target predictions, applies real-time control, updates historical datasets,
    computes monthly peak demand statistics

    Parameters
    ----------
    test_df: Test data
    train_df: Training data
    training_net_demand: Historical max daily net demand
    all_training_demand: Historical demand
    training_SoC: Historical SoC values
    summer_months: Months classified as summer (different regression subset)
    params: Kernel regression hyperparameters (sigma, K, window, alpha)
    E_batt: Battery energy capacity (MWh)
    P_batt: Battery power limit (MW)
    E_min: SoC lower bound (MWh)
    e0: Initial SoC (MWh)
    eta: Battery efficiency (round-trip)
    Ts: Timestep size (h)
    mins_in_peak: Duration of peak demand window for peak shaving (minutes)

    Returns
    -------
    PS_net_demand: Net demand after battery dispatch (MW)
    PS_q: Battery charging power (MW)
    PS_d: Battery discharging power (MW)
    PS_SoC: Battery SoC trajectory (MWh)
    regression_SoC: SoC reserve values predicted by kernel regression (MWh)
    p_record: Final peak target applied each time-step (MW)
    monthly_peaks: Dictionary of monthly original & optimized peaks (MW)
    """

    # ----------------------------------------------------
    # 1. Initialize Output Containers
    # ----------------------------------------------------
    PS_net_demand, PS_q, PS_d, PS_SoC, regression_SoC, p_record = [], [], [], [], [], []
    e_day_start = e0
    current_month_tracker = -1

    # ----------------------------------------------------
    # 2. Initialize Running Historical Arrays
    # ----------------------------------------------------
    D_hist_run = all_training_demand.copy()
    SoC_hist_run = training_SoC.copy()
    netD_hist_run = training_net_demand.copy()

    # Precomputes sin/cos time-of-day signals for training data
    train_minutes = train_df.index.hour * 60 + train_df.index.minute
    sin_time_run = np.sin(2 * np.pi * train_minutes / (24 * 60)).to_numpy()
    cos_time_run = np.cos(2 * np.pi * train_minutes / (24 * 60)).to_numpy()

    # ----------------------------------------------------
    # 3. Build Seasonal Subsets (Summer vs. Other Months)
    # ----------------------------------------------------
    hist_df = train_df.reset_index()
    summer_mask = hist_df['time'].dt.month.isin(summer_months)
    other_mask = ~summer_mask

    D_hist_summer = all_training_demand[summer_mask.values].copy()
    SoC_hist_summer = training_SoC[summer_mask.values].copy()
    netD_hist_summer = training_net_demand[summer_mask.values].copy()
    sin_summer = sin_time_run[summer_mask.values].copy()
    cos_summer = cos_time_run[summer_mask.values].copy()

    D_hist_other = all_training_demand[other_mask.values].copy()
    SoC_hist_other = training_SoC[other_mask.values].copy()
    netD_hist_other = training_net_demand[other_mask.values].copy()
    sin_other = sin_time_run[other_mask.values].copy()
    cos_other = cos_time_run[other_mask.values].copy()

    D_min_hist = np.min(all_training_demand)
    D_max_hist = np.max(all_training_demand)

    # ----------------------------------------------------
    # 4. Main Loop (Daily)
    # ----------------------------------------------------
    test_grouped = test_df.groupby([test_df.index.year, test_df.index.month, test_df.index.day])
    monthly_peaks = {}

    for (year, month, day), group in test_grouped:
        demand_day_raw = group['demand_MW'].values
        
        # ----------------------------------------------------
        # A. Monthly Peak Shaving Target Reset Logic
        # ----------------------------------------------------
        if month != current_month_tracker:
            current_month_tracker = month
            p_final = np.max(netD_hist_run[-int(48/Ts):]) 
            if verbose:
                print(f"→ Starting new month: {year}-{month:02d}")

        # ----------------------------------------------------
        # B. Select Seasonal Dataset
        # ----------------------------------------------------
        if month in summer_months:
            D_hist_target, SoC_hist_target, netD_hist_target = D_hist_summer, SoC_hist_summer, netD_hist_summer
            sin_hist_target, cos_hist_target = sin_summer, cos_summer
        else:
            D_hist_target, SoC_hist_target, netD_hist_target = D_hist_other, SoC_hist_other, netD_hist_other
            sin_hist_target, cos_hist_target = sin_other, cos_other

        # ----------------------------------------------------
        # C. Normalize Data
        # ----------------------------------------------------
        D_hist_norm = (D_hist_target - D_min_hist) / (D_max_hist - D_min_hist)
        D_test_norm = (demand_day_raw - D_min_hist) / (D_max_hist - D_min_hist)
        netD_hist_norm = (netD_hist_target - D_min_hist) / (D_max_hist - D_min_hist)
        e0_norm = e_day_start / E_batt
        minSoC_hist_norm = SoC_hist_target / E_batt

        # ----------------------------------------------------
        # D. Construct Recent-History Input Window for Kernel Regression
        # ----------------------------------------------------
        T_window = int(params['window'] / Ts)
        D_prev = (D_hist_run[-T_window:] - D_min_hist) / (D_max_hist - D_min_hist)
        sin_prev = sin_time_run[-T_window:]
        cos_prev = cos_time_run[-T_window:]

        minutes = group.index.hour * 60 + group.index.minute
        frac_day = minutes / (24 * 60)
        sin_test_arr = np.sin(2 * np.pi * frac_day).to_numpy()
        cos_test_arr = np.cos(2 * np.pi * frac_day).to_numpy()

        d_input = np.concatenate([D_prev, D_test_norm])
        sin_input = np.concatenate([sin_prev, sin_test_arr])
        cos_input = np.concatenate([cos_prev, cos_test_arr])

        # ----------------------------------------------------
        # E. Kernel Regression
        # ----------------------------------------------------
        SoC_reserve_norm, timestep_p_norm, _, _ = kernel_regression(D=d_input, D_hist=D_hist_norm, minSoC_hist=minSoC_hist_norm,
                                                  p_hist=netD_hist_norm, T=T_window, e0=e0_norm, sigma=params['sigma'],
                                                  K=params['K'], alpha_SoC=params['alpha'], sin_time_hist=sin_hist_target,
                                                  cos_time_hist=cos_hist_target, sin_time_test=sin_input, cos_time_test=cos_input)

        if SoC_reserve_norm is None:
            raise RuntimeError(f"KNN prediction failed on {year}-{month:02d}-{day:02d}")

        SoC_reserve = SoC_reserve_norm * E_batt
        timestep_p = timestep_p_norm * (D_max_hist - D_min_hist) + D_min_hist
        p_init = max(p_final, timestep_p[0])

        # ----------------------------------------------------
        # F. Real-Time Control
        # ----------------------------------------------------
        net_demand_day, q_test_day, d_test_day, SoC_day, p_final = real_time_control(
            demand_day_raw, SoC_reserve, Ts, eta, p_init, timestep_p, P_batt, E_batt, e_day_start)

        PS_net_demand.append(net_demand_day)
        PS_q.append(q_test_day)
        PS_d.append(d_test_day)
        PS_SoC.append(SoC_day)
        regression_SoC.append(SoC_reserve)
        p_record.append(np.full_like(net_demand_day, p_final))

        e_day_start = SoC_day[-1] # next day initial SoC is today's final SoC

        # ----------------------------------------------------
        # G. Online Learning Update
        # ----------------------------------------------------
        net_power_opt, power_opt, SoC_opt, d_opt, q_opt, _ = training_data_generation(
            E_batt, P_batt, E_min, SoC_hist_run[-1], eta, demand_day_raw, Ts, mins_in_peak, delta=0.01)

        # Append to chronological datasets
        D_hist_run = np.concatenate([D_hist_run, demand_day_raw])
        SoC_hist_run = np.concatenate([SoC_hist_run, SoC_opt])
        netD_hist_run = np.concatenate([netD_hist_run, net_power_opt])
        sin_time_run = np.concatenate([sin_time_run, sin_test_arr])
        cos_time_run = np.concatenate([cos_time_run, cos_test_arr])

        # Append to seasonal datasets
        if month in summer_months:
            D_hist_summer = np.concatenate([D_hist_summer, demand_day_raw])
            SoC_hist_summer = np.concatenate([SoC_hist_summer, SoC_opt])
            netD_hist_summer = np.concatenate([netD_hist_summer, net_power_opt])
            sin_summer = np.concatenate([sin_summer, sin_test_arr])
            cos_summer = np.concatenate([cos_summer, cos_test_arr])
        else:
            D_hist_other = np.concatenate([D_hist_other, demand_day_raw])
            SoC_hist_other = np.concatenate([SoC_hist_other, SoC_opt])
            netD_hist_other = np.concatenate([netD_hist_other, net_power_opt])
            sin_other = np.concatenate([sin_other, sin_test_arr])
            cos_other = np.concatenate([cos_other, cos_test_arr])

    # ----------------------------------------------------
    # 5. Final Output Concatenation
    # ----------------------------------------------------
    PS_net_demand = np.concatenate(PS_net_demand)
    PS_q = np.concatenate(PS_q)
    PS_d = np.concatenate(PS_d)
    PS_SoC = np.concatenate(PS_SoC)
    regression_SoC = np.concatenate(regression_SoC)
    p_record = np.concatenate(p_record)

    # ----------------------------------------------------
    # 6. Monthly Peak Statistics
    # ----------------------------------------------------
    test_df_temp = test_df.iloc[:len(PS_net_demand)].copy()
    test_df_temp["net_demand_opt"] = PS_net_demand
    test_df_temp["demand_orig"] = test_df["demand_MW"].values[:len(PS_net_demand)]

    for month, group in test_df_temp.groupby(test_df_temp.index.month):
        window_size = int(mins_in_peak / 60 / Ts)
        rolling_orig = np.convolve(group["demand_orig"], np.ones(window_size)/window_size, mode='valid')
        rolling_opt = np.convolve(group["net_demand_opt"], np.ones(window_size)/window_size, mode='valid')
        monthly_peaks[month] = {"original_peak_MW": np.max(rolling_orig), "optimized_peak_MW": np.max(rolling_opt)}

    return PS_net_demand, PS_q, PS_d, PS_SoC, regression_SoC, p_record, monthly_peaks
    

#------------------------------------------------------------------------------------------------
# Add in Arbitrage to Peak Shaving Results
#------------------------------------------------------------------------------------------------
def add_arbitrage(D, q_PS, d_PS, SoC_PS, P_batt, E_batt, Ts, p, eta, q_arb, d_arb, e0):

    """
    Combine arbitrage actions with existing peak-shaving actions while enforcing
    battery power/energy limits and peak shaving target.

    Parameters
    ----------
    D: Demand profile (MW)
    q_PS: Charging actions from the peak-shaving controller (MW)
    d_PS: Discharging actions from the peak-shaving controller (MW)
    SoC_PS: SoC trajectory from the peak-shaving controller (used for discharge bounds) (MWh)
    P_batt: Battery power rating (MW)
    E_batt: Battery energy capacity (MWh)
    Ts: Timestep size (h)
    p: Peak-shaving target
    eta: Battery efficiency
    q_arb: Arbitrage-only charging schedule (MW).
    d_arb: Arbitrage-only discharging schedule (MW).
    e0: Initial SoC at the start of the evaluation window (MWh).

    Returns
    -------
    SoC: SoC trajectory for both peak shaving and arbitrage (MWh)
    net_demand: Final net demand after applying peak shaving and arbitrage (MW)
    q: Charging profile (peak shaving and arbitrage) (MW)
    d: Discharging profile (peak shaving and arbitrage) (MW)
    """

    T = len(q_PS)
    SoC = np.zeros(T+1)
    q = np.zeros(T)
    d = np.zeros(T)
    
    net_demand = np.zeros(T)

    SoC[0] = e0 

    for t in range(T):
        SoC_prev = SoC[t]  # SoC at start of timestep
        #---------------------------------------
        # Feasible charging/discharging limits
        #---------------------------------------
        max_charge = min(P_batt, max(0, p[t] - D[t]), (E_batt - SoC_prev)/(eta*Ts))
        max_discharge = min(P_batt, max(0, (SoC_prev - SoC_PS[t]) * eta / Ts))

        #---------------------------------------
        # Case 1: Peak shaving is charging
        #---------------------------------------
        if q_PS[t] > 0:
            q[t] = q_PS[t] + q_arb[t]
            q[t] = np.clip(q[t], 0, max_charge)
            d[t] = 0

        #---------------------------------------
        # Case 2: Peak shaving is discharging
        #---------------------------------------
        elif d_PS[t] > 0:
            d[t] = d_PS[t] + d_arb[t]
            d[t] = np.clip(d[t], 0, max_discharge)
            q[t] = 0

        #---------------------------------------
        # Case 3: Only arbitrage is occurring
        #---------------------------------------
        else:
            q[t] = q_arb[t]
            q[t] = np.clip(q[t], 0, max_charge)
            d[t] = d_arb[t]
            d[t] = np.clip(d[t], 0, max_discharge)
            

        # Update SoC
        SoC[t+1] = SoC_prev + (q[t]*eta - d[t]/eta) * Ts

        # Compute net demand
        net_demand[t] = D[t] + q[t] - d[t]

    return SoC, net_demand, q, d

    
#------------------------------------------------------------------------------------------------
# Deterministic Controller (with Maximum Annual Cycles Enforced)
#------------------------------------------------------------------------------------------------
def deterministic_controller(
    E_batt, P_batt, E_min, e0, eta, 
    D_full,          # Full-year demand (e.g., 8760 timesteps)
    prices_full,     # Full-year prices
    Ts, c, delta,
    monthly_peak_prices, # Array/list of 12 peak prices
    month_timesteps,     # Array/list of 12 counts (e.g., [744, 672, ...])
    max_annual_cycles=None, # Annual cycle limit
    arb=True):

    '''
    Solves a joint peak-shaving and arbitrage optimization over an entire year using Gurobi.
       Enforces monthly peak demand limits and a maximum annual cycle limit
       Arbitrage is optional (set arb=False to turn off)

    Parameters
    ----------
      E_batt: Battery energy capacity (MWh)
      P_batt: Battery power limit (MW)
      E_min: SoC lower bound (MWh)
      e0: Initial SoC (MWh)
      eta: Round-trip efficiency
      D_full: Demand profile for the entire year (MW)
      prices_full: Energy prices for each timestep ($/MWh)
      Ts: Timestep size (h)
      c: Degradation cost coefficient ($/MWh)
      delta: SoC tiebreaking penalty
      monthly_peak_prices: Array of 12 monthly peak demand charge rates ($/MW)
      month_timesteps: Array of 12 counts for number of timesteps per month
      max_annual_cycles: Optional cap on total annual battery cycles
      arb: Enables arbitrage when True; disables when False
    
    Returns
    ----------
      net_demand_opt: Optimized net demand profile (MW)
      e_opt: Optimal SoC trajectory (MWh)
      d_opt, q_opt: Discharge/charge power profiles (MW)
      p_values: Monthly peak demand values (MW)
    '''

    T = len(D_full)          # Total timesteps in year
    M = len(month_timesteps) # Number of months
    
    if T != sum(month_timesteps):
        raise ValueError("Total timesteps (T) does not match sum of month_timesteps")
    if M != len(monthly_peak_prices):
         raise ValueError("Number of months (M) does not match monthly_peak_prices")

    # disable arbitrage if needed 
    if not arb:
        prices_full = np.zeros_like(prices_full)
        c = 0

    # --------------------------------------
    # Build Gurobi Model
    # --------------------------------------
    model = gp.Model("deterministic_PS_full_year")
    model.Params.OutputFlag = 0

    # Variables
    d = model.addVars(T, lb=0, ub=P_batt, name="discharge") # discharge power
    q = model.addVars(T, lb=0, ub=P_batt, name="charge")    # charge power
    e = model.addVars(T, lb=E_min, ub=E_batt, name="SoC")   # SoC
    
    # Monthly peak variables
    p = model.addVars(M, vtype=GRB.CONTINUOUS, name="monthly_peak")
    
    # --------------------------------------
    # Objective Function
    # --------------------------------------
    # Energy charge, degradation cost, SOC tiebreaker
    energy_cost = gp.quicksum(prices_full[t]*(D_full[t] - d[t] + q[t])*Ts + 1e-6*c*d[t]*Ts + delta*e[t] for t in range(T))
    
    # Monthly peak cost
    peak_cost = gp.quicksum(monthly_peak_prices[m] * p[m] for m in range(M))
    
    model.setObjective(energy_cost + peak_cost, GRB.MINIMIZE)

    # --------------------------------------
    # Objective Function
    # --------------------------------------
    # SOC dynamics
    model.addConstr(e[0] == e0 - d[0]/eta*Ts + q[0]*eta*Ts, name="SoC_init")
    model.addConstrs((e[t] == e[t-1] - d[t]/eta*Ts + q[t]*eta*Ts for t in range(1, T)), name="SoC_update")

    # Monthly Peak Constraints
    start_t = 0
    for m in range(M):
        num_steps_in_month = month_timesteps[m]
        model.addConstrs((p[m] >= D_full[t] - d[t] + q[t] for t in range(start_t, start_t + num_steps_in_month)),
            name=f"peak_month_{m}")
        start_t += num_steps_in_month 

    # Annual cycle limit
    if max_annual_cycles is not None:
        total_throughput = gp.quicksum(d[t]/eta + q[t]*eta for t in range(T)) * Ts
        total_cycles = (total_throughput / 2.0) / E_batt
        
        model.addConstr(total_cycles <= max_annual_cycles, name="annual_cycle_limit")

    # --------------------------------------
    # Solve
    # --------------------------------------
    model.Params.OutputFlag = 0
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print("Optimization was not successful")
        print("Gurobi status:", model.status)
        return None # Return None to signal failure

    # --------------------------------------
    # Extract Results
    # --------------------------------------
    d_opt = np.array([d[t].X for t in range(T)])
    q_opt = np.array([q[t].X for t in range(T)])
    e_opt = np.array([e[t].X for t in range(T)])

    # Clean numerical noise
    d_opt[d_opt < 1e-6] = 0
    q_opt[q_opt < 1e-6] = 0

    power = d_opt - q_opt
    net_demand_opt = D_full - power
    p_values = np.array([p[m].X for m in range(M)]) # monthly peak values
    
    # Annual cycle number
    final_cycles = (np.sum(d_opt/eta + q_opt*eta) * Ts / 2.0) / E_batt
    print(f"Optimization complete. Total cycles: {final_cycles:.2f}")

    return net_demand_opt, e_opt, d_opt, q_opt, p_values


# -------------------------------------------------------------------------------------------------
# Hierarchical Parameter Search
# -------------------------------------------------------------------------------------------------
def hierarchical_search(test_df: pd.DataFrame, train_df: pd.DataFrame, training_net_demand: np.ndarray, 
                        all_training_demand: np.ndarray, training_SoC: np.ndarray, summer_months: list, E_batt: float, 
                        P_batt: float, E_min: float, e0: float, eta: float, Ts: float, q_arb, d_arb, mins_in_peak: int,
                        base_params: dict, window_candidates: list, sigma_candidates: list, K_candidates: list,
                        WINDOW_MIN: float, WINDOW_MAX: float, SIGMA_MIN: float, SIGMA_MAX: float, K_MIN: int, K_MAX: int,
                        window_step=0.5, sigma_step=0.05, K_step=50, conv_thresh=50.0, monthly_charge=71, 
                        RTP=None, autosave_path=None):
    """
    Performs a hierarchical search (window -> sigma -> K) to find the best hyperparameters for a kernel regression-based
    peak shaving and arbitrage controller. Hyperparameters are chosen based on the max cost savings (with peak shaving
    and arbitrage). The search is conducted in stages for each hyperparameter: a coarse grid search followed by 
    greedy local refinement. Previous results are cached so searches can be stopped and resumed.
    
    Parameters
    ----------
    test_df: Test dataset containing demand and timestamp information
    train_df: Training dataset 
    training_net_demand: Historical net demand values (MW)
    all_training_demand: Historical demand data (MW)
    training_SoC: Historical battery SoC (MWh)
    summer_months: List of months considered "summer" for season-specific calculations.
    E_batt: Battery energy capacity (MWh)
    P_batt: Battery power capacity (MW)
    E_min: SoC lower bound (MWh)
    e0: Initial SoC (MWh)
    eta: Round-trip efficiency of the battery
    Ts: Timestep size (h)
    q_arb, d_arb: Arbitrage charge/discharge schedules (MW)
    mins_in_peak: Duration of peak demand calculation (min)
    base_params: Initial hyperparameters for the search (window, sigma, K, alpha)
    window_candidates, sigma_candidates, K_candidates: Lists of coarse candidate values for each hyperparameter
    WINDOW_MIN, WINDOW_MAX: Bounds for refining the window hyperparameter
    SIGMA_MIN, SIGMA_MAX: Bounds for refining the sigma hyperparameter
    K_MIN, K_MAX: Bounds for refining the K hyperparameter.
    window_step, sigma_step, K_step: Step sizes for refined search
    conv_thresh: Minimum savings improvement ($) required to continue refinement
    monthly_charge: Fixed monthly charge used in bill calculations ($)
    RTP: Real-time energy price array ($/MWh) for bill calculation.
    autosave_path: Path to CSV file to save intermediate results and resume search.

    Returns
    ----------
    best_global: Dictionary containing the best hyperparameter combination found and associated metrics:
        - 'params': dict of optimal hyperparameters
        - 'savings_usd': savings compared to no storage ($)
        - 'bill_PSA': total bill with peak shaving and arbitrage ($)
        - 'bill_PS': total bill with peak shaving only ($)
        - 'cycle_number_PSA': total battery cycles for peak shaving and arbitrage ($)
        - 'status': 'ok' or 'failed'
    all_results_df : pd.DataFrame
        DataFrame containing all evaluated parameter sets with their associated performance metrics.

    """
    all_results = []
    tested_keys = set()
    best_global = None

    # ----------------------------------------
    # Load previous results if available
    # ----------------------------------------
    if autosave_path and os.path.exists(autosave_path):
        print(f"Loading previous results from {autosave_path}...")
        try:
            previous_results_df = pd.read_csv(autosave_path)
            
            # Convert DataFrame records to dictionaries
            all_results.extend(previous_results_df.to_dict('records'))

            # Rebuild the tested_keys set
            for record in all_results:
                params = ast.literal_eval(record['params']) 
                key = (round(params['window'], 2), round(params['sigma'], 6), int(params['K']), round(params['alpha'], 2))
                tested_keys.add(key)
                
            if all_results:
                # Find the global best from the loaded data to continue the search
                best_global = max(all_results, key=lambda x: x['savings_usd'])
                print(f"Loaded {len(all_results)} previous runs. Current max savings: ${best_global['savings_usd']:,.2f}")
            
        except Exception as e:
            print(f"No previous results from {e}. Starting fresh.")
            all_results = [] # Reset to empty lists if loading fails
            tested_keys = set()
            best_global = None

    # ----------------------------------------
    # Run Simulation and Record Results
    # ----------------------------------------
    def run_and_record(params):
        '''Run peak shaving and arbitrage with given params and record results.'''
        key = (round(params['window'], 2), round(params['sigma'], 6), int(params['K']), round(params['alpha'], 2))
        
        if key in tested_keys:
            return None
        
        tested_keys.add(key)

        try:
            # Peak Shaving
            PS_net_demand, PS_q, PS_d, PS_SoC, regression_SoC, p_record, _ = kernel_regression_controller(
                test_df, train_df, training_net_demand, all_training_demand, training_SoC,
                summer_months, params, E_batt, P_batt, E_min, e0, eta, Ts, mins_in_peak, verbose=False)

            # Arbitrage
            SoC_comb, net_demand_comb, q_comb, d_comb = add_arbitrage(test_df['demand_MW'].values,
                PS_q, PS_d, PS_SoC, P_batt, E_batt, Ts, p_record, eta, q_arb, d_arb, e0)

            # Bill calculation
            bill_PSA = calculate_total_bill(net_demand_comb, test_df.index, Ts, mins_in_peak, monthly_charge, RTP)
            bill_no_storage = calculate_total_bill(test_df['demand_MW'].values, test_df.index, Ts, mins_in_peak, monthly_charge, RTP)
            bill_PS = calculate_total_bill(PS_net_demand, test_df.index, Ts, mins_in_peak, monthly_charge, RTP)
            savings_usd = bill_no_storage - bill_PSA
            cycles_PSA = sum(d_comb)/eta*Ts/E_batt

            # Record results
            params_clean = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in params.items()}
            result = {'params': params_clean, 'bill_PSA': float(bill_PSA), 'bill_PS': float(bill_PS),
                'savings_usd': float(savings_usd), 'cycle_number_PSA': float(cycles_PSA), 'status': 'ok'}
            
            all_results.append(result)
            if autosave_path:
                pd.DataFrame(all_results).to_csv(autosave_path, index=False)

            return result

        except Exception as e:
            print(f"Failed params {params}: {e}")
            return {'status': 'failed', 'params': params, 'error': str(e), 'savings_usd': -np.inf}

    # -----------------------------
    # Initialize search with baseline params
    # -----------------------------
    current_params = base_params.copy()
            
    if best_global is None:
        # If no results were loaded frome existing csv, run the baseline
        print(f"--- Starting Hierarchical Search ---")
        print(f"Baseline Params: {current_params}")
        best_global = run_and_record(current_params)
        if best_global is None or best_global['status'] == 'failed':
             raise RuntimeError("Baseline simulation failed.")
        print(f"Baseline Savings: ${best_global['savings_usd']:,.2f}")
    else:
        # If best_global is loaded from csv, update current_params to the best found so far
        # best params will serve as new baseline
        loaded_params_str = best_global['params']
        loaded_params_dict = ast.literal_eval(loaded_params_str)
        current_params = loaded_params_dict.copy()
        
        print(f"--- Continuing Hierarchical Search ---")
        print(f"Current Best Params (Loaded): {current_params}")
        print(f"Max Savings (Loaded): ${best_global['savings_usd']:,.2f}")

    # ----------------------------------------
    # Stages: Window -> Sigma -> K (Coarse + Refined)
    # ----------------------------------------
    
    for param_name, candidates, step, min_val, max_val, refine_type in [
        ('window', window_candidates, window_step, WINDOW_MIN, WINDOW_MAX, 'linear'),
        ('sigma', sigma_candidates, sigma_step, SIGMA_MIN, SIGMA_MAX, 'log'),
        ('K',      K_candidates,      K_step, K_MIN,      K_MAX,      'linear')]:

        # Coarse search
        print(f"\n--- Coarse {param_name.capitalize()} Search ---")
        for val in candidates:
            p = current_params.copy()
            p[param_name] = val
            res = run_and_record(p)
            if res and res['status']=='ok' and res['savings_usd']>best_global['savings_usd']:
                best_global = res
                current_params[param_name] = val
                print(f"   → New best {param_name}: {val} (Savings: ${res['savings_usd']:.2f})")

        # Refined Search
        print(f"--- Refined {param_name.capitalize()} Search ---")
        improving = True
        while improving:
            current_val = current_params[param_name]
            # Generate neighbors based on refinement type, filter by bounds
            if refine_type == 'linear':
                neighbors = [v for v in [current_val - step, current_val + step] if min_val <= v <= max_val]
            elif refine_type == 'log':
                log_val = np.log10(current_val)
                neighbors = [10**(log_val - step), 10**(log_val + step)]
                neighbors = [v for v in neighbors if min_val <= v <= max_val]

            # Test neighbors
            step_best_res = None
            for v in neighbors:
                p = current_params.copy()
                p[param_name] = v
                res = run_and_record(p)
                if res and res['status']=='ok' and (step_best_res is None or res['savings_usd']>step_best_res['savings_usd']):
                    step_best_res = res

            # Compare step best to global best
            if step_best_res:
                improvement = step_best_res['savings_usd'] - best_global['savings_usd']
                if improvement >= conv_thresh:
                    best_global = step_best_res
                    current_params[param_name] = step_best_res['params'][param_name]
                    improving = True
                    print(f"   → Refined {param_name}: {current_params[param_name]} (Savings: ${step_best_res['savings_usd']:.2f})")
                else:
                    improving = False
            else:
                improving = False

    # ----------------------------------------
    # Final Output
    # ----------------------------------------
    print("\nHierarchical Search Complete")
    print("Best Parameters:", best_global['params'])
    print(f"Max Savings: ${best_global['savings_usd']:.2f}")

    return best_global, pd.DataFrame(all_results)
