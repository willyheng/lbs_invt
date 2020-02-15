from scipy.optimize import minimize
import numpy as np, pandas as pd
import sys, os

def continuous_to_annual(rate):
    return np.exp(rate)-1

def annual_to_continuous(rate):
    return np.log(1+rate)

def cf_pv(cf, time, rates):
    """Calculate the present value of a list of cashflows

    Args:
        cf (list): Cashflows
        time (list): Timing of each corresponding cashflow
        rates (float or list): Discount rates for corresponding time, if only int is provided, it is used for all periods

    Returns:
        float: Present value of cashflows
    """
    if isinstance(rates, list):
        return sum([discount(cf, r, t) for cf, t, r in zip(cf, time, rates)])
    else:
        return sum([discount(cf, rates, t) for cf, t in zip(cf, time)])

def ytm_from_cf(price, cf_df):
    """Calculate yield to maturity from dataframe of cashflows
    
    Uses scipy.optimize.minize to calculate the yield to maturity of the bond

    Args:
        price (float): Price of the bond
        cf_df (list): Dataframe of cashflows with columns 'cashflow' and 'time'

    Returns:
        float: Yield to maturity of the bond
    """
    
    return minimize(lambda x: (price - cf_pv(cf_df.cashflow, cf_df.time, x))**2, [-0.01]).x[0]

def discount(pmt, r, t, method = "continuous"):   
    return pmt / ((1 + r)**t) if method == "annual" else pmt / np.exp(r * t)

def bond_cf(face, coupon, years, freq):
    """Generates the cashflows of a bond given parameters
    
    Args:
        face (float): Face value of the bond
        coupon (float): Coupon of the bond as a fraction of face, in decimals
        years (float): Years to expiry
        freq (int): How often the bond pays interests in a year

    Returns:
        DataFrame: contains columns 'cashflow' and 'time'
    """
    return pd.DataFrame({'cashflow': [(coupon/freq + (1 if t == int(years*freq) else 0)) * face for t in range(1, int(years * freq)+1) ],
                         'time': [t / freq for t in range(1, int(years * freq)+1)]})

def bond_pv(face, coupon, years, ytm, freq = 1):
    """Calculate present value of a bond based on its parameters and yield to maturity (continuous compounded)
    
    Args:
        face (float): Face value of the bond
        coupon (float): Coupon of the bond as a fraction of face, in decimals
        years (float): Years to expiry
        ytm (float): Yield to maturity
        freq (int): How often the bond pays interests in a year

    Returns:
        float: Yield to maturity of the bond
    """
        
    return sum([discount((coupon/freq + (1 if t == int(years*freq) else 0)) * face, ytm/freq, t) for t in range(1, int(years*freq)+1)])

def ytm_from_param(face, pv, coupon, years, freq):
    """Calculate yield to maturity given bond parameters
    
    Args:
        face (float): Face value of the bond
        pv (float): Price of the bond
        coupon (float): Coupon of the bond as a fraction of face, in decimals
        years (float): Years to expiry
        freq (int): How often the bond pays interests in a year

    Returns:
        float: Yield to maturity of the bond
    """
        
    return minimize(lambda x: (pv - bond_pv(face, coupon, years, x, freq))**2, [0.01]).x[0]

def pv_from_df(df, term_structure, max_rate = np.nan, debug = False):
    """Calculate the present value of a bond from a dataframe of cashflows
    
    Args:
        df (DataFrame): Contains the columns 'cashflow' and 'time'
        term_structure (DataFrame): Contains columns index of time and column for yield
        max_rate (float): If the term_structure does not sufficient periods compared to bond, this will be used as the last 
            period rate, default of np.nan is for optimization purposes
        debug (boolean): returns the dataframe of discounted cashflows when True, defaults to False

    Returns:
        float: Present value of cashflows (returns tuple if debug = True)
    """
    
    df = df.copy()
    df['min_t'] = df.apply(lambda x: max(term_structure.index[term_structure.index <= x.time], default = min(term_structure.index)),axis = 1)
    df['min_rate'] = df.apply(lambda x: term_structure.loc[x.min_t], axis = 1)

    df['max_t'] = df.apply(lambda x: min(term_structure.index[term_structure.index >= x.time], default = np.nan),axis = 1)
    df['max_rate'] = df.apply(lambda x: term_structure.loc[x.max_t] if not np.isnan(x.max_t) else np.nan, axis = 1)
    df['max_t'] = df.max_t.fillna(df.time.max())
    
    df['max_rate'] = df.max_rate.fillna(max_rate)
    df['rate'] = df.apply(lambda x: 
                      x.min_rate if x.min_t == x.max_t 
                      else (x.min_rate * (x.max_t - x.time) + x.max_rate * (x.time - x.min_t)) / (x.max_t - x.min_t), axis = 1)
    
    if any(np.isnan(df.max_rate)):
        sys.exit("Insufficient rates in term_structure and max_rate not provided")
    
    df['discounted'] = df.cashflow / (1 + df.rate) ** df.time  
    
    return df.discounted.sum() if not debug else (df.discounted.sum(), df)

def __bootstrap_single_bond(bond_price, bond_cf, term_structure = None):
    """Calculate the latest rate in the term_structure using a single bond
    
    If term_structure is None, calculates the yield to maturity of the bond, used to bootstrap the first bond 
    
    Args:
        bond_price (float): Current bond price
        bond_cf (DataFrame): Contains the columns 'cashflow' and 'time'
        term_structure (DataFrame): Contains columns index of time and column for yield, defaults to None

    Returns:
        DataFrame: Updated term_structure
    """
    
    if term_structure is None:
        new_rate = ytm_from_cf(bond_price, bond_cf)
    else:
        new_rate = minimize(lambda x: (bond_price - pv_from_df(bond_cf, term_structure, x[0]))**2, [0.001]).x[0]
        if (abs(bond_price - pv_from_df(bond_cf, term_structure, new_rate)) > 0.01):
            sys.exit("Error: Could not find a rate that allows price to converge\nExpected bond price: {}\nCalculated bond price: {}\nProjected rate: {}".format(bond_price, pv_from_df(bond_cf, new_rate, term_structure), new_rate))
    new_period = bond_cf.time.max()

    term_structure = pd.concat([term_structure, 
              pd.DataFrame({"rate": new_rate}, index = [new_period])])
    
    return term_structure

def bootstrap(prices, bonds):
    """Calculate the term_structure using all the bonds
        
    Args:
        prices (list): List of bond prices
        bonds (list): List of DataFrames, each with columns 'cashflow' and 'time'

    Returns:
        DataFrame: Updated term_structure
    """
    sorted_bonds = [b for _, b in sorted(zip([x.time.max() for x in bonds], bonds))]
    
    term_structure = None
    for price, bond in zip(prices, sorted_bonds):
        term_structure = __bootstrap_single_bond(price, bond, term_structure)
    
    return term_structure