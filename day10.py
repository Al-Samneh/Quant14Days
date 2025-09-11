"""
Day 10: Professional SPX European Option Pricing and Greeks Analysis

This retrieves near-term European call option data for the S&P 500 Index (^SPX)
using the yfinance library. It then calculates the theoretical price and Greeks 
(Delta, Gamma, Vega, Theta, Rho) for each option using two distinct methods:
1.  The analytical Black-Scholes model.
2.  A Monte Carlo simulation with finite differences for the Greeks.

The reason we utilize the European option is because we can use the Black-Scholes
formula to price it, which is more accurate than the American option.

PROFESSIONAL TRADING LOGIC:
- Uses ASK prices (not midpoint) for buy decisions to reflect executable costs
- Filters for liquid options only (volume > 50, open interest > 100)
- Requires tight bid-ask spreads (<10% of ask) for reliable execution
- Compares executable ask price directly to theoretical Black-Scholes value
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Monte Carlo Simulation Parameters
MONTE_CARLO_SIMULATIONS = 100_000 # Number of paths for the simulation

# Finite Difference Parameters for Greeks
SPOT_BUMP_SIZE = 1.0              # Spot price move for Delta and Gamma
VOL_BUMP_SIZE = 0.01              # Absolute volatility move (1%) for Vega
TIME_BUMP_DAYS = 0.25             # Time decay step in days for Theta
RATE_BUMP_SIZE = 0.0001           # Interest rate move (1 basis point) for Rho

# Trading Logic & Market Data Parameters
# A negative threshold means we look for options where the market price is
# cheaper than the theoretical price. e.g., -0.05 means market price must
# be at least 5% below the theoretical value to be flagged as a "BUY".
UNDERPRICING_THRESHOLD = -0.05

RISK_FREE_RATE = 0.0433 # Risk Free Rate of a US Treasury Bond
MIN_TIME_TO_EXPIRY = 1 / 365.0    # Min time in years to avoid numerical instability
SPX_TICKER = "^SPX"               # Ticker for S&P 500 Index
EXPIRY_LOOKAHEAD_DAYS = 7         # How many days into the future to scan for expiries

np.random.seed(42) # For reproducible Monte Carlo results

# Pre-generate random numbers for efficiency
_PRECOMPUTED_RANDOMS = None

def get_random_numbers(num_simulations: int) -> np.ndarray:
    """Get pre-computed random numbers for Monte Carlo simulation."""
    global _PRECOMPUTED_RANDOMS
    if _PRECOMPUTED_RANDOMS is None or len(_PRECOMPUTED_RANDOMS) < num_simulations:
        _PRECOMPUTED_RANDOMS = np.random.randn(max(num_simulations, MONTE_CARLO_SIMULATIONS))
    return _PRECOMPUTED_RANDOMS[:num_simulations]


def days_to_years(days: float) -> float:
    """Converts a number of days to a fraction of a year.
    
    Uses 365 days (calendar time) which is standard for option pricing.
    Note: 252 is used for business days in volatility calculations, but
    Black-Scholes uses calendar time for time to expiry.
    """
    return days / 365.0


def bs_price_and_greeks(
    S0: float, K: float, T: float, r: float, sigma: float
) -> Dict[str, float]:
    """
    Calculates the analytical price and Greeks for a European call option
    using the Black-Scholes model.

    Args:
        S0: Current spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiry in years.
        r: Risk-free interest rate (annualized).
        sigma: Implied volatility of the underlying asset (annualized).

    Returns:
        A dictionary containing the calculated Price and Greeks.
    """
    # Input validation
    if S0 <= 0 or K <= 0 or sigma <= 0:
        raise ValueError("S0, K, and sigma must be positive")
    if T < 0:
        raise ValueError("Time to expiry cannot be negative")
    
    out = {g: 0.0 for g in ['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']}
    
    if T < MIN_TIME_TO_EXPIRY:
        intrinsic_value = max(S0 - K, 0)
        out.update({
            'Price': intrinsic_value,
            'Delta': 1.0 if S0 > K else 0.0,
        })
        return out

    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - (sigma * sqrtT)

    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)

    # --- Greeks ---
    # Delta: Sensitivity to a change in the underlying spot price
    delta = norm.cdf(d1)
    # Gamma: Rate of change of Delta
    gamma = pdf_d1 / (S0 * sigma * sqrtT)
    # Vega: Sensitivity to a change in volatility (returns price change per 1% vol change)
    vega = S0 * pdf_d1 * sqrtT * 0.01
    # Theta: Sensitivity to the passage of time (returns price change per calendar day)
    theta = (-S0 * pdf_d1 * sigma / (2 * sqrtT) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    # Rho: Sensitivity to a change in the risk-free rate (returns price change per 1% rate change)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01

    out.update({
        'Price': price,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    })
    return out


def monte_carlo_price_and_greeks(
    S0: float, K: float, T: float, r: float, sigma: float,
    num_simulations: int = MONTE_CARLO_SIMULATIONS,
    h: float = SPOT_BUMP_SIZE, dv: float = VOL_BUMP_SIZE,
    dt_days: float = TIME_BUMP_DAYS, dr: float = RATE_BUMP_SIZE
) -> Dict[str, float]:
    """
    Calculates the price & Greeks for a European call option using a Monte Carlo
    simulation with finite differences for the Greeks. Implements the common random
    numbers technique to reduce variance.

    Args:
        S0, K, T, r, sigma (as we did before)
        num_simulations: Number of Monte Carlo paths to generate.
        h: Finite difference step for spot price.
        dv: Finite difference step for volatility.
        dt_days: Finite difference step for time in days.
        dr: Finite difference step for risk-free rate.

    Returns:
        A dictionary containing the calculated Price and Greeks.
    """
    out = {g: 0.0 for g in ['Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']}

    if T < MIN_TIME_TO_EXPIRY:
        intrinsic_value = max(S0 - K, 0)
        out.update({
            'Price': intrinsic_value,
            'Delta': 1.0 if S0 > K else 0.0,
        })
        return out

    # 1. Generate a single set of random numbers for all calculations
    # This is the core of the "common random numbers" variance reduction technique.
    Z = get_random_numbers(num_simulations)

    # 2. Define a reusable pricing function
    def price_path(spot, time, rate, vol):
        drift = (rate - 0.5 * vol**2) * time # Drift is the expected return of the asset (Term 1)
        diffusion = vol * np.sqrt(time) * Z # Diffusion is the random shock of the asset (Term 2)
        ST = spot * np.exp(drift + diffusion) # ST is the stock price at time T
        payoff = np.maximum(ST - K, 0) # Payoff is the maximum of the difference between the stock price and the strike price
        return np.mean(payoff) * np.exp(-rate * time) # Return the mean of the payoff and the discount factor

    # 3. Calculate base price and prices for bumped parameters
    V0 = price_path(S0, T, r, sigma) # Base price
    
    # For Delta and Gamma, to reduce the variance, while not increasing the simulation count, faster and more efficient.
    V_plus_h = price_path(S0 + h, T, r, sigma)
    V_minus_h = price_path(S0 - h, T, r, sigma)
    
    # For Vega
    V_plus_dv = price_path(S0, T, r, sigma + dv)
    
    # For Theta
    dt = min(days_to_years(dt_days), T / 2) # Ensure time step doesn't exceed half the expiry
    V_minus_dt = price_path(S0, T - dt, r, sigma)
    
    # For Rho
    V_plus_dr = price_path(S0, T, r + dr, sigma) # I had initially bumped the risk-free rate and the volatility, but it was not necessary

    # 4. Calculate Greeks using finite differences
    delta = (V_plus_h - V_minus_h) / (2 * h)
    gamma = (V_plus_h - 2 * V0 + V_minus_h) / (h**2)
    vega = (V_plus_dv - V0) / dv # Vega per 100% change, multiply by 0.01 for 1%
    theta = (V_minus_dt - V0) / dt # Theta per year, divide by 365 for daily
    rho = (V_plus_dr - V0) / dr # Rho per 100% change, multiply by 0.01 for 1%

    out.update({
        'Price': V0,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega * 0.01,
        'Theta': theta / 365.0,
        'Rho': rho * 0.01
    })
    return out

def analyze_option(call_data: pd.Series, S0: float, expiry: pd.Timestamp, today: pd.Timestamp) -> Optional[Dict]:
    """
    Analyze a single option and return pricing results.
    
    Args:
        call_data: Series containing option data (strike, IV, bid, ask, etc.)
        S0: Current spot price
        expiry: Option expiry date
        today: Current date
        
    Returns:
        Dictionary with analysis results or None if option should be skipped
    """
    K = call_data['strike']
    sigma = call_data['impliedVolatility']
    
    # Use the actual executable price (ask) for buy decisions
    bid = call_data.get('bid', 0)
    ask = call_data.get('ask', 0)
    
    # Must have valid bid and ask prices (already filtered in main, but double-check)
    if bid <= 0 or ask <= 0:
        return None
    
    # Use ask price as the executable market price for buying
    market_price = ask
    midpoint_price = (bid + ask) / 2  # Keep for comparison/display
    
    T_days = (expiry - today).days
    T_years = max(days_to_years(T_days), MIN_TIME_TO_EXPIRY)
    
    try:
        bs = bs_price_and_greeks(S0, K, T_years, RISK_FREE_RATE, sigma)
        mc = monte_carlo_price_and_greeks(S0, K, T_years, RISK_FREE_RATE, sigma)
        
        # Professional Decision Logic: Compare executable ask price to theoretical value
        if ask < bs['Price']:
            price_diff = (ask - bs['Price']) / bs['Price']
            decision = f"BUY (Ask ${ask:.2f} is {abs(price_diff):.1%} below theoretical ${bs['Price']:.2f})"
        else:
            decision = f"HOLD (Ask ${ask:.2f} >= theoretical ${bs['Price']:.2f})"
        
        return {
            'strike': K,
            'ask_price': ask,
            'bid_price': bid,
            'midpoint_price': midpoint_price,
            'implied_vol': sigma,
            'bs_results': bs,
            'mc_results': mc,
            'decision': decision
        }
    except (ValueError, ZeroDivisionError) as e:
        print(f"Error analyzing option with strike {K}: {e}")
        return None


def print_option_analysis(result: Dict):
    """Print formatted analysis results for a single option."""
    print(f"\n--- Strike: {result['strike']: <8} | Bid: ${result['bid_price']:<5.2f} | Ask: ${result['ask_price']:<5.2f} | Mid: ${result['midpoint_price']:<5.2f} | IV: {result['implied_vol']:.3f} ---")
    print(f"{'Metric':<10} | {'Black-Scholes':<15} | {'Monte Carlo':<15}")
    print("-" * 55)
    for key in result['bs_results']:
        print(f"{key:<10} | {result['bs_results'][key]:<15.4f} | {result['mc_results'][key]:<15.4f}")
    print("-" * 55)
    print(f"Decision: {result['decision']}")


# --------------------------------------------------------------------------
def main():
    """Main execution function to fetch data and perform analysis."""
    try:
        ticker = yf.Ticker(SPX_TICKER)
        S0 = ticker.history(period='1d')['Close'].iloc[-1]
        expirations = list(ticker.options)
        
        if not expirations:
            print("No option expirations found for ticker.")
            return

    except Exception as e:
        print(f"Failed to retrieve market data for {SPX_TICKER}: {e}")
        return

    today = pd.Timestamp.today().normalize() # Today's date, we normalize it to remove the time, so the format is YYYY-MM-DD
    expiry_dates = [
        pd.to_datetime(e) for e in expirations
        if 0 <= (pd.to_datetime(e) - today).days <= EXPIRY_LOOKAHEAD_DAYS
    ] # We filter the expirations to only include the expirations in the next 7 days

    if not expiry_dates:
        print(f"No expirations found within the next {EXPIRY_LOOKAHEAD_DAYS} days.")
        return

    print(f"SPX Spot Price: {S0:.2f}")
    print(f"Risk-Free Rate: {RISK_FREE_RATE:.4f}")
    print(f"Scanning {len(expiry_dates)} expiries...\n")

    for expiry in sorted(expiry_dates):
        expiry_str = expiry.strftime('%Y-%m-%d')
        try:
            calls = ticker.option_chain(expiry_str).calls
        except Exception as e:
            print(f"Could not retrieve option chain for {expiry_str}: {e}")
            continue

        # Filter for liquid options with professional criteria
        valid_calls = calls[
            (calls['impliedVolatility'] > 0.001) &     # Minimum IV
            (calls['volume'] > 50) &                   # Minimum daily volume for liquidity
            (calls['openInterest'] > 100) &            # Minimum open interest for liquidity
            (calls['ask'] > 0) &                       # Must have valid ask price
            (calls['bid'] > 0) &                       # Must have valid bid price
            # Spread must be less than 10% of ask price for reasonable execution
            ((calls['ask'] - calls['bid']) / calls['ask'] < 0.10)
        ].copy()

        if valid_calls.empty:
            print(f"No liquid options found for {expiry_str} (after professional filters)")
            continue
            
        print(f"===================== Expiry: {expiry_str} =====================")
        print(f"Found {len(valid_calls)} liquid options (from {len(calls)} total options)")

        analyzed_count = 0
        buy_signals = 0
        
        for _, call in valid_calls.iterrows():
            result = analyze_option(call, S0, expiry, today)
            if result:
                print_option_analysis(result)
                analyzed_count += 1
                if "BUY" in result['decision']:
                    buy_signals += 1
        
        print(f"\nSummary for {expiry_str}: Analyzed {analyzed_count} options, {buy_signals} BUY signals")


if __name__ == "__main__":
    main()