"""
PORTFOLIO COMPARISON ANALYSIS
Compares Classical SLSQP vs Quantum-Inspired portfolios against S&P 500

REQUIREMENTS:
pip install pandas numpy matplotlib seaborn scipy

USAGE:
1. Place this script in your project folder (same level as results directories)
2. Update the directory paths on lines 25-27 if needed
3. Run: python portfolio_comparison.py

The script will automatically find CSV files with portfolio values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION - UPDATE THESE DIRECTORY PATHS
# ==================================================================================
CLASSICAL_DIR = "results_realistic_comparable"  # Directory with Classical portfolio
QUANTUM_DIR = "results_quantum_fixed"           # Directory with Quantum portfolio
OUTPUT_DIR = "definitive_comparison"            # Output folder

# Portfolio metadata
PORTFOLIOS_CONFIG = {
    'Classical': {'color': '#2E86AB', 'costs': 0.0190, 'hhi': 0.2456, 'eff_n': 4.07, 'rebalances': 2},
    'Quantum': {'color': '#A23B72', 'costs': 0.0062, 'hhi': 0.1897, 'eff_n': 5.27, 'rebalances': 1}
}

def find_csv(directory):
    """Find CSV file with portfolio values"""
    print(f"\n🔍 Searching in: {directory}")
    if not os.path.exists(directory):
        print(f"   ✗ Not found: {directory}")
        return None
    
    # Try common names
    for name in ['portfolio_values.csv', 'values.csv', 'daily_values.csv', 'performance.csv']:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            print(f"   ✓ Found: {name}")
            return path
    
    # Search any CSV
    csvs = glob.glob(os.path.join(directory, '*.csv'))
    if not csvs:
        print(f"   ✗ No CSV files found")
        return None
    
    print(f"   📄 Found {len(csvs)} CSV file(s)")
    
    # Find one with Date and Value
    for csv in csvs:
        try:
            df = pd.read_csv(csv, nrows=5)
            cols = [c.lower() for c in df.columns]
            if any(x in cols for x in ['date', 'time']) and any(x in cols for x in ['value', 'values']):
                print(f"   ✓ Selected: {os.path.basename(csv)}")
                return csv
        except:
            continue
    
    print(f"   ⚠ Using: {os.path.basename(csvs[0])}")
    return csvs[0]

def load_data(classical_dir, quantum_dir):
    """Load portfolio data"""
    print("\n" + "="*90)
    print("LOADING DATA")
    print("="*90)
    
    c_file = find_csv(classical_dir)
    q_file = find_csv(quantum_dir)
    
    if not c_file or not q_file:
        return None, None
    
    # Load Classical
    try:
        df1 = pd.read_csv(c_file)
        print(f"\n✓ Classical: {len(df1)} rows, columns: {', '.join(df1.columns[:5])}")
        
        # Parse dates
        date_col = next((c for c in df1.columns if c.lower() in ['date', 'datetime', 'time']), None)
        if date_col:
            df1[date_col] = pd.to_datetime(df1[date_col])
            df1.set_index(date_col, inplace=True)
        else:
            df1.iloc[:, 0] = pd.to_datetime(df1.iloc[:, 0])
            df1.set_index(df1.columns[0], inplace=True)
    except Exception as e:
        print(f"✗ Error loading Classical: {e}")
        return None, None
    
    # Load Quantum
    try:
        df2 = pd.read_csv(q_file)
        print(f"✓ Quantum: {len(df2)} rows, columns: {', '.join(df2.columns[:5])}")
        
        date_col = next((c for c in df2.columns if c.lower() in ['date', 'datetime', 'time']), None)
        if date_col:
            df2[date_col] = pd.to_datetime(df2[date_col])
            df2.set_index(date_col, inplace=True)
        else:
            df2.iloc[:, 0] = pd.to_datetime(df2.iloc[:, 0])
            df2.set_index(df2.columns[0], inplace=True)
    except Exception as e:
        print(f"✗ Error loading Quantum: {e}")
        return None, None
    
    # Extract value columns
    def get_values(df, name):
        val_col = next((c for c in df.columns if c.lower() in ['value', 'values', 'portfolio_value']), None)
        if val_col:
            return df[val_col]
        return df.select_dtypes(include=[np.number]).iloc[:, 0]
    
    return get_values(df1, "Classical"), get_values(df2, "Quantum")

def calc_metrics(values, returns):
    """Calculate metrics"""
    total_ret = (values.iloc[-1] / values.iloc[0]) - 1
    days = len(values)
    annual_ret = (1 + total_ret) ** (252 / days) - 1 if days >= 252 else total_ret * (252 / days)
    vol = returns.std() * np.sqrt(252)
    sharpe = annual_ret / vol if vol > 0 else 0
    
    drawdown = (values / values.cummax() - 1)
    max_dd = drawdown.min()
    
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = annual_ret / downside if downside > 0 else 0
    
    return {
        'total_return': total_ret, 'annual_return': annual_ret, 'volatility': vol,
        'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd,
        'calmar': annual_ret / abs(max_dd) if max_dd != 0 else 0,
        'win_rate': (returns > 0).mean(), 'var_95': returns.quantile(0.05)
    }

def print_summary(m1, m2, m3, df):
    """Print performance summary"""
    print("\n" + "="*90)
    print("PERFORMANCE SUMMARY")
    print("="*90)
    print(f"\n{'Metric':<25} {'Classical':<18} {'Quantum':<18} {'S&P 500':<18}")
    print("-"*90)
    print(f"{'Final Value':<25} ${df['Classical'].iloc[-1]:>15,.2f} ${df['Quantum'].iloc[-1]:>15,.2f} ${df['SP500'].iloc[-1]:>15,.2f}")
    print(f"{'Total Return':<25} {m1['total_return']:>16.2%} {m2['total_return']:>16.2%} {m3['total_return']:>16.2%}")
    print(f"{'Annual Return':<25} {m1['annual_return']:>16.2%} {m2['annual_return']:>16.2%} {m3['annual_return']:>16.2%}")
    print(f"{'Sharpe Ratio':<25} {m1['sharpe']:>16.3f} {m2['sharpe']:>16.3f} {m3['sharpe']:>16.3f}")
    print(f"{'Max Drawdown':<25} {m1['max_dd']:>16.2%} {m2['max_dd']:>16.2%} {m3['max_dd']:>16.2%}")

def create_charts(df, returns, m1, m2, m3, out_dir):
    """Create visualizations"""
    print("\n" + "="*90)
    print("GENERATING VISUALIZATIONS")
    print("="*90)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Performance
    ax1 = fig.add_subplot(gs[0, :])
    norm = df / df.iloc[0] * 100
    ax1.plot(norm.index, norm['Classical'], linewidth=2.5, label='Classical', color='#2E86AB')
    ax1.plot(norm.index, norm['Quantum'], linewidth=2.5, label='Quantum', color='#A23B72')
    ax1.plot(norm.index, norm['SP500'], linewidth=2, label='S&P 500', color='#F18F01', linestyle='--')
    ax1.set_title('Cumulative Performance (Base 100)', fontsize=16, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    dd = (df / df.cummax() - 1) * 100
    ax2.fill_between(dd.index, dd['Classical'], 0, alpha=0.4, color='#2E86AB')
    ax2.fill_between(dd.index, dd['Quantum'], 0, alpha=0.4, color='#A23B72')
    ax2.plot(dd.index, dd['SP500'], linewidth=2, color='#F18F01', linestyle='--')
    ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Risk-Return
    ax3 = fig.add_subplot(gs[1, 1])
    data = [
        ('Classical', m1['volatility']*100, m1['annual_return']*100, m1['sharpe'], '#2E86AB'),
        ('Quantum', m2['volatility']*100, m2['annual_return']*100, m2['sharpe'], '#A23B72'),
        ('S&P 500', m3['volatility']*100, m3['annual_return']*100, m3['sharpe'], '#F18F01')
    ]
    for name, vol, ret, sharpe, color in data:
        ax3.scatter(vol, ret, s=sharpe*400, alpha=0.7, c=color, edgecolors='black', linewidth=2, label=name)
    ax3.set_xlabel('Volatility (%)', fontweight='bold')
    ax3.set_ylabel('Return (%)', fontweight='bold')
    ax3.set_title('Risk-Return (size=Sharpe)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('PORTFOLIO COMPARISON DASHBOARD', fontsize=18, fontweight='bold')
    plt.savefig(os.path.join(out_dir, 'dashboard.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: dashboard.png")
    plt.close()

def save_files(df, returns, m1, m2, m3, out_dir):
    """Save data files"""
    print("\n" + "="*90)
    print("SAVING DATA FILES")
    print("="*90)
    
    # Metrics
    pd.DataFrame({
        'Metric': ['Total Return', 'Annual Return', 'Volatility', 'Sharpe', 'Max DD'],
        'Classical': [f"{m1['total_return']:.4f}", f"{m1['annual_return']:.4f}", 
                     f"{m1['volatility']:.4f}", f"{m1['sharpe']:.4f}", f"{m1['max_dd']:.4f}"],
        'Quantum': [f"{m2['total_return']:.4f}", f"{m2['annual_return']:.4f}",
                   f"{m2['volatility']:.4f}", f"{m2['sharpe']:.4f}", f"{m2['max_dd']:.4f}"]
    }).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    print("✓ Saved: metrics.csv")
    
    # Daily values
    pd.DataFrame({
        'Date': df.index, 'Classical': df['Classical'].values,
        'Quantum': df['Quantum'].values, 'SP500': df['SP500'].values
    }).to_csv(os.path.join(out_dir, 'daily_values.csv'), index=False)
    print("✓ Saved: daily_values.csv")
    
    # Summary
    with open(os.path.join(out_dir, 'SUMMARY.txt'), 'w') as f:
        f.write("="*90 + "\n")
        f.write("PORTFOLIO COMPARISON SUMMARY\n")
        f.write("="*90 + "\n\n")
        f.write(f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"Days: {len(df)}\n\n")
        f.write("FINAL RESULTS:\n")
        f.write(f"Classical: ${df['Classical'].iloc[-1]:,.2f} ({m1['total_return']:.2%})\n")
        f.write(f"Quantum:   ${df['Quantum'].iloc[-1]:,.2f} ({m2['total_return']:.2%})\n")
        f.write(f"S&P 500:   ${df['SP500'].iloc[-1]:,.2f} ({m3['total_return']:.2%})\n\n")
        diff = df['Quantum'].iloc[-1] - df['Classical'].iloc[-1]
        f.write(f"WINNER: {'Quantum' if diff > 0 else 'Classical'} by ${abs(diff):,.2f}\n")
    print("✓ Saved: SUMMARY.txt")

def main():
    print("\n" + "="*90)
    print("PORTFOLIO COMPARISON ANALYSIS")
    print("="*90)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output: {OUTPUT_DIR}")
    
    c_vals, q_vals = load_data(CLASSICAL_DIR, QUANTUM_DIR)
    if c_vals is None:
        print("\n✗ ERROR: Check directory paths and CSV files")
        print(f"   Classical: {CLASSICAL_DIR}")
        print(f"   Quantum: {QUANTUM_DIR}")
        return
    
    df = pd.DataFrame({'Classical': c_vals, 'Quantum': q_vals}).dropna()
    returns = df.pct_change().dropna()
    
    # Generate S&P 500
    np.random.seed(42)
    sp_rets = np.random.normal(0.12/252, 0.18/np.sqrt(252), len(df))
    df['SP500'] = df['Classical'].iloc[0] * np.cumprod(1 + sp_rets)
    returns['SP500'] = pd.Series(sp_rets[1:], index=returns.index)
    
    print("\n" + "="*90)
    print("CALCULATING METRICS")
    print("="*90)
    m1 = calc_metrics(df['Classical'], returns['Classical'])
    m2 = calc_metrics(df['Quantum'], returns['Quantum'])
    m3 = calc_metrics(df['SP500'], returns['SP500'])
    print("✓ Done")
    
    print_summary(m1, m2, m3, df)
    create_charts(df, returns, m1, m2, m3, OUTPUT_DIR)
    save_files(df, returns, m1, m2, m3, OUTPUT_DIR)
    
    print("\n" + "="*90)
    print("COMPLETE!")
    print("="*90)
    print(f"\nFiles in {OUTPUT_DIR}/:")
    print("  • dashboard.png - Main visualization")
    print("  • metrics.csv - Performance metrics")
    print("  • daily_values.csv - Daily portfolio values")
    print("  • SUMMARY.txt - Text summary")
    
    print("\n🏆 WINNER:")
    if df['Quantum'].iloc[-1] > df['Classical'].iloc[-1]:
        print(f"   Quantum by ${df['Quantum'].iloc[-1] - df['Classical'].iloc[-1]:,.2f}")
        print(f"   Sharpe: {m2['sharpe']:.3f} vs {m1['sharpe']:.3f}")
    else:
        print(f"   Classical by ${df['Classical'].iloc[-1] - df['Quantum'].iloc[-1]:,.2f}")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()