"""
Portfolio Comparison Tool - Enhanced Statistics Edition (COMPLETE)
===================================================================

Confronta portafogli con statistiche avanzate e tabelle dettagliate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurazione
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
sns.set_palette(colors)


class EnhancedPortfolioComparison:
    """Analisi completa con statistiche avanzate"""
    
    def __init__(self):
        self.portfolios = {}
        self.sp500_data = None
        
    def load_portfolio(self, name, portfolio_dir, color=None):
        """Carica portafoglio"""
        portfolio_dir = Path(portfolio_dir)
        
        print(f"\n Caricamento: {name}")
        print(f"   Directory: {portfolio_dir}")
        
        if not portfolio_dir.exists():
            print(f"    Directory non trovata!")
            return False
        
        values_files = list(portfolio_dir.glob('portfolio_values*.csv'))
        if not values_files:
            print(f"   File non trovato!")
            return False
        
        values_file = sorted(values_files)[-1]
        print(f"    File: {values_file.name}")
        
        df = pd.read_csv(values_file)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            values = df['Value'] if 'Value' in df.columns else df.iloc[:, 0]
        else:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0]).sort_index()
            values = df.iloc[:, 0]
        
        returns = values.pct_change().dropna()
        
        self.portfolios[name] = {
            'values': values,
            'returns': returns,
            'color': color
        }
        
        print(f"   ✓ Caricato: {len(values)} giorni")
        return True
    
    def load_sp500(self):
        """Carica/simula S&P 500"""
        print(f"\n Benchmark S&P 500...")
        
        if len(self.portfolios) == 0:
            print(f"    Nessun portafoglio caricato!")
            return False
        
        first_portfolio = list(self.portfolios.values())[0]
        dates = first_portfolio['values'].index
        
        annual_return = 0.12
        annual_vol = 0.18
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)
        
        np.random.seed(42)
        returns = np.random.normal(daily_return, daily_vol, len(dates))
        
        initial_value = first_portfolio['values'].iloc[0]
        values = initial_value * (1 + returns).cumprod()
        
        self.sp500_values = pd.Series(values, index=dates)
        self.sp500_returns = pd.Series(returns, index=dates)
        
        print(f"   ✓ S&P 500: {len(self.sp500_values)} giorni")
        return True
    
    def table_performance_overview(self):
        """Tabella 1: Performance Overview"""
        print("\n" + "="*100)
        print(" TABELLA 1: PERFORMANCE OVERVIEW")
        print("="*100)
        
        data = []
        
        for name, portfolio in self.portfolios.items():
            values = portfolio['values']
            initial = values.iloc[0]
            final = values.iloc[-1]
            total_return = (final / initial - 1) * 100
            n_years = len(values) / 252
            cagr = ((final / initial) ** (1/n_years) - 1) * 100
            
            data.append({
                'Portfolio': name,
                'Initial Value': f'${initial:,.0f}',
                'Final Value': f'${final:,.0f}',
                'Total Return (%)': f'{total_return:.2f}%',
                'CAGR (%)': f'{cagr:.2f}%',
                'Days': len(values),
                'Years': f'{n_years:.2f}'
            })
        
        if self.sp500_values is not None:
            initial_sp = self.sp500_values.iloc[0]
            final_sp = self.sp500_values.iloc[-1]
            total_return_sp = (final_sp / initial_sp - 1) * 100
            n_years_sp = len(self.sp500_values) / 252
            cagr_sp = ((final_sp / initial_sp) ** (1/n_years_sp) - 1) * 100
            
            data.append({
                'Portfolio': 'S&P 500 (Benchmark)',
                'Initial Value': f'${initial_sp:,.0f}',
                'Final Value': f'${final_sp:,.0f}',
                'Total Return (%)': f'{total_return_sp:.2f}%',
                'CAGR (%)': f'{cagr_sp:.2f}%',
                'Days': len(self.sp500_values),
                'Years': f'{n_years_sp:.2f}'
            })
        
        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))
        return df
    
    def table_risk_metrics(self):
        """Tabella 2: Risk Metrics"""
        print("\n" + "="*100)
        print(" TABELLA 2: RISK METRICS")
        print("="*100)
        
        data = []
        
        for name, portfolio in self.portfolios.items():
            returns = portfolio['returns']
            
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252) * 100
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100
            avg_dd = drawdown[drawdown < 0].mean() * 100 if any(drawdown < 0) else 0
            
            var_95 = returns.quantile(0.05) * 100
            cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100
            
            downside_returns = returns[returns < 0]
            downside_dev = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
            
            data.append({
                'Portfolio': name,
                'Annual Vol (%)': f'{annual_vol:.2f}%',
                'Downside Dev (%)': f'{downside_dev:.2f}%',
                'Max Drawdown (%)': f'{max_dd:.2f}%',
                'Avg Drawdown (%)': f'{avg_dd:.2f}%',
                'VaR 95% (%)': f'{var_95:.2f}%',
                'CVaR 95% (%)': f'{cvar_95:.2f}%'
            })
        
        if self.sp500_returns is not None:
            returns_sp = self.sp500_returns
            annual_vol_sp = returns_sp.std() * np.sqrt(252) * 100
            
            cumulative_sp = (1 + returns_sp).cumprod()
            running_max_sp = cumulative_sp.cummax()
            drawdown_sp = (cumulative_sp - running_max_sp) / running_max_sp
            max_dd_sp = drawdown_sp.min() * 100
            avg_dd_sp = drawdown_sp[drawdown_sp < 0].mean() * 100 if any(drawdown_sp < 0) else 0
            
            var_95_sp = returns_sp.quantile(0.05) * 100
            cvar_95_sp = returns_sp[returns_sp <= returns_sp.quantile(0.05)].mean() * 100
            
            downside_returns_sp = returns_sp[returns_sp < 0]
            downside_dev_sp = downside_returns_sp.std() * np.sqrt(252) * 100 if len(downside_returns_sp) > 0 else 0
            
            data.append({
                'Portfolio': 'S&P 500',
                'Annual Vol (%)': f'{annual_vol_sp:.2f}%',
                'Downside Dev (%)': f'{downside_dev_sp:.2f}%',
                'Max Drawdown (%)': f'{max_dd_sp:.2f}%',
                'Avg Drawdown (%)': f'{avg_dd_sp:.2f}%',
                'VaR 95% (%)': f'{var_95_sp:.2f}%',
                'CVaR 95% (%)': f'{cvar_95_sp:.2f}%'
            })
        
        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))
        return df
    
    def table_risk_adjusted_returns(self):
        """Tabella 3: Risk-Adjusted Returns"""
        print("\n" + "="*100)
        print(" TABELLA 3: RISK-ADJUSTED RETURNS")
        print("="*100)
        
        data = []
        risk_free_rate = 0.02
        
        for name, portfolio in self.portfolios.items():
            values = portfolio['values']
            returns = portfolio['returns']
            
            n_years = len(values) / 252
            cagr = ((values.iloc[-1] / values.iloc[0]) ** (1/n_years) - 1)
            annual_vol = returns.std() * np.sqrt(252)
            
            excess_returns = returns - (risk_free_rate / 252)
            sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
            sortino = (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(drawdown.min())
            calmar = cagr / max_dd if max_dd > 0 else 0
            
            data.append({
                'Portfolio': name,
                'CAGR (%)': f'{cagr*100:.2f}%',
                'Volatility (%)': f'{annual_vol*100:.2f}%',
                'Sharpe Ratio': f'{sharpe:.3f}',
                'Sortino Ratio': f'{sortino:.3f}',
                'Calmar Ratio': f'{calmar:.3f}'
            })
        
        if self.sp500_values is not None:
            values_sp = self.sp500_values
            returns_sp = self.sp500_returns
            
            n_years_sp = len(values_sp) / 252
            cagr_sp = ((values_sp.iloc[-1] / values_sp.iloc[0]) ** (1/n_years_sp) - 1)
            annual_vol_sp = returns_sp.std() * np.sqrt(252)
            
            excess_returns_sp = returns_sp - (risk_free_rate / 252)
            sharpe_sp = (excess_returns_sp.mean() / returns_sp.std()) * np.sqrt(252) if returns_sp.std() > 0 else 0
            
            downside_returns_sp = returns_sp[returns_sp < 0]
            downside_std_sp = downside_returns_sp.std() * np.sqrt(252) if len(downside_returns_sp) > 0 else 0.0001
            sortino_sp = (cagr_sp - risk_free_rate) / downside_std_sp if downside_std_sp > 0 else 0
            
            cumulative_sp = (1 + returns_sp).cumprod()
            running_max_sp = cumulative_sp.cummax()
            drawdown_sp = (cumulative_sp - running_max_sp) / running_max_sp
            max_dd_sp = abs(drawdown_sp.min())
            calmar_sp = cagr_sp / max_dd_sp if max_dd_sp > 0 else 0
            
            data.append({
                'Portfolio': 'S&P 500',
                'CAGR (%)': f'{cagr_sp*100:.2f}%',
                'Volatility (%)': f'{annual_vol_sp*100:.2f}%',
                'Sharpe Ratio': f'{sharpe_sp:.3f}',
                'Sortino Ratio': f'{sortino_sp:.3f}',
                'Calmar Ratio': f'{calmar_sp:.3f}'
            })
        
        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))
        return df
    
    def table_monthly_statistics(self):
        """Tabella 4: Monthly Statistics"""
        print("\n" + "="*100)
        print(" TABELLA 4: MONTHLY STATISTICS")
        print("="*100)
        
        data = []
        
        for name, portfolio in self.portfolios.items():
            returns = portfolio['returns']
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            avg_monthly = monthly_returns.mean() * 100
            std_monthly = monthly_returns.std() * 100
            best_month = monthly_returns.max() * 100
            worst_month = monthly_returns.min() * 100
            positive_months = (monthly_returns > 0).sum()
            total_months = len(monthly_returns)
            win_rate = (positive_months / total_months) * 100
            
            data.append({
                'Portfolio': name,
                'Avg Monthly (%)': f'{avg_monthly:.2f}%',
                'Std Monthly (%)': f'{std_monthly:.2f}%',
                'Best Month (%)': f'{best_month:.2f}%',
                'Worst Month (%)': f'{worst_month:.2f}%',
                'Win Rate (%)': f'{win_rate:.1f}%',
                'Positive Months': f'{positive_months}/{total_months}'
            })
        
        if self.sp500_returns is not None:
            monthly_returns_sp = self.sp500_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            avg_monthly_sp = monthly_returns_sp.mean() * 100
            std_monthly_sp = monthly_returns_sp.std() * 100
            best_month_sp = monthly_returns_sp.max() * 100
            worst_month_sp = monthly_returns_sp.min() * 100
            positive_months_sp = (monthly_returns_sp > 0).sum()
            total_months_sp = len(monthly_returns_sp)
            win_rate_sp = (positive_months_sp / total_months_sp) * 100
            
            data.append({
                'Portfolio': 'S&P 500',
                'Avg Monthly (%)': f'{avg_monthly_sp:.2f}%',
                'Std Monthly (%)': f'{std_monthly_sp:.2f}%',
                'Best Month (%)': f'{best_month_sp:.2f}%',
                'Worst Month (%)': f'{worst_month_sp:.2f}%',
                'Win Rate (%)': f'{win_rate_sp:.1f}%',
                'Positive Months': f'{positive_months_sp}/{total_months_sp}'
            })
        
        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))
        return df
    
    def table_correlation_matrix(self):
        """Tabella 5: Correlation Matrix"""
        print("\n" + "="*100)
        print(" TABELLA 5: CORRELATION MATRIX")
        print("="*100)
        
        returns_dict = {}
        for name, portfolio in self.portfolios.items():
            returns_dict[name] = portfolio['returns']
        
        if self.sp500_returns is not None:
            returns_dict['S&P 500'] = self.sp500_returns
        
        df_returns = pd.DataFrame(returns_dict)
        corr_matrix = df_returns.corr()
        
        print("\n" + corr_matrix.to_string())
        return corr_matrix
    
    def generate_all_tables(self, output_dir='.'):
        """Genera tutte le tabelle"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*100)
        print("🚀 GENERAZIONE STATISTICHE COMPLETE")
        print("="*100)
        
        all_tables = {}
        
        all_tables['performance_overview'] = self.table_performance_overview()
        all_tables['risk_metrics'] = self.table_risk_metrics()
        all_tables['risk_adjusted_returns'] = self.table_risk_adjusted_returns()
        all_tables['monthly_statistics'] = self.table_monthly_statistics()
        all_tables['correlation_matrix'] = self.table_correlation_matrix()
        
        # Salva Excel
        excel_path = output_dir / 'portfolio_statistics_complete.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            all_tables['performance_overview'].to_excel(writer, sheet_name='Performance', index=False)
            all_tables['risk_metrics'].to_excel(writer, sheet_name='Risk Metrics', index=False)
            all_tables['risk_adjusted_returns'].to_excel(writer, sheet_name='Risk-Adjusted', index=False)
            all_tables['monthly_statistics'].to_excel(writer, sheet_name='Monthly Stats', index=False)
            all_tables['correlation_matrix'].to_excel(writer, sheet_name='Correlation')
        
        print(f"\n✅ Tutte le tabelle salvate in: {excel_path}")
        
        # Salva report TXT
        report_path = output_dir / 'STATISTICS_REPORT.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("PORTFOLIO STATISTICS REPORT\n")
            f.write("="*100 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for table_name, table_df in all_tables.items():
                f.write("\n" + "="*100 + "\n")
                f.write(f"{table_name.upper().replace('_', ' ')}\n")
                f.write("="*100 + "\n\n")
                f.write(table_df.to_string(index=True if table_name == 'correlation_matrix' else False))
                f.write("\n\n")
        
        print(f" Report testuale salvato in: {report_path}")
        print("\n" + "="*100)
        print(" ANALISI COMPLETATA!")
        print("="*100)
        
        return all_tables


def main():
    """Main execution"""
    print("\n" + "="*100)
    print("ENHANCED PORTFOLIO COMPARISON - YOUR 3 STRATEGIES")
    print("="*100)
    
    portfolios_config = [
        ('Classical V2.0', 'results_realistic_classical', colors[0]),
        ('Classical V2.0 Comparable', 'results_realistic_comparable', colors[1]),
        ('Quantum V2.5 Fixed', 'results_quantum_fixed', colors[2]),
    ]
    
    output_dir = "comparison_statistics_complete"
    
    print("\n Portafogli da caricare:")
    print("-" * 80)
    for idx, (name, directory, _) in enumerate(portfolios_config, 1):
        print(f"  {idx}. {name}")
        print(f"     Directory: ./{directory}/")
    print("-" * 80)
    
    comparator = EnhancedPortfolioComparison()
    
    loaded = 0
    for name, directory, color in portfolios_config:
        if comparator.load_portfolio(name, directory, color):
            loaded += 1
    
    if loaded == 0:
        print("\n ERRORE: Nessun portafoglio caricato!")
        return 1
    
    print("\n" + "="*100)
    print(f" Caricati {loaded}/{len(portfolios_config)} portafogli")
    print("="*100)
    
    comparator.load_sp500()
    
    print("\n" + "="*100)
    print(" GENERAZIONE STATISTICHE IN CORSO...")
    print("="*100)
    
    try:
        all_tables = comparator.generate_all_tables(output_dir)
        
        print("\n" + "="*100)
        print(" ANALISI COMPLETATA CON SUCCESSO!")
        print("="*100)
        print(f"\n File salvati in: ./{output_dir}/")
        print("-" * 80)
        print("   portfolio_statistics_complete.xlsx")
        print("   STATISTICS_REPORT.txt")
        print("-" * 80)
        
        # Vincitore
        risk_adj = all_tables.get('risk_adjusted_returns')
        if risk_adj is not None and len(risk_adj) > 0:
            print("\n" + "="*100)
            print("🏆 VINCITORE (per Sharpe Ratio):")
            print("="*100)
            
            sharpe_data = []
            for _, row in risk_adj.iterrows():
                if row['Portfolio'] != 'S&P 500':
                    try:
                        sharpe_val = float(row['Sharpe Ratio'])
                        sharpe_data.append((row['Portfolio'], sharpe_val))
                    except:
                        pass
            
            if sharpe_data:
                sharpe_data.sort(key=lambda x: x[1], reverse=True)
                print(f"\n  🥇 {sharpe_data[0][0]}")
                print(f"     Sharpe: {sharpe_data[0][1]:.3f}")
                
                if len(sharpe_data) > 1:
                    print(f"\n  🥈 {sharpe_data[1][0]}")
                    print(f"     Sharpe: {sharpe_data[1][1]:.3f}")
        
        print("\n" + "="*100)
        return 0
        
    except Exception as e:
        print(f"\n ERRORE: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrotto")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n ERRORE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)