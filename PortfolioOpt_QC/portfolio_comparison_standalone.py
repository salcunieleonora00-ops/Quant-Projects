"""
Portfolio Comparison Tool - Standalone Version
==============================================

Confronta 3 portafogli:
1. Quantum Generale (results)
2. Classical Optimized (results_realistic_comparable)  
3. Pure Quantum Fixed (results_quantum_fixed)

Usage:
------
python portfolio_comparison_standalone.py

Author: Portfolio Comparison System  
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurazione grafici
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#FAFAFA'
plt.rcParams['font.size'] = 10


class PortfolioComparisonFromFiles:
    """
    Legge i risultati dai file e crea confronto completo
    """
    
    def __init__(self):
        self.portfolios = {}
        self.sp500_data = None
        
    def load_portfolio_from_files(self, name, portfolio_dir, color=None):
        """Carica dati portafoglio da directory"""
        portfolio_dir = Path(portfolio_dir)
        
        print(f"\n Caricamento: {name}")
        print(f"   Directory: {portfolio_dir}")
        
        if not portfolio_dir.exists():
            print(f"    Directory non trovata!")
            return False
        
        # Cerca file portfolio_values
        values_files = list(portfolio_dir.glob('portfolio_values_*.csv'))
        if not values_files:
            print(f"    File portfolio_values_*.csv non trovato!")
            return False
        
        values_file = values_files[0]
        print(f"    File valori: {values_file.name}")
        
        # Leggi valori portafoglio
        df_values = pd.read_csv(values_file)
        
        # Assicurati che le colonne siano corrette
        if 'Date' in df_values.columns and 'Value' in df_values.columns:
            df_values['Date'] = pd.to_datetime(df_values['Date'])
            df_values = df_values.set_index('Date').sort_index()
            values = df_values['Value']
        elif 'date' in df_values.columns and 'portfolio_value' in df_values.columns:
            df_values['date'] = pd.to_datetime(df_values['date'])
            df_values = df_values.set_index('date').sort_index()
            values = df_values['portfolio_value']
        else:
            df_values.iloc[:, 0] = pd.to_datetime(df_values.iloc[:, 0])
            df_values = df_values.set_index(df_values.columns[0]).sort_index()
            values = df_values.iloc[:, 0]
        
        returns = values.pct_change().dropna()
        
        # Cerca metriche da JSON
        metrics = {}
        json_files = list(portfolio_dir.glob('complete_report_*.json'))
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
                if 'out_of_sample' in data:
                    oos = data['out_of_sample']
                    metrics = {
                        'annual_return': oos.get('annual_return', 0),
                        'sharpe_ratio': oos.get('sharpe_ratio', 0),
                        'max_drawdown': oos.get('max_drawdown', 0)
                    }
                    print(f"   ✓ Metriche da JSON")
        
        self.portfolios[name] = {
            'values': values,
            'returns': returns,
            'metrics': metrics,
            'color': color
        }
        
        print(f"   ✓ Caricato: {len(values)} giorni")
        print(f"   ✓ Return: {metrics.get('annual_return', 0):.2%}")
        print(f"   ✓ Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        
        return True
    
    def load_sp500_benchmark(self):
        """Scarica S&P 500 o lo simula"""
        print(f"\n Download S&P 500 Benchmark...")
        
        try:
            import yfinance as yf
            
            first_portfolio = list(self.portfolios.values())[0]
            start_date = first_portfolio['values'].index.min()
            end_date = first_portfolio['values'].index.max()
            
            print(f"   Periodo: {start_date} → {end_date}")
            
            sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            
            if len(sp500) == 0 or 'Adj Close' not in sp500.columns:
                raise Exception("Nessun dato")
            
            self.sp500_data = sp500['Adj Close']
            self.sp500_returns = self.sp500_data.pct_change().dropna()
            
            first_value = first_portfolio['values'].iloc[0]
            self.sp500_values = (self.sp500_data / self.sp500_data.iloc[0]) * first_value
            
            print(f"   ✓ S&P 500: {len(self.sp500_data)} giorni")
            return True
            
        except:
            print(f"     Uso simulazione S&P 500")
            return self._simulate_sp500()
    
    def _simulate_sp500(self):
        """Simula S&P 500"""
        first_portfolio = list(self.portfolios.values())[0]
        dates = first_portfolio['values'].index
        
        annual_return = 0.20
        annual_vol = 0.18
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)
        
        np.random.seed(42)
        returns = np.random.normal(daily_return, daily_vol, len(dates))
        
        initial_value = first_portfolio['values'].iloc[0]
        values = initial_value * (1 + returns).cumprod()
        
        self.sp500_values = pd.Series(values, index=dates)
        self.sp500_returns = pd.Series(returns, index=dates)
        
        print(f"   ✓ S&P 500 simulato: {len(self.sp500_values)} giorni")
        return True
    
    def calculate_metrics_table(self):
        """Crea tabella metriche"""
        print("\n Calcolo metriche comparative...")
        
        metrics_list = []
        
        for name, portfolio in self.portfolios.items():
            values = portfolio['values']
            returns = portfolio['returns']
            
            n_days = len(values)
            n_years = n_days / 252
            total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
            annual_return = ((values.iloc[-1] / values.iloc[0]) ** (1/n_years) - 1) * 100
            annual_vol = returns.std() * np.sqrt(252) * 100
            
            excess_returns = returns - (0.02 / 252)
            sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino = ((annual_return/100) - 0.02) / downside_std if downside_std > 0 else 0
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100
            
            calmar = (annual_return/100) / abs(max_dd/100) if max_dd != 0 else 0
            win_rate = (returns > 0).sum() / len(returns) * 100
            
            metrics_list.append({
                'Portafoglio': name,
                'Valore Finale ($)': f'{values.iloc[-1]:,.0f}',
                'Rendimento Totale (%)': round(total_return, 2),
                'Rendimento Annualizzato (%)': round(annual_return, 2),
                'Volatilità Annua (%)': round(annual_vol, 2),
                'Sharpe Ratio': round(sharpe, 2),
                'Sortino Ratio': round(sortino, 2),
                'Max Drawdown (%)': round(max_dd, 2),
                'Calmar Ratio': round(calmar, 2),
                'Win Rate (%)': round(win_rate, 2)
            })
        
        # S&P 500
        if self.sp500_values is not None:
            values = self.sp500_values
            returns = self.sp500_returns
            
            n_days = len(values)
            n_years = n_days / 252
            total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
            annual_return = ((values.iloc[-1] / values.iloc[0]) ** (1/n_years) - 1) * 100
            annual_vol = returns.std() * np.sqrt(252) * 100
            
            excess_returns = returns - (0.02 / 252)
            sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino = ((annual_return/100) - 0.02) / downside_std if downside_std > 0 else 0
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100
            
            calmar = (annual_return/100) / abs(max_dd/100) if max_dd != 0 else 0
            win_rate = (returns > 0).sum() / len(returns) * 100
            
            metrics_list.append({
                'Portafoglio': 'S&P 500 (Benchmark)',
                'Valore Finale ($)': f'{values.iloc[-1]:,.0f}',
                'Rendimento Totale (%)': round(total_return, 2),
                'Rendimento Annualizzato (%)': round(annual_return, 2),
                'Volatilità Annua (%)': round(annual_vol, 2),
                'Sharpe Ratio': round(sharpe, 2),
                'Sortino Ratio': round(sortino, 2),
                'Max Drawdown (%)': round(max_dd, 2),
                'Calmar Ratio': round(calmar, 2),
                'Win Rate (%)': round(win_rate, 2)
            })
        
        df = pd.DataFrame(metrics_list)
        return df
    
    def create_comparison_dashboard(self, output_path='comparison_dashboard.png'):
        """Crea dashboard con 6 grafici"""
        print("\n Generazione dashboard...")
        
        fig = plt.figure(figsize=(18, 11))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 1. PERFORMANCE CUMULATIVA
        ax1 = fig.add_subplot(gs[0, :])
        
        for i, (name, portfolio) in enumerate(self.portfolios.items()):
            values = portfolio['values']
            color = portfolio.get('color', colors[i])
            ax1.plot(values.index, values.values,
                    linewidth=2.5, label=name, color=color, alpha=0.9)
        
        if self.sp500_values is not None:
            ax1.plot(self.sp500_values.index, self.sp500_values.values,
                    linewidth=2.5, label='S&P 500', color=colors[3],
                    linestyle='--', alpha=0.8)
        
        if self.portfolios:
            initial_value = list(self.portfolios.values())[0]['values'].iloc[0]
            ax1.axhline(y=initial_value, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax1.set_xlabel('Data', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Valore Portafoglio ($)', fontweight='bold', fontsize=11)
        ax1.set_title('Performance Cumulativa dei Portafogli', 
                     fontweight='bold', fontsize=14, pad=15)
        ax1.legend(loc='best', fontsize=10, framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. DRAWDOWN
        ax2 = fig.add_subplot(gs[1, 0])
        
        for i, (name, portfolio) in enumerate(self.portfolios.items()):
            returns = portfolio['returns']
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = ((cumulative - running_max) / running_max) * 100
            color = portfolio.get('color', colors[i])
            ax2.plot(drawdown.index, drawdown.values,
                    linewidth=2, label=name, color=color, alpha=0.85)
        
        if self.sp500_values is not None:
            cumulative_sp = (1 + self.sp500_returns).cumprod()
            running_max_sp = cumulative_sp.cummax()
            drawdown_sp = ((cumulative_sp - running_max_sp) / running_max_sp) * 100
            ax2.plot(drawdown_sp.index, drawdown_sp.values,
                    linewidth=2, label='S&P 500', color=colors[3],
                    linestyle='--', alpha=0.8)
        
        ax2.fill_between(ax2.get_xlim(), 0, -100, alpha=0.05, color='red')
        ax2.set_xlabel('Data', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Drawdown (%)', fontweight='bold', fontsize=10)
        ax2.set_title('Drawdown nel Tempo', fontweight='bold', fontsize=12, pad=10)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # 3. RISK-RETURN SCATTER
        ax3 = fig.add_subplot(gs[1, 1])
        
        for i, (name, portfolio) in enumerate(self.portfolios.items()):
            returns = portfolio['returns']
            values = portfolio['values']
            
            n_years = len(values) / 252
            ann_return = ((values.iloc[-1] / values.iloc[0]) ** (1/n_years) - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100
            color = portfolio.get('color', colors[i])
            
            ax3.scatter(volatility, ann_return,
                       s=300, alpha=0.8, color=color,
                       edgecolors='black', linewidth=2, marker='o', zorder=3)
            
            ax3.annotate(name,
                        xy=(volatility, ann_return),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='white',
                                edgecolor=color,
                                alpha=0.9))
        
        if self.sp500_values is not None:
            n_years = len(self.sp500_values) / 252
            ann_return_sp = ((self.sp500_values.iloc[-1] / self.sp500_values.iloc[0]) ** (1/n_years) - 1) * 100
            volatility_sp = self.sp500_returns.std() * np.sqrt(252) * 100
            
            ax3.scatter(volatility_sp, ann_return_sp,
                       s=300, alpha=0.8, color=colors[3],
                       edgecolors='black', linewidth=2, marker='D', zorder=3)
            
            ax3.annotate('S&P 500',
                        xy=(volatility_sp, ann_return_sp),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='white',
                                edgecolor=colors[3],
                                alpha=0.9))
        
        ax3.set_xlabel('Volatilità Annua (%)', fontweight='bold', fontsize=10)
        ax3.set_ylabel('Rendimento Annualizzato (%)', fontweight='bold', fontsize=10)
        ax3.set_title('Profilo Rischio-Rendimento', fontweight='bold', fontsize=12, pad=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. BAR CHART - SHARPE RATIO
        ax4 = fig.add_subplot(gs[1, 2])
        
        names = []
        sharpes = []
        bar_colors = []
        
        for i, (name, portfolio) in enumerate(self.portfolios.items()):
            returns = portfolio['returns']
            excess_returns = returns - (0.02 / 252)
            sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            names.append(name)
            sharpes.append(sharpe)
            bar_colors.append(portfolio.get('color', colors[i]))
        
        if self.sp500_values is not None:
            excess_returns_sp = self.sp500_returns - (0.02 / 252)
            sharpe_sp = (excess_returns_sp.mean() / self.sp500_returns.std()) * np.sqrt(252) if self.sp500_returns.std() > 0 else 0
            names.append('S&P 500')
            sharpes.append(sharpe_sp)
            bar_colors.append(colors[3])
        
        bars = ax4.barh(names, sharpes, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.axvline(x=1, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Sharpe = 1')
        ax4.set_xlabel('Sharpe Ratio', fontweight='bold', fontsize=10)
        ax4.set_title('Confronto Sharpe Ratio', fontweight='bold', fontsize=12, pad=10)
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.legend(fontsize=8)
        
        for i, (bar, sharpe) in enumerate(zip(bars, sharpes)):
            ax4.text(sharpe + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{sharpe:.2f}',
                    va='center', fontweight='bold', fontsize=9)
        
        # 5. DISTRIBUZIONE RENDIMENTI
        ax5 = fig.add_subplot(gs[2, 0])
        
        for i, (name, portfolio) in enumerate(self.portfolios.items()):
            returns = portfolio['returns'] * 100
            color = portfolio.get('color', colors[i])
            ax5.hist(returns, bins=40, alpha=0.6, label=name,
                    color=color, edgecolor='black', linewidth=0.5)
        
        if self.sp500_values is not None:
            returns_sp = self.sp500_returns * 100
            ax5.hist(returns_sp, bins=40, alpha=0.4, label='S&P 500',
                    color=colors[3], edgecolor='black', linewidth=0.5)
        
        ax5.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax5.set_xlabel('Rendimento Giornaliero (%)', fontweight='bold', fontsize=10)
        ax5.set_ylabel('Frequenza', fontweight='bold', fontsize=10)
        ax5.set_title('Distribuzione Rendimenti', fontweight='bold', fontsize=12, pad=10)
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. ROLLING SHARPE
        ax6 = fig.add_subplot(gs[2, 1:])
        
        window = 63
        
        for i, (name, portfolio) in enumerate(self.portfolios.items()):
            returns = portfolio['returns']
            rolling_sharpe = (returns.rolling(window).mean() / 
                            returns.rolling(window).std()) * np.sqrt(252)
            color = portfolio.get('color', colors[i])
            ax6.plot(rolling_sharpe.index, rolling_sharpe.values,
                    linewidth=2, label=name, color=color, alpha=0.85)
        
        if self.sp500_values is not None:
            rolling_sharpe_sp = (self.sp500_returns.rolling(window).mean() / 
                               self.sp500_returns.rolling(window).std()) * np.sqrt(252)
            ax6.plot(rolling_sharpe_sp.index, rolling_sharpe_sp.values,
                    linewidth=2, label='S&P 500', color=colors[3],
                    linestyle='--', alpha=0.8)
        
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax6.axhline(y=1, color='green', linestyle=':', linewidth=1, alpha=0.5)
        ax6.set_xlabel('Data', fontweight='bold', fontsize=10)
        ax6.set_ylabel(f'Sharpe Ratio ({window}d)', fontweight='bold', fontsize=10)
        ax6.set_title('Sharpe Ratio Mobile', fontweight='bold', fontsize=12, pad=10)
        ax6.legend(loc='best', fontsize=9, ncol=2)
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Confronto Portafogli: Quantum Generale vs Classical vs Pure Quantum Fixed vs S&P 500',
                    fontsize=16, fontweight='bold', y=0.998)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Dashboard salvata: {output_path}")
        plt.close(fig)
        
        return output_path
    
    def generate_full_report(self, output_dir='.'):
        """Genera report completo"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*90)
        print("GENERAZIONE REPORT COMPLETO")
        print("="*90)
        
        metrics_df = self.calculate_metrics_table()
        
        print("\n" + metrics_df.to_string(index=False))
        
        csv_path = output_dir / 'portfolio_comparison_metrics.csv'
        metrics_df.to_csv(csv_path, index=False)
        print(f"\n✓ Metriche salvate: {csv_path}")
        
        dashboard_path = output_dir / 'portfolio_comparison_dashboard.png'
        self.create_comparison_dashboard(dashboard_path)
        
        report_lines = []
        report_lines.append("="*90)
        report_lines.append("PORTFOLIO COMPARISON REPORT")
        report_lines.append("="*90)
        report_lines.append(f"\nData: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report_lines.append("\n" + "-"*90)
        report_lines.append("PORTAFOGLI ANALIZZATI:")
        report_lines.append("-"*90)
        
        for i, (name, portfolio) in enumerate(self.portfolios.items(), 1):
            values = portfolio['values']
            report_lines.append(f"\n{i}. {name}")
            report_lines.append(f"   Valore Iniziale: ${values.iloc[0]:,.0f}")
            report_lines.append(f"   Valore Finale: ${values.iloc[-1]:,.0f}")
            report_lines.append(f"   Giorni: {len(values)}")
        
        report_lines.append("\n" + "-"*90)
        report_lines.append("METRICHE COMPARATIVE:")
        report_lines.append("-"*90)
        report_lines.append("\n" + metrics_df.to_string(index=False))
        
        report_lines.append("\n" + "="*90)
        report_lines.append("ANALISI:")
        report_lines.append("="*90)
        
        best_return_idx = metrics_df['Rendimento Annualizzato (%)'].astype(float).idxmax()
        best_sharpe_idx = metrics_df['Sharpe Ratio'].astype(float).idxmax()
        low_vol_idx = metrics_df['Volatilità Annua (%)'].astype(float).idxmin()
        best_dd_idx = metrics_df['Max Drawdown (%)'].astype(float).idxmax()
        
        report_lines.append(f"\n🏆 Miglior Rendimento: {metrics_df.loc[best_return_idx, 'Portafoglio']}")
        report_lines.append(f" Miglior Sharpe: {metrics_df.loc[best_sharpe_idx, 'Portafoglio']}")
        report_lines.append(f"  Minor Volatilità: {metrics_df.loc[low_vol_idx, 'Portafoglio']}")
        report_lines.append(f" Miglior Drawdown: {metrics_df.loc[best_dd_idx, 'Portafoglio']}")
        
        report_lines.append("\n" + "="*90)
        
        report_text = '\n'.join(report_lines)
        
        txt_path = output_dir / 'portfolio_comparison_report.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ Report salvato: {txt_path}")
        
        print("\n" + "="*90)
        print(" REPORT COMPLETATO!")
        print("="*90)
        print(f"\n📁 File salvati in: {output_dir}/")
        print(f"   • {csv_path.name}")
        print(f"   • {dashboard_path.name}")
        print(f"   • {txt_path.name}")
        
        return metrics_df


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Portfolio Comparison Tool')
    parser.add_argument('--quantum_general_dir', type=str, default='./results',
                       help='Directory portafoglio Quantum Generale')
    parser.add_argument('--classical_dir', type=str, default='./results_realistic_comparable',
                       help='Directory portafoglio Classical')
    parser.add_argument('--quantum_fixed_dir', type=str, default='./results_quantum_fixed',
                       help='Directory portafoglio Pure Quantum Fixed')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help='Directory output')
    parser.add_argument('--no_sp500', action='store_true',
                       help='Non includere benchmark S&P 500')
    
    args = parser.parse_args()
    
    print("\n" + "="*90)
    print("PORTFOLIO COMPARISON TOOL")
    print("="*90)
    print(f"\nLegge i risultati reali dai tuoi file CSV e genera confronto completo")
    
    comparator = PortfolioComparisonFromFiles()
    
    loaded_count = 0
    
    # 1. Quantum Generale
    if Path(args.quantum_general_dir).exists():
        if comparator.load_portfolio_from_files(
            'Quantum Generale', 
            args.quantum_general_dir, 
            color=colors[0]
        ):
            loaded_count += 1
    else:
        print(f"\n  Directory non trovata: {args.quantum_general_dir}")
    
    # 2. Classical
    if Path(args.classical_dir).exists():
        if comparator.load_portfolio_from_files(
            'Classical Optimized',
            args.classical_dir,
            color=colors[1]
        ):
            loaded_count += 1
    else:
        print(f"\n  Directory non trovata: {args.classical_dir}")
    
    # 3. Pure Quantum Fixed
    if Path(args.quantum_fixed_dir).exists():
        if comparator.load_portfolio_from_files(
            'Pure Quantum Fixed',
            args.quantum_fixed_dir,
            color=colors[2]
        ):
            loaded_count += 1
    else:
        print(f"\n  Directory non trovata: {args.quantum_fixed_dir}")
    
    if loaded_count == 0:
        print("\n ERRORE: Nessun portafoglio caricato!")
        return 1
    
    print(f"\n✓ Caricati {loaded_count} portafogli")
    
    if not args.no_sp500:
        comparator.load_sp500_benchmark()
    
    metrics_df = comparator.generate_full_report(args.output_dir)
    
    print("\n Analisi completata con successo!")
    
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n  Interrotto dall'utente")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n ERRORE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)