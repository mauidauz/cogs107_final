import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

PERCENTILES = [10, 30, 50, 70, 90]
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}
OUTPUT_DIR = Path(__file__).parent / 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_data(file_path, prepare_for='sdt', display=False):
    df = pd.read_csv(file_path)

    for col, mapping in MAPPINGS.items():
        if df[col].dtype == 'object':
            df[col] = df[col].map(mapping)

    df['condition'] = df['stimulus_type'] + df['difficulty'] * 2
    df['pnum'] = df['participant_id']

    if prepare_for == 'sdt':
        grouped = df.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                signal = c_data[c_data['signal'] == 0]
                noise = c_data[c_data['signal'] == 1]
                if not signal.empty and not noise.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal['correct'].iloc[0],
                        'misses': signal['nTrials'].iloc[0] - signal['correct'].iloc[0],
                        'false_alarms': noise['nTrials'].iloc[0] - noise['correct'].iloc[0],
                        'correct_rejections': noise['correct'].iloc[0],
                        'nSignal': signal['nTrials'].iloc[0],
                        'nNoise': noise['nTrials'].iloc[0]
                    })
        return pd.DataFrame(sdt_data)
    
    if prepare_for == 'delta plots':
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', *[f'p{p}' for p in PERCENTILES]])
        for pnum in df['pnum'].unique():
            for condition in df['condition'].unique():
                cond_data = df[(df['pnum'] == pnum) & (df['condition'] == condition)]
                if cond_data.empty:
                    continue
                for mode, filt in [('overall', True), ('accurate', cond_data['accuracy'] == 1), ('error', cond_data['accuracy'] == 0)]:
                    subset = cond_data if filt is True else cond_data[filt]
                    if subset.empty:
                        continue
                    percentiles = {f'p{p}': np.percentile(subset['rt'], p) for p in PERCENTILES}
                    dp_data = pd.concat([dp_data, pd.DataFrame([{
                        'pnum': pnum,
                        'condition': condition,
                        'mode': mode,
                        **percentiles
                    }])])
    return dp_data

def apply_hierarchical_sdt_model(data):
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())

    with pm.Model() as model:
        mean_d = pm.Normal("mean_d", mu=0, sigma=1, shape=C)
        sd_d = pm.HalfNormal("sd_d", sigma=1)

        mean_c = pm.Normal("mean_c", mu=0, sigma=1, shape=C)
        sd_c = pm.HalfNormal("sd_c", sigma=1)

        d = pm.Normal("d", mu=mean_d, sigma=sd_d, shape=(P, C))
        c = pm.Normal("c", mu=mean_c, sigma=sd_c, shape=(P, C))

        hit_p = pm.math.invlogit(d - c)
        fa_p = pm.math.invlogit(-c)

        obs_hit = pm.Binomial("hit_obs", n=data['nSignal'], 
                              p=hit_p[data['pnum'], data['condition']], observed=data['hits'])
        obs_fa = pm.Binomial("fa_obs", n=data['nNoise'], 
                             p=fa_p[data['pnum'], data['condition']], observed=data['false_alarms'])

        trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)
    return model, trace

def plot_delta_plots(raw_df):
    """Draw delta plots comparing RT distributions between condition pairs."""
    dp_data = read_data(raw_df, prepare_for='delta plots') if isinstance(raw_df, str) else read_data(raw_df, prepare_for='delta plots')

    for pnum in dp_data['pnum'].unique():
        data = dp_data[dp_data['pnum'] == pnum]
        conditions = sorted(data['condition'].unique())
        n = len(conditions)

        fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
        marker_style = {
            'marker': 'o',
            'markersize': 6,
            'markerfacecolor': 'white',
            'markeredgewidth': 1.2,
            'linewidth': 1.5
        }

        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions):
                ax = axes[i, j]

                if i == j:
                    ax.axis('off')
                    continue

                p_vals = [f'p{p}' for p in PERCENTILES]

                def get_percentiles(cond, mode):
                    row = data[(data['condition'] == cond) & (data['mode'] == mode)]
                    if row.empty:
                        return np.full(len(PERCENTILES), np.nan)
                    return row[p_vals].iloc[0].values

                if i < j:
                    rt1 = get_percentiles(cond1, 'overall')
                    rt2 = get_percentiles(cond2, 'overall')
                    delta = rt2 - rt1
                    ax.plot(PERCENTILES, delta, color='black', label='Overall', **marker_style)
                else:
                    rt1_err = get_percentiles(cond1, 'error')
                    rt2_err = get_percentiles(cond2, 'error')
                    rt1_acc = get_percentiles(cond1, 'accurate')
                    rt2_acc = get_percentiles(cond2, 'accurate')
                    ax.plot(PERCENTILES, rt2_err - rt1_err, color='red', label='Error', **marker_style)
                    ax.plot(PERCENTILES, rt2_acc - rt1_acc, color='green', label='Accurate', **marker_style)
                    ax.legend(fontsize=8)

                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_ylim(-0.3, 0.5)
                if j == 0:
                    ax.set_ylabel('Δ RT (s)')
                if i == n - 1:
                    ax.set_xlabel('Percentile')

                # Add condition label
                label = f"{CONDITION_NAMES.get(cond2, cond2)} - {CONDITION_NAMES.get(cond1, cond1)}"
                ax.text(50, -0.25, label, ha='center', va='top', fontsize=8)

        plt.suptitle(f'Delta Plot Matrix - Participant {pnum}', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'delta_plot_p{pnum}.png')
        plt.close()

if __name__ == "__main__":
    path = Path(__file__).parent / 'data.csv'

    # Step 1: Read and prepare data for SDT model
    sdt_data = read_data(path, prepare_for='sdt')
    sdt_data['pnum'] = pd.Categorical(sdt_data['pnum']).codes

    # Step 2: Fit hierarchical SDT model
    model, trace = apply_hierarchical_sdt_model(sdt_data)

    # Step 3: Save posterior summaries
    summary = az.summary(trace, var_names=["mean_d", "mean_c", "sd_d", "sd_c"])
    print(summary)
    summary.to_csv(OUTPUT_DIR / 'sdt_summary.csv')

    # Step 4: Posterior plots
    az.plot_posterior(trace, var_names=["mean_d", "mean_c"], hdi_prob=0.94)
    plt.suptitle("Posterior Distributions (94% HDI)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sdt_posteriordistributions.png")
    plt.close()

    # Step 5: Trace plots
    az.plot_trace(trace, var_names=["mean_d", "mean_c", "sd_d", "sd_c"])
    plt.suptitle("Trace Plots of SDT Parameters")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sdt_traceplots.png")
    plt.close()

    # Step 6: Delta plots
    plot_delta_plots(str(path))

    print("All figures and CSV summaries saved in:", OUTPUT_DIR)