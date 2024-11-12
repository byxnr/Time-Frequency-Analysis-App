import os
import numpy as np
import json
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet

# Set MNE log level to suppress unnecessary output
mne.set_log_level('error')

# Define paths
data_path = os.path.expanduser("C:/Users/bndmk/Downloads/sample_introduction/data")
epoch_fname = os.path.join(data_path, "output/out_dir/meg-epo.fif")

'''
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load brainlife config.json
with open(__location__+'/config.json') as config_json:
    config = json.load(config_json)

'''

# Define output directories
out_dir = os.path.join(os.getcwd(), 'out_dir')
out_figs = os.path.join(out_dir, 'out_figs')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_figs, exist_ok=True)

# Initialize the HTML report
report = mne.Report(title='Time-Frequency Analysis Report')

# Load epochs
epochs = mne.read_epochs(epoch_fname, proj=True)

# Define frequency bands and compute time-frequency representations
freqs = np.logspace(*np.log10([5, 30]), num=8)
n_cycles = 3.0
power, itc = tfr_morlet(
    epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
    return_itc=True, decim=3, n_jobs=1
)

# Plot average power topomap and save it
fig1 = power.plot_topo(baseline=(-0.3, 0), mode="logratio", title="Average power", show=False)
fig1_path = os.path.join(out_figs, 'average_power_topo.png')
fig1.savefig(fig1_path)
report.add_figure(fig=fig1, title="Average Power Topomap", caption="Topographic plot of average power across channels.", image_format='png')

# Plot specific channels and save (modify the channel index as needed)
fig2 = power.plot(picks=[83], baseline=(-0.3, 0), mode="logratio", title=power.ch_names[83], show=False)
fig2_path = os.path.join(out_figs, 'power_channel_83.png')
if isinstance(fig2, list):
    for i, fig in enumerate(fig2):
        fig.savefig(os.path.join(out_figs, f'power_channel_83_{i}.png'))
        report.add_figure(fig=fig, title=f"Power for Channel {power.ch_names[83]} ({i})", caption=f"Power spectrum for channel {power.ch_names[83]}.", image_format='png')
else:
    fig2.savefig(fig2_path)
    report.add_figure(fig=fig2, title=f"Power for Channel {power.ch_names[83]}", caption=f"Power spectrum for channel {power.ch_names[83]}.", image_format='png')

# Define topomap plot settings and plot frequency bands
fig, axes = plt.subplots(1, 2, figsize=(7, 4), layout="constrained")
topomap_kw = dict(
    ch_type="mag", tmin=0.5, tmax=1.5, baseline=(-0.3, 0), mode="logratio", show=False
)
plot_dict = {
    'Alpha': {'fmin': 8, 'fmax': 12},
    'Beta': {'fmin': 12, 'fmax': 30},
    'Theta': {'fmin': 4, 'fmax': 8},
    'Gamma': {'fmin': 30, 'fmax': 40}
}
for ax, (title, fmin_fmax) in zip(axes, plot_dict.items()):
    power.plot_topomap(fmin=fmin_fmax['fmin'], fmax=fmin_fmax['fmax'], axes=ax, **topomap_kw)
    ax.set_title(title)
fig_path = os.path.join(out_figs, 'frequency_bands_topomap.png')
fig.savefig(fig_path)
report.add_figure(fig=fig, title="Frequency Bands Topomap", caption="Topographic plots for different frequency bands.", image_format='png')

# Plot joint time-frequency power representation and save
fig3 = power.plot_joint(
    baseline=(-0.3, 0), mode="mean", tmin=-0.5, tmax=2, timefreqs=[(0.15, 6.), (1., 10.)], show=False
)
if isinstance(fig3, list):
    for i, fig in enumerate(fig3):
        fig.savefig(os.path.join(out_figs, f'joint_timefreq_power_{i}.png'))
        report.add_figure(fig=fig, title=f"Joint Time-Frequency Power {i}", caption="Joint time-frequency power representation.", image_format='png')
else:
    fig3_path = os.path.join(out_figs, 'joint_timefreq_power.png')
    fig3.savefig(fig3_path)
    report.add_figure(fig=fig3, title="Joint Time-Frequency Power", caption="Joint time-frequency power representation.", image_format='png')

# Plot inter-trial coherence topomap and save
fig4 = itc.plot_topo(title="Inter-Trial coherence", vmin=0.0, vmax=1.0, cmap="Reds", show=False)
if isinstance(fig4, list):
    for i, fig in enumerate(fig4):
        fig.savefig(os.path.join(out_figs, f'itc_topo_{i}.png'))
        report.add_figure(fig=fig, title=f"Inter-Trial Coherence {i}", caption="Inter-trial coherence topographic plot.", image_format='png')
else:
    fig4_path = os.path.join(out_figs, 'itc_topo.png')
    fig4.savefig(fig4_path)
    report.add_figure(fig=fig4, title="Inter-Trial Coherence", caption="Inter-trial coherence topographic plot.", image_format='png')

# Close all figures to free memory
plt.close('all')

# Save the HTML report
report.save(os.path.join(out_dir, 'time_frequency_analysis_report.html'), overwrite=True, open_browser=False)
