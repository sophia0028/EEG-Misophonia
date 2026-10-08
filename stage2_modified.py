import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend
import matplotlib.pyplot as plt
import time

import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep import NoisyChannels

def status(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}")

sub_errors = []

subject_list = ['196', '202']

# 139 ses, 154 stage 1 

for sub in subject_list:
    try:
        CONFIG = {
            "results_dir": Path(r"/Volumes/Misophonia_EEG"),
            "subject": sub
        }
        sub = CONFIG["subject"]
        # status(f"Loading stage-1 checkpoint for subject {sub}...")
        out_dir = CONFIG["results_dir"] / sub

        raw = mne.io.read_raw_fif(out_dir / f"{sub}_stage1_resampled_raw.fif", preload=True)
        events = np.load(out_dir / f"{sub}_stage1_events.npy")
        with open(out_dir / f"{sub}_stage1_event_id.json") as f:
            event_id = json.load(f)

        sfreq = raw.info["sfreq"]
        first_t = (events[0, 0] - raw.first_samp) / sfreq
        last_t = (events[-1, 0] - raw.first_samp) / sfreq

        tmin = max(0.0, first_t - 10.0)
        tmax = min(raw.times[-1], last_t + 10.0)

        montage = raw.get_montage()

        # print(f"Loaded: {raw.info['sfreq']} Hz, {len(raw.ch_names)} channels, {len(events)} events")
        # print("event_id:", event_id)
        # print(f"Montage: {len(montage.ch_names) if montage else 0} positions")

        # print(f"\nTotal events: {len(events)}")
        for label, code in event_id.items():
            count = np.sum(events[:, 2] == code)
            # print(f"  {label}: {count}")


        # status("Computing PSD before filtering (this can take a moment)...")
        report = mne.Report(title=f"Stage 2 Preprocessing — Subject {sub}")
        report.add_raw(raw, title="Stage 1 checkpoint (as loaded)", psd=False)

        psd_prefilter = raw.compute_psd(
            method="welch", fmin=0.05, fmax=60, picks="eeg",
            n_fft=2048, n_overlap=1024, average="median",
        )
        fig_psd_prefilter = psd_prefilter.plot(picks="eeg", exclude=[], dB=True)
        report.add_figure(fig_psd_prefilter, title="PSD before filtering")

        # status("Applying high-pass -> notch -> low-pass filters...")
        raw.filter(l_freq=0.1, h_freq=None, picks="eeg", fir_design="firwin", phase="zero")
        raw.notch_filter(freqs=50, picks="eeg", notch_widths=1, fir_design="firwin")
        raw.filter(l_freq=None, h_freq=40, picks="eeg", fir_design="firwin", phase="zero")

        # print(f"Filtered: {raw.info['highpass']}-{raw.info['lowpass']} Hz (notch at 50 Hz)")

        report.add_html(
            f"<p>High-pass 0.1 Hz &rarr; notch 50 Hz (width 1 Hz) &rarr; low-pass 40 Hz</p>",
            title="Filtering applied",
        )

        # status("Computing PSD after filtering...")
        psd = raw.compute_psd(
            method="welch", fmin=0.05, fmax=60, picks="eeg",
            n_fft=2048, n_overlap=1024, average="median",
        )
        fig_psd = psd.plot(picks="eeg", exclude=[], dB=True)
        report.add_figure(fig_psd, title="PSD after filtering")

        stage1_bads = list(raw.info["bads"])
        # print("Bads carried from stage 1:", stage1_bads)
        raw.info["bads"] = []
        # print("Reset. Current bads:", raw.info["bads"])

        report.add_html(
            f"<p>Bads carried from stage 1: {stage1_bads or 'none'} &rarr; reset to empty list.</p>",
            title="Reset bad channels from stage 1",
        )


        MAX_BADS = 4

        # Dropped right after cropping (never used)
        EARLY_DROP_CHANNELS = ["F11", "F12", "FT11", "FT12"]
        # Kept through bad-channel detection and ICA (blink signature), never interpolated, dropped after ICA
        FRONTAL_POLE_CHANNELS = ["Fp1", "Fp2"]

        dur_before_crop = raw.times[-1]
        raw.crop(tmin=tmin, tmax=tmax)
        early_upper = [ch.upper() for ch in EARLY_DROP_CHANNELS]
        early_present = [ch for ch in raw.ch_names if ch.upper() in early_upper]
        raw.drop_channels(early_present)
        # status(f"Trimmed 10 seconds from beginning and end of triggers. Dropped channels {early_present}.")

        report.add_html(
            f"<p>Kept {tmin:.1f}&ndash;{tmax:.1f} s (first trigger &minus; 10 s to last trigger + 10 s).</p>"
            f"<p>Duration: {dur_before_crop:.1f} s &rarr; {raw.times[-1]:.1f} s</p>"
            f"<p>Dropped channels: {early_present}</p>",
            title="Automatically crop bad segments",
        )


        # status("Average re-reference #1 (before bad-channel detection)...")
        raw.set_eeg_reference(ref_channels="average", projection=False)


        # status("Running automated bad-channel detection (pyprep RANSAC) — this is usually the slowest step, can take a few minutes...")
        nc = NoisyChannels(raw.copy(), random_state=35)
        nc.find_all_bads(ransac=True)
        bads = nc.get_bads(verbose=True)

        criteria = {
            "correlation": nc.bad_by_correlation,
            "deviation": nc.bad_by_deviation,
            "hf_noise": nc.bad_by_hf_noise,
            "ransac": nc.bad_by_ransac,
            "nan": nc.bad_by_nan,
            "flat": nc.bad_by_flat,
            "snr": nc.bad_by_SNR,
            "dropout": nc.bad_by_dropout,
            "psd": nc.bad_by_psd,
            "manual": nc.bad_by_manual,
        }

        # print(f"Bad channels detected (post-reference #1): {bads}")
        # for name, chs in criteria.items():
        #     if chs:
        #         print(f"  by {name}: {chs}")
        # status("Bad-channel detection complete.")

        # FP1/FP2 are never interpolated (kept as-is for ICA, dropped afterwards)
        frontal_upper = [ch.upper() for ch in FRONTAL_POLE_CHANNELS]
        fp_bads = [ch for ch in bads if ch.upper() in frontal_upper]
        interp_bads = [ch for ch in bads if ch.upper() not in frontal_upper]
        if fp_bads:
            # print(f"Frontal pole channels flagged but not interpolated: {fp_bads}")
            pass

        flagged = len(interp_bads) > MAX_BADS

        # Too many bads: interpolate only the MAX_BADS least-bad channels (fewest pyprep criteria
        # flagging them, ties broken by smallest robust deviation z-score). The rest stay marked
        # in raw.info["bads"], so they are excluded from re-referencing and ICA and saved as bads.
        not_interp_bads = []
        if flagged:
            deviations = dict(zip(
                nc.ch_names_new,
                np.abs(nc._extra_info["bad_by_deviation"]["robust_channel_deviations"]),
            ))
            def badness(ch):
                n_criteria = sum(ch in chs for chs in criteria.values())
                return (n_criteria, deviations.get(ch, np.inf))
            ranked = sorted(interp_bads, key=badness)
            interp_bads, not_interp_bads = ranked[:MAX_BADS], ranked[MAX_BADS:]
            # print(
            #     f"FLAGGED: {len(ranked)} bad channels exceeds MAX_BADS={MAX_BADS}. "
            #     f"Interpolating {interp_bads}; leaving marked bad (not interpolated): {not_interp_bads}"
            # )

        report.add_html(
            f"<p>Detected: {bads}</p><pre>{json.dumps(criteria, indent=2)}</pre>"
            f"<p>To interpolate: {interp_bads or 'none'}</p>"
            f"<p>Exceeded MAX_BADS={MAX_BADS}, left marked bad (not interpolated): {not_interp_bads or 'none'}</p>"
            f"<p>Flagged FP1/FP2 (not interpolated): {fp_bads or 'none'}</p>"
            f"<p>Flagged for manual review: {flagged}</p>",
            title="Bad channel detection (pyprep)",
        )

        if interp_bads:
            raw.info["bads"] = interp_bads
            raw.interpolate_bads(reset_bads=True)
        raw.info["bads"] = not_interp_bads
        # print(f"Done. Interpolated: {interp_bads}")


        # Re-reference #2: always run, so the reference reflects interpolated channels
        # status("Average re-reference #2 (after interpolation)...")
        raw.set_eeg_reference(ref_channels="average", projection=False)
        report.add_html(
            f"<p>Average reference re-applied after interpolating {interp_bads or 'no channels'}.</p>",
            title="Average re-reference #2",
        )



        # status("Fitting ICA (infomax, 20 components)...")
        raw_ica_fit = raw.copy().filter(l_freq=1.0, h_freq=None, picks="eeg", fir_design="firwin")
        ica = ICA(n_components=20, method="infomax", fit_params=dict(extended=True),
                random_state=35, max_iter=800)
        ica.fit(raw_ica_fit, picks="eeg")
        print(f"ICA fit complete: {ica.n_components_} components")


        # status("Rendering ICA component topographies...")
        ica.plot_components()


        # status("Running ICLabel classification...")
        matplotlib.use("Agg")

        raw_iclabel = raw.copy().filter(l_freq=1.0, h_freq=40.0, picks="eeg", fir_design="firwin")
        ic_labels = label_components(raw_iclabel, ica, method="iclabel")

        # print("Labels:", ic_labels["labels"])
        # print("Probabilities:", ic_labels["y_pred_proba"])

        eog_iclabel_indices = [
            i for i, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"]))
            if label == "eye blink" and prob > 0.8
        ]
        # print(f"ICLabel-flagged eye components (prob > 0.8): {eog_iclabel_indices}")


        ica.exclude = list(eog_iclabel_indices)
        report.add_ica(
            ica,
            title="ICA decomposition",
            inst=raw_iclabel,
            picks=eog_iclabel_indices if eog_iclabel_indices else None,
        )

        iclabel_df = pd.DataFrame({
            "component": [f"ICA{i:03d}" for i in range(ica.n_components_)],
            "label": ic_labels["labels"],
            "probability": np.round(ic_labels["y_pred_proba"], 3),
            "excluded": [i in eog_iclabel_indices for i in range(ica.n_components_)],
        })
        report.add_html(
            iclabel_df.style
            .apply(lambda r: ["background-color: #fdd" if r["excluded"] else "" for _ in r], axis=1)
            .hide(axis="index")
            .to_html(),
            title="ICLabel classification",
        )


        # status("Applying ICA — removing flagged ocular components from raw...")
        ica.apply(raw, exclude=eog_iclabel_indices)
        # print(f"Applied ICA. Removed components: {eog_iclabel_indices}")

        # Match names case-insensitively (e.g. Fp1 vs FP1) since drop_channels is case-sensitive
        fp_present = [ch for ch in raw.ch_names if ch.upper() in frontal_upper]
        raw.drop_channels(fp_present)
        # status(f"Dropped channels {fp_present}.")

        var_removed = (
            ica.get_explained_variance_ratio(raw_ica_fit, components=eog_iclabel_indices, ch_type="eeg")["eeg"]
            if eog_iclabel_indices else 0.0
        )
        report.add_html(
            f"<p>Removed components: {eog_iclabel_indices or 'none'}</p>"
            f"<p>Variance explained by removed components: {var_removed:.1%}</p>",
            title="ICA applied",
        )



        # status("Saving stage-2 outputs (raw, JSON report, ICA solution, HTML report)...")
        stage2_raw_path = out_dir / f"{sub}_stage2_cleaned_raw.fif"
        raw.save(stage2_raw_path, overwrite=True)
        # print(f"Saved: {stage2_raw_path}")

        stage2_report = {
            "subject": sub,
            "bads_detected": bads,
            "bads_criteria": criteria,
            "n_bads": len(bads),
            "bads_interpolated": interp_bads,
            "frontal_pole_bads_not_interpolated": fp_bads,
            "flagged_for_review": flagged,
            "highpass_hz": raw.info["highpass"],
            "lowpass_hz": raw.info["lowpass"],
            "ica_eog_components_removed": list(map(int, eog_iclabel_indices)),
            "n_ica_components": ica.n_components_,
        }
        with open(out_dir / f"{sub}_stage2_report.json", "w") as f:
            json.dump(stage2_report, f, indent=2)
        # print(f"Saved: {out_dir / f'{sub}_stage2_report.json'}")

        ica.save(out_dir / f"{sub}_stage2_ica.fif", overwrite=True)
        # print(f"Saved: {out_dir / f'{sub}_stage2_ica.fif'}")

        report_html_path = out_dir / f"{sub}_stage2_report.html"
        report.save(report_html_path, overwrite=True, open_browser=False)
        # print(f"Saved: {report_html_path}")

    except Exception as e:
                print(f"Error occurred for {sub}: {e}")
                sub_errors.append(sub)
                continue

print(f"\nSubjects returning errors: {sub_errors}.")