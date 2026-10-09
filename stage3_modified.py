import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import mne
from autoreject import AutoReject

mne.set_log_level("WARNING")

RESULTS_DIR = Path(r"/Volumes/Misophonia_EEG")

subject_list = ["103", "104", "105", "106", "107", "108", "109",
                "110", "112", "113", "115", "119", "121", "122",
                "123", "124", "126", "129", "130", "131", "133",
                "134", "145", "147", "148", "152", "153", "155",
                "156", "157", "158", "159", "162", "174", "178",
                "179", "196", "202", "213"]
# 139 ses-2, 154 stage 1

# Early auditory ERP components: P1, N1, P2
EARLY_ERP_ROI = [
    "F3", "Fz", "F4",
    "FC3", "FCz", "FC4",
    "C3", "Cz", "C4",
]

# Late positive potential: LPP
LPP_ROI = [
    "CP3", "CPz", "CP4",
    "P3", "Pz", "P4",
    "O1", "Oz", "O2",
]

ERP_ROI = EARLY_ERP_ROI + LPP_ROI

# A full recording has 202 events; the first 2 are excluded
EXPECTED_N_EVENTS = 202
N_LEADING_EVENTS_TO_DROP = 2

EPOCH_TMIN = -0.2
EPOCH_TMAX = 3.0
BASELINE = (-0.2, 0)

GRADIENT_THRESHOLD = 5          # µV/ms
WINDOW_PTP_THRESHOLD = 100      # µV
ABS_AMPLITUDE_THRESHOLD = 100   # µV

PERSISTENT_CHANNEL_THRESHOLD = 50  # % of epochs artifact-flagged
EPOCH_REJECT_N_ROI = 4

AR_N_INTERPOLATE = [1, 2, 4]
AR_CONSENSUS = [0.50, 0.75, 1.00]
AR_RANDOM_STATE = 42

COMPONENT_WINDOWS = {
    "P1": (0.030, 0.060),
    "N1": (0.080, 0.110),
    "P2": (0.120, 0.200),
    "LPP": (0.300, 0.800),
}

LOG_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("stage3")


def setup_logging():
    # Console + one run-level log file. Per-subject log files are attached in main().
    # Root stays at WARNING so third-party INFO chatter is suppressed; this script logs at INFO.
    run_log = RESULTS_DIR / f"stage3_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    console = logging.StreamHandler()
    run_file = logging.FileHandler(run_log)
    for handler in (console, run_file):
        handler.setFormatter(LOG_FORMATTER)
    logging.basicConfig(level=logging.WARNING, handlers=[console, run_file], force=True)
    logger.setLevel(logging.INFO)
    # Route Python/MNE RuntimeWarnings (e.g. empty-slice nanmean) into the logs too
    logging.captureWarnings(True)
    return run_log


def format_event_counts(events, event_id):
    return "\n".join(
        f"  {label}: {np.sum(events[:, 2] == code)}" for label, code in event_id.items()
    )


def summarize_metric(values, name, unit):
    values = np.asarray(values).ravel()

    return pd.Series({
        "metric": name,
        "unit": unit,
        "mean": np.mean(values),
        "SD": np.std(values, ddof=1),
        "median": np.median(values),
        "IQR": np.percentile(values, 75) - np.percentile(values, 25),
        "min": np.min(values),
        "P01": np.percentile(values, 1),
        "P05": np.percentile(values, 5),
        "P25": np.percentile(values, 25),
        "P75": np.percentile(values, 75),
        "P95": np.percentile(values, 95),
        "P99": np.percentile(values, 99),
        "max": np.max(values),
    })


def plot_condition_erps(evokeds, sub, title_prefix):
    fig, ax = plt.subplots(figsize=(12, 6))

    for condition, evoked in evokeds.items():
        # Average across the ERP ROI; nanmean skips channels with no usable trials
        roi_present = [ch for ch in ERP_ROI if ch in evoked.ch_names]
        roi_evoked = evoked.copy().pick(roi_present)
        waveform = np.nanmean(roi_evoked.data, axis=0) * 1e6

        ax.plot(roi_evoked.times, waveform, label=condition, linewidth=1.8)

    # Component windows
    for name, (start, stop) in COMPONENT_WINDOWS.items():
        ax.axvspan(start, stop, alpha=0.12)
        ax.text(
            (start + stop) / 2, 0.98, name,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=10, fontweight="bold",
        )

    # Reference lines
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8, linestyle="--")

    ax.set_xlim(-0.2, 0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{sub} — {title_prefix} — ERP ROI Average")
    ax.legend(title="Condition", loc="best", frameon=True)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    return fig


def process_subject(sub):
    out_dir = RESULTS_DIR / sub
    report = mne.Report(title=f"Stage 3 Epoching & Artifact Rejection — Subject {sub}")

    # =========================================================
    # Load stage-2 checkpoint
    # =========================================================

    logger.info(f"Loading stage-2 checkpoint for subject {sub}...")

    raw_file = out_dir / f"{sub}_stage2_cleaned_raw.fif"
    events_file = out_dir / f"{sub}_stage1_events.npy"
    event_id_file = out_dir / f"{sub}_stage1_event_id.json"

    for file in [raw_file, events_file, event_id_file]:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    raw = mne.io.read_raw_fif(raw_file, preload=True)
    events = np.load(events_file)
    with open(event_id_file) as f:
        event_id = json.load(f)

    montage = raw.get_montage()

    logger.info(
        f"Loaded: {raw.info['sfreq']} Hz, {len(raw.ch_names)} channels, {len(events)} events, "
        f"{len(montage.ch_names) if montage else 0} montage positions"
    )
    logger.info(f"event_id: {event_id}")
    logger.info("Event counts (all):\n" + format_event_counts(events, event_id))

    n_leading_dropped = 0
    if len(events) == EXPECTED_N_EVENTS:
        n_leading_dropped = N_LEADING_EVENTS_TO_DROP
        events = events[n_leading_dropped:].copy()
        logger.info(
            f"Exactly {EXPECTED_N_EVENTS} events detected — "
            f"excluding the first {n_leading_dropped} events."
        )
    elif len(events) < EXPECTED_N_EVENTS:
        logger.warning(
            f"{len(events)} events detected (<{EXPECTED_N_EVENTS}) — keeping all events."
        )
    else:
        raise ValueError(
            f"Unexpected number of events: {len(events)}. "
            f"More than {EXPECTED_N_EVENTS} events found."
        )

    logger.info(
        f"Events used for analysis: {len(events)}\n" + format_event_counts(events, event_id)
    )

    missing_roi = [ch for ch in ERP_ROI if ch not in raw.ch_names]
    if missing_roi:
        raise ValueError(f"Missing ERP ROI channels: {missing_roi}")

    # Channels stage 2 left marked bad (too many bads to interpolate) are still in the data
    # but were never repaired, so they are excluded from the ERP ROI up front.
    stage2_bads = list(raw.info["bads"])
    stage2_roi_bads = [ch for ch in ERP_ROI if ch in stage2_bads]
    if stage2_roi_bads:
        logger.warning(f"Stage-2 bad channels inside the ERP ROI (excluded): {stage2_roi_bads}")
    logger.info(f"Stage-2 bad channels carried over: {stage2_bads or 'none'}")

    logger.info(
        f"ERP regions of interest: P1/N1/P2 {EARLY_ERP_ROI}; LPP {LPP_ROI} "
        f"({len(ERP_ROI)} channels)"
    )

    # =========================================================
    # Master baseline-corrected epochs
    # =========================================================

    logger.info("Constructing master baseline-corrected epochs...")

    epochs_master = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=BASELINE,
        picks="eeg",
        preload=True,
        reject=None,
        detrend=None,
    )

    mne_dropped = {
        i: reasons for i, reasons in enumerate(epochs_master.drop_log)
        if reasons and reasons != ("IGNORED",)
    }
    if mne_dropped:
        logger.warning(f"MNE dropped {len(mne_dropped)} epochs while epoching: {mne_dropped}")

    logger.info(
        f"Master epochs: {len(epochs_master)} "
        f"({epochs_master.tmin:.3f} to {epochs_master.tmax:.3f} s, "
        f"{epochs_master.info['sfreq']:.2f} Hz, {len(epochs_master.ch_names)} EEG channels)\n"
        + "\n".join(f"  {label}: {len(epochs_master[label])}" for label in event_id)
    )

    condition_labels = {v: k for k, v in event_id.items()}

    epochs_master.metadata = pd.DataFrame({
        "subject": sub,
        "condition": [condition_labels[code] for code in epochs_master.events[:, 2]],
        "event_code": epochs_master.events[:, 2],
        "trial_number": np.arange(len(epochs_master)),
        # Row in the stage-1 events file (accounts for leading events dropped above
        # and for any epochs MNE dropped)
        "original_event_index": epochs_master.selection + n_leading_dropped,
    })

    # =========================================================
    # Artifact metrics
    # =========================================================

    logger.info("Calculating artifact metrics...")

    eeg_picks = mne.pick_types(epochs_master.info, eeg=True, exclude=[])
    data_uv = epochs_master.get_data(picks=eeg_picks) * 1e6
    ch_names = [epochs_master.ch_names[i] for i in eeg_picks]
    sfreq = epochs_master.info["sfreq"]
    n_epochs, n_channels, n_times = data_uv.shape

    logger.info(f"Data shape: {data_uv.shape}")

    # 1. Maximum voltage gradient
    ms_per_sample = 1000.0 / sfreq
    gradient = np.abs(np.diff(data_uv, axis=2)) / ms_per_sample
    max_gradient = gradient.max(axis=2)

    # 2. Maximum 500-ms peak-to-peak amplitude
    window_samples = int(round(0.5 * sfreq))
    window_max_ptp = np.zeros((n_epochs, n_channels))
    step = max(1, window_samples // 4)

    for start in range(0, n_times - window_samples + 1, step):
        window = data_uv[:, :, start:start + window_samples]
        ptp = window.max(axis=2) - window.min(axis=2)
        window_max_ptp = np.maximum(window_max_ptp, ptp)

    # 3. Maximum absolute amplitude
    max_abs_amplitude = np.abs(data_uv).max(axis=2)

    metric_summary = pd.DataFrame([
        summarize_metric(max_gradient, "Maximum gradient per channel × epoch", "µV/ms"),
        summarize_metric(window_max_ptp, "Maximum 500-ms PTP per channel × epoch", "µV"),
        summarize_metric(max_abs_amplitude, "Maximum absolute amplitude per channel × epoch", "µV"),
    ])

    logger.info(
        "Artifact metric distributions:\n"
        + metric_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    gradient_bad = max_gradient > GRADIENT_THRESHOLD
    window_bad = window_max_ptp > WINDOW_PTP_THRESHOLD
    abs_bad = max_abs_amplitude > ABS_AMPLITUDE_THRESHOLD

    combined_bad = gradient_bad | window_bad | abs_bad

    flag_lines = [
        f"  {name}: {mask.sum():,} channel × epoch observations; "
        f"{np.any(mask, axis=1).sum()} / {n_epochs} epochs affected"
        for name, mask in [
            (f"Gradient > {GRADIENT_THRESHOLD} µV/ms", gradient_bad),
            (f"500-ms PTP > {WINDOW_PTP_THRESHOLD} µV", window_bad),
            (f"Absolute amp > {ABS_AMPLITUDE_THRESHOLD} µV", abs_bad),
            ("Combined", combined_bad),
        ]
    ]
    logger.info("Artifact screening:\n" + "\n".join(flag_lines))

    channel_diagnostics = pd.DataFrame({
        "channel": ch_names,
        "combined_n": combined_bad.sum(axis=0),
        "combined_pct": 100 * combined_bad.mean(axis=0),
    })

    logger.info(
        "Channel-level artifact burden (top 10; full table saved to CSV):\n"
        + channel_diagnostics.sort_values("combined_pct", ascending=False)
        .head(10).to_string(index=False)
    )

    # =========================================================
    # Rule-based artifact decisions
    # =========================================================

    # Decision hierarchy:
    #
    # 1. Persistent channel:
    #       >50% of epochs artifact-flagged
    #       → exclude channel for this participant
    #
    # 2. Whole-epoch rejection:
    #       >=4 usable ERP-ROI channels artifact-flagged
    #       → reject entire epoch
    #
    # 3. Retained epoch with 1–3 bad channels:
    #       → keep epoch
    #       → exclude affected channels from that epoch's ERP average

    logger.info("Applying rule-based artifact decisions...")

    # 1. Persistent channels
    persistent_channels = channel_diagnostics.loc[
        channel_diagnostics["combined_pct"] > PERSISTENT_CHANNEL_THRESHOLD, "channel"
    ].tolist()

    if persistent_channels:
        logger.warning(
            f"Persistent channels (>{PERSISTENT_CHANNEL_THRESHOLD}% epochs flagged): "
            f"{persistent_channels}"
        )
    else:
        logger.info("Persistent channels: none")

    # 2. Usable ERP ROI
    usable_roi_channels = [
        ch for ch in ERP_ROI
        if ch not in persistent_channels and ch not in stage2_bads
    ]
    if not usable_roi_channels:
        raise ValueError("No usable ERP ROI channels left after persistent / stage-2 exclusions")
    usable_roi_idx = [ch_names.index(ch) for ch in usable_roi_channels]

    logger.info(
        f"ERP ROI: {len(ERP_ROI)} original, "
        f"{len(ERP_ROI) - len(usable_roi_channels)} excluded (persistent or stage-2 bad), "
        f"{len(usable_roi_channels)} usable"
    )

    # 3. Channel × epoch artifact mask
    roi_bad = combined_bad[:, usable_roi_idx]

    # 4. Number of bad ERP channels per epoch
    bad_channels_usable_roi = roi_bad.sum(axis=1)

    # 5. Whole-epoch rejection
    epoch_reject_rule = bad_channels_usable_roi >= EPOCH_REJECT_N_ROI
    epoch_keep_rule = ~epoch_reject_rule

    # 6. Channel usability
    channel_usable_rule = ~roi_bad

    # 7. Record artifact information in metadata
    metadata_rule = epochs_master.metadata.copy()
    metadata_rule["bad_erp_channel_count"] = bad_channels_usable_roi
    metadata_rule["rule_epoch_rejected"] = epoch_reject_rule

    # Specific bad ERP channels for EACH epoch
    metadata_rule["bad_erp_channels"] = [
        ";".join(usable_roi_channels[ch_i] for ch_i in np.where(roi_bad[epoch_i])[0])
        for epoch_i in range(n_epochs)
    ]

    # 8. Summary
    logger.info(
        "RULE-BASED ARTIFACT RESULTS: "
        f"rejected {epoch_reject_rule.sum()} / {n_epochs}, "
        f"retained {epoch_keep_rule.sum()} / {n_epochs} "
        f"({100 * epoch_keep_rule.mean():.1f}%)\n"
        "Bad ERP channels per retained epoch:\n"
        + pd.Series(bad_channels_usable_roi[epoch_keep_rule])
        .value_counts().sort_index().to_string()
    )

    # Create rule-based clean epochs
    epochs_rule_clean = epochs_master[epoch_keep_rule].copy()

    metadata_clean = metadata_rule.loc[epoch_keep_rule].reset_index(drop=True).copy()

    # Preserve persistent bad channels separately from trial-specific flags.
    metadata_clean["persistent_bad_channels"] = ";".join(persistent_channels)
    metadata_clean["stage2_bad_channels"] = ";".join(stage2_bads)

    # bad_erp_channels contains trial-specific artifact flags
    # among usable ROI channels. Persistent channels are excluded
    # from every trial's ERP calculation as well, and are marked
    # bad in the saved epochs so downstream averaging skips them.
    epochs_rule_clean.metadata = metadata_clean
    epochs_rule_clean.info["bads"] = stage2_bads + [
        ch for ch in persistent_channels if ch not in stage2_bads
    ]

    logger.info(
        f"Rule-based clean epochs: {len(epochs_master)} → {len(epochs_rule_clean)}; "
        f"bads marked in saved epochs: {epochs_rule_clean.info['bads'] or 'none'}"
    )

    # ------------------------------------------------------------
    # Artifact matrix
    #
    # rows    = usable ERP ROI channels
    # columns = epochs
    #
    # 0 = channel usable in that epoch
    # 1 = channel flagged as artifact in that epoch
    # ------------------------------------------------------------

    rule_artifact_matrix = roi_bad.T

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(rule_artifact_matrix, aspect="auto", interpolation="nearest", origin="upper")

    ax.set_yticks(np.arange(len(usable_roi_channels)))
    ax.set_yticklabels(usable_roi_channels)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ERP ROI channel")
    ax.set_title(f"{sub} — Rule-Based ERP ROI Artifact / Rejection Map")

    # Mark boundary between Early ERP and LPP
    n_early_usable = sum(ch in usable_roi_channels for ch in EARLY_ERP_ROI)
    if 0 < n_early_usable < len(usable_roi_channels):
        ax.axhline(n_early_usable - 0.5, linewidth=1.5)

    # Mark whole-epoch rejected trials (>= EPOCH_REJECT_N_ROI bad ROI channels)
    for epoch_idx in np.where(epoch_reject_rule)[0]:
        ax.axvline(epoch_idx, linewidth=1.5, alpha=0.8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Usable", "Channel artifact"])
    plt.tight_layout()

    report.add_html(
        metric_summary.to_html(index=False, float_format=lambda x: f"{x:.2f}")
        + "<pre>" + "\n".join(flag_lines) + "</pre>",
        title="Artifact metric distributions",
    )
    report.add_figure(fig, title="Rule-based ERP ROI artifact / rejection map")
    plt.close(fig)
    report.add_html(
        f"<p>Persistent channels: {persistent_channels or 'none'}</p>"
        f"<p>Stage-2 bad channels: {stage2_bads or 'none'}</p>"
        f"<p>Usable ROI channels ({len(usable_roi_channels)}): {usable_roi_channels}</p>"
        f"<p>Retained {epoch_keep_rule.sum()} / {n_epochs} epochs "
        f"({100 * epoch_keep_rule.mean():.1f}%)</p>",
        title="Rule-based results",
    )

    # =========================================================
    # AutoReject
    # =========================================================

    logger.info("Running AutoReject...")

    ar = AutoReject(
        n_interpolate=AR_N_INTERPOLATE,
        consensus=AR_CONSENSUS,
        random_state=AR_RANDOM_STATE,
        verbose=True,
    )

    epochs_ar_clean, reject_log_ar = ar.fit_transform(epochs_master.copy(), return_log=True)

    n_original = len(epochs_master)
    n_clean = len(epochs_ar_clean)
    n_rejected = int(reject_log_ar.bad_epochs.sum())
    retention = 100 * n_clean / n_original

    logger.info(
        f"AutoReject results: {n_original} → {n_clean} epochs; rejected {n_rejected}; "
        f"retention {retention:.1f}%; n_interpolate {ar.n_interpolate_}; "
        f"consensus {ar.consensus_}"
    )

    # AutoReject skips channels in info["bads"] (their labels are NaN), so leave them off the map
    ar_roi_channels = [ch for ch in ERP_ROI if ch not in epochs_master.info["bads"]]
    ar_roi_idx = [epochs_master.ch_names.index(ch) for ch in ar_roi_channels]
    ar_roi_matrix = (reject_log_ar.labels[:, ar_roi_idx] > 0).T

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(ar_roi_matrix, aspect="auto", interpolation="nearest", origin="upper")

    ax.set_yticks(np.arange(len(ar_roi_channels)))
    ax.set_yticklabels(ar_roi_channels)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ERP ROI channel")
    ax.set_title(f"{sub} — AutoReject ERP ROI Artifact Map")

    n_early = sum(ch in ar_roi_channels for ch in EARLY_ERP_ROI)
    if 0 < n_early < len(ar_roi_channels):
        ax.axhline(n_early - 0.5, linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Not flagged", "Flagged"])
    plt.tight_layout()

    report.add_figure(fig, title="AutoReject ERP ROI artifact map")
    plt.close(fig)

    # =========================================================
    # Evokeds
    # =========================================================

    evokeds_ar = {}

    for condition in event_id:
        condition_epochs = epochs_ar_clean[epochs_ar_clean.metadata["condition"] == condition]
        if len(condition_epochs) == 0:
            logger.warning(f"AutoReject: no epochs left for condition {condition}; skipping evoked.")
            continue
        evokeds_ar[condition] = condition_epochs.average(picks="eeg")

    logger.info(
        "AutoReject evokeds:\n" + "\n".join(
            f"  {condition}: {len(evoked.ch_names)} channels, {evoked.nave} epochs"
            for condition, evoked in evokeds_ar.items()
        )
    )

    # Rule-based ROI-aware evokeds: each channel is averaged only over the trials
    # in which it was not artifact-flagged (excluded channels are left out entirely).
    evokeds_rule = {}

    rule_roi_idx = [epochs_rule_clean.ch_names.index(ch) for ch in usable_roi_channels]
    rule_roi_data = epochs_rule_clean.get_data(picks=rule_roi_idx)  # epochs × channels × times
    rule_roi_usable = channel_usable_rule[epoch_keep_rule]
    rule_roi_info = mne.pick_info(epochs_rule_clean.info, rule_roi_idx)

    for condition in event_id:
        cond_mask = (metadata_clean["condition"] == condition).to_numpy()
        if not cond_mask.any():
            logger.warning(f"Rule-based: no epochs left for condition {condition}; skipping evoked.")
            continue

        roi_data = rule_roi_data[cond_mask].copy()
        usable = rule_roi_usable[cond_mask]
        roi_data[~usable] = np.nan

        no_trials = [ch for ch, ok in zip(usable_roi_channels, usable.any(axis=0)) if not ok]
        if no_trials:
            logger.warning(f"Rule-based {condition}: no usable trials for channels {no_trials} (NaN)")

        # Average over trials separately for each channel
        mean_data = np.nanmean(roi_data, axis=0)

        evokeds_rule[condition] = mne.EvokedArray(
            mean_data,
            rule_roi_info,
            tmin=epochs_rule_clean.tmin,
            nave=int(cond_mask.sum()),
            comment=condition,
        )

    logger.info(
        "Rule-based ROI-aware evokeds:\n" + "\n".join(
            f"  {condition}: {len(evoked.ch_names)} ROI channels, {evoked.nave} epochs"
            for condition, evoked in evokeds_rule.items()
        )
    )

    fig = plot_condition_erps(evokeds_rule, sub, "Rule-Based")
    report.add_figure(fig, title="Rule-based ERP ROI average")
    plt.close(fig)

    fig = plot_condition_erps(evokeds_ar, sub, "AutoReject")
    report.add_figure(fig, title="AutoReject ERP ROI average")
    plt.close(fig)

    # =========================================================
    # Save rule-based outputs
    # =========================================================

    rule_dir = out_dir / "stage3_rule_based"
    rule_dir.mkdir(parents=True, exist_ok=True)

    # Clean epochs (includes trial-level artifact metadata)
    epochs_rule_clean.save(rule_dir / f"{sub}_stage3_rule_based_clean-epo.fif", overwrite=True)

    # Artifact masks and decisions
    np.savez_compressed(
        rule_dir / f"{sub}_stage3_rule_based_artifact_masks.npz",
        gradient_bad=gradient_bad,
        window_bad=window_bad,
        abs_bad=abs_bad,
        combined_bad=combined_bad,
        roi_bad=roi_bad,
        channel_usable_rule=channel_usable_rule,
        epoch_reject_rule=epoch_reject_rule,
        epoch_keep_rule=epoch_keep_rule,
        bad_channels_usable_roi=bad_channels_usable_roi,
        ch_names=np.array(ch_names),
        usable_roi_channels=np.array(usable_roi_channels),
        persistent_channels=np.array(persistent_channels),
    )

    # Channel-level diagnostic summary
    channel_diagnostics.to_csv(
        rule_dir / f"{sub}_stage3_channel_artifact_diagnostics.csv", index=False
    )

    # Processing configuration
    artifact_config = {
        "thresholds": {
            "gradient_uv_per_ms": GRADIENT_THRESHOLD,
            "window_ptp_uv": WINDOW_PTP_THRESHOLD,
            "absolute_amplitude_uv": ABS_AMPLITUDE_THRESHOLD,
        },
        "persistent_channel_threshold_pct": PERSISTENT_CHANNEL_THRESHOLD,
        "epoch_reject_n_roi": EPOCH_REJECT_N_ROI,
        "epoch_tmin": EPOCH_TMIN,
        "epoch_tmax": EPOCH_TMAX,
        "baseline": list(BASELINE),
        "event_id": event_id,
        "n_leading_events_dropped": n_leading_dropped,
        "persistent_channels": persistent_channels,
        "stage2_bad_channels": stage2_bads,
        "usable_roi_channels": usable_roi_channels,
    }

    with open(rule_dir / f"{sub}_stage3_rule_based_config.json", "w") as f:
        json.dump(artifact_config, f, indent=2)

    logger.info(f"Rule-based outputs saved to: {rule_dir}")

    # =========================================================
    # Save AutoReject outputs
    # =========================================================

    ar_dir = out_dir / "stage3_autoreject"
    ar_dir.mkdir(parents=True, exist_ok=True)

    # Cleaned epochs and reject log
    epochs_ar_clean.save(ar_dir / f"{sub}_stage3_autoreject_clean-epo.fif", overwrite=True)

    np.savez_compressed(
        ar_dir / f"{sub}_stage3_autoreject_log.npz",
        labels=reject_log_ar.labels,
        bad_epochs=reject_log_ar.bad_epochs,
        ch_names=np.array(epochs_master.ch_names),
    )

    # Channel-level artifact summary
    ar_channel_counts = pd.DataFrame({
        "channel": epochs_master.ch_names,
        "bad_count": (reject_log_ar.labels > 0).sum(axis=0),
    })
    ar_channel_counts["bad_pct"] = 100 * ar_channel_counts["bad_count"] / n_original
    ar_channel_counts = ar_channel_counts.sort_values("bad_pct", ascending=False)
    ar_channel_counts.to_csv(
        ar_dir / f"{sub}_stage3_autoreject_channel_diagnostics.csv", index=False
    )

    # Learned thresholds (convert V to µV)
    autoreject_thresholds = pd.DataFrame({
        "channel": list(ar.threshes_.keys()),
        "threshold_uv": [x * 1e6 for x in ar.threshes_.values()],
    }).sort_values("threshold_uv")
    autoreject_thresholds.to_csv(ar_dir / f"{sub}_stage3_autoreject_thresholds.csv", index=False)

    # Configuration and results
    autoreject_config = {
        "n_interpolate_candidates": AR_N_INTERPOLATE,
        "consensus_candidates": AR_CONSENSUS,
        "selected_n_interpolate": int(ar.n_interpolate_["eeg"]),
        "selected_consensus": float(ar.consensus_["eeg"]),
        "random_state": AR_RANDOM_STATE,
        "n_original_epochs": n_original,
        "n_clean_epochs": n_clean,
        "n_rejected_epochs": n_rejected,
        "retention_pct": retention,
        "epoch_tmin": EPOCH_TMIN,
        "epoch_tmax": EPOCH_TMAX,
        "baseline": list(BASELINE),
    }

    with open(ar_dir / f"{sub}_stage3_autoreject_config.json", "w") as f:
        json.dump(autoreject_config, f, indent=2)

    logger.info(f"AutoReject outputs saved to: {ar_dir}")

    report.add_html(
        f"<p>Epochs: {n_original} &rarr; {n_clean}; rejected {n_rejected}; "
        f"retention {retention:.1f}%</p>"
        f"<p>Selected n_interpolate: {ar.n_interpolate_}; consensus: {ar.consensus_}</p>",
        title="AutoReject results",
    )
    report_html_path = out_dir / f"{sub}_stage3_report.html"
    report.save(report_html_path, overwrite=True, open_browser=False)
    logger.info(f"Report saved to: {report_html_path}")

    return {
        "subject": sub,
        "n_epochs": n_epochs,
        "rule_retained": int(epoch_keep_rule.sum()),
        "rule_retention_pct": round(100 * float(epoch_keep_rule.mean()), 1),
        "persistent_channels": ";".join(persistent_channels),
        "stage2_bads": ";".join(stage2_bads),
        "ar_retained": n_clean,
        "ar_retention_pct": round(retention, 1),
    }


def main(subjects=subject_list):
    run_log = setup_logging()
    logger.info(f"Stage 3 started for {len(subjects)} subjects. Run log: {run_log}")

    summaries = []
    sub_errors = []

    for sub in subjects:
        out_dir = RESULTS_DIR / sub
        if not out_dir.is_dir():
            logger.error(f"Subject {sub}: directory {out_dir} not found; skipping.")
            sub_errors.append(sub)
            continue

        # Per-subject log file alongside that subject's outputs
        sub_handler = logging.FileHandler(out_dir / f"{sub}_stage3.log", mode="w")
        sub_handler.setFormatter(LOG_FORMATTER)
        logging.getLogger().addHandler(sub_handler)

        t0 = time.time()
        try:
            logger.info(f"===== Subject {sub} =====")
            summaries.append(process_subject(sub))
            logger.info(f"Subject {sub} finished in {time.time() - t0:.0f} s")
        except Exception:
            logger.exception(f"Error occurred for {sub}")
            sub_errors.append(sub)
        finally:
            plt.close("all")
            logging.getLogger().removeHandler(sub_handler)
            sub_handler.close()

    if summaries:
        logger.info("Stage 3 summary:\n" + pd.DataFrame(summaries).to_string(index=False))
    logger.info(f"Subjects returning errors: {sub_errors}.")


if __name__ == "__main__":
    main()
