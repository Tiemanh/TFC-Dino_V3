"""Reproducible EDA of raw InGesture; never requires model weights or torch."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['other', 'fluid_intake', 'answering_cell_phone', 'scratching_head',
           'passing_hand_over_face', 'adjusting_glasses_touching_temples',
           'holding_chin', 'stretching_with_hands_behind_neck']
NAMES = ['Other', 'Fluid intake', 'Answer phone', 'Scratch head', 'Hand over face',
         'Adjust glasses', 'Hold chin', 'Stretch neck']
COLORS = ['#2563eb', '#f59e0b', '#14b8a6']


def find_root(base):
    """Detect raw subject folders, including nested Kaggle ZIP extraction."""
    base = Path(base)
    candidates = [base, *sorted(base.rglob('subject_1'))]
    for candidate in candidates:
        root = candidate.parent if candidate.name == 'subject_1' else candidate
        if next(root.glob('subject_*/*__*.csv'), None) is not None:
            return root
    raise FileNotFoundError('Attach/extract InGesture containing subject_*/*__*.csv')


def inventory(root, metadata):
    rows, previews, train_points = [], {}, []
    subject_split = {s: split for key, split in [('train_subjects', 'train'),
                     ('validation_subjects', 'validation'), ('test_subjects', 'test')]
                     for s in metadata[key]}
    rng = np.random.default_rng(42)
    for path in sorted(Path(root).glob('subject_*/*__*.csv')):
        gesture = path.stem.split('__', 1)[0]
        if gesture not in CLASSES:
            continue
        subject = path.parent.name
        if subject not in subject_split:
            raise ValueError(f'Unknown subject {subject}; use matching metadata.json')
        frame = pd.read_csv(path, usecols=['x', 'y', 'z']).apply(pd.to_numeric, errors='coerce')
        frame = frame.replace([np.inf, -np.inf], np.nan)
        missing = int(frame.isna().sum().sum())
        clean = frame.interpolate(limit_direction='both').dropna().to_numpy(dtype=np.float32)
        split = subject_split[subject]
        windows = max(0, 1 + (len(clean) - metadata['window_size']) // metadata['stride'])
        rows.append(dict(file=str(path.relative_to(root)).replace('\\', '/'),
                         subject=subject, gesture=gesture, split=split,
                         raw_samples=len(frame), clean_samples=len(clean),
                         nonfinite_cells=missing, windows=windows,
                         recording_hash=hashlib.sha256(clean.tobytes()).hexdigest(),
                         constant_recording=bool(len(clean) and np.all(np.ptp(clean,axis=0)<1e-8)),
                         seconds=len(clean) / metadata['sampling_rate_hz']))
        if split == 'train' and len(clean):
            chosen = rng.choice(len(clean), min(50, len(clean)), replace=False)
            train_points.append(clean[chosen])
            if gesture not in previews and len(clean) >= metadata['window_size']:
                previews[gesture] = clean[:min(1200, len(clean))]
    if not rows or set(previews) != set(CLASSES):
        raise ValueError('Need nonempty raw data and a training example for each of 8 classes')
    return pd.DataFrame(rows), previews, np.concatenate(train_points)


def save(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / name, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output / name


def class_distribution(frame, metadata, output):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, field, title in zip(axes, ['file', 'windows'], ['Recordings per gesture', 'Overlapping windows per gesture']):
        counts = (frame.groupby('gesture').size() if field == 'file' else frame.groupby('gesture').windows.sum()).reindex(CLASSES)
        ax.barh(NAMES, counts, color='#2563eb')
        ax.set_xscale('log'); ax.invert_yaxis(); ax.set_title(title)
        ax.set_xlabel('Count (log scale)')
        for i, value in enumerate(counts): ax.text(value * 1.03, i, f'{value:,}', va='center', fontsize=8)
        ax.set_xlim(0.8, counts.max() * 2)
    return save(fig, output, '01_class_distribution.png')


def split_distribution(frame, metadata, output):
    counts = frame.pivot_table(index='gesture', columns='split', values='windows', aggfunc='sum', fill_value=0).reindex(CLASSES)[['train', 'validation', 'test']]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    image = ax.imshow(np.log1p(counts.to_numpy()), cmap='Blues', aspect='auto')
    ax.set_xticks(range(3), counts.columns); ax.set_yticks(range(8), NAMES)
    for i in range(8):
        for j in range(3): ax.text(j, i, f'{counts.iloc[i,j]:,}', ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=.75, edgecolor='none'))
    ax.set_title('Windows by subject-disjoint split (counts; log color scale)')
    fig.colorbar(image, ax=ax, label='log(1 + window count)')
    return save(fig, output, '02_split_distribution.png')


def subject_coverage(frame, metadata, output):
    counts = pd.crosstab(frame.subject, frame.gesture).reindex(columns=CLASSES, fill_value=0)
    counts = counts.loc[sorted(counts.index, key=lambda x: int(x.split('_')[1]))]
    fig, ax = plt.subplots(figsize=(13, 13), constrained_layout=True)
    image = ax.imshow(counts, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(8), NAMES, rotation=35, ha='right')
    ax.set_yticks(range(len(counts)), counts.index, fontsize=7)
    ax.set_title('Recording coverage per subject and gesture')
    fig.colorbar(image, ax=ax, label='Number of segmented recordings')
    return save(fig, output, '03_subject_coverage.png')


def quality(frame, metadata, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].boxplot([frame.loc[frame.gesture == c, 'seconds'] for c in CLASSES], vert=False, tick_labels=NAMES, showfliers=False)
    axes[0].set_xscale('log'); axes[0].set_xlabel('Seconds, approximated from sample count / nominal Hz')
    axes[0].set_title('Duration by gesture (outliers hidden)')
    values = [frame.nonfinite_cells.sum(), (frame.raw_samples - frame.clean_samples).sum(), (frame.windows == 0).sum()]
    axes[1].bar(['Nonfinite\ncells', 'Dropped\nrows', 'Files shorter\nthan window'], values, color=['#f59e0b', '#ef4444', '#64748b'])
    for i, v in enumerate(values): axes[1].text(i, v, f'{int(v):,}', ha='center', va='bottom')
    axes[1].set_ylim(0, max(1, max(values) * 1.25)); axes[1].set_title('Data quality audit (different units)')
    return save(fig, output, '04_quality_duration.png')


def signal_examples(previews, metadata, output, frequency=False):
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), constrained_layout=True)
    hz = metadata['sampling_rate_hz']
    for ax, gesture, name in zip(axes.flat, CLASSES, NAMES):
        values = previews[gesture]
        if frequency:
            centered = values - values.mean(axis=0)
            spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(values))[:, None], axis=0)) / len(values)
            domain = np.fft.rfftfreq(len(values), d=1/hz)
            for k in range(3): ax.plot(domain, spectrum[:, k], color=COLORS[k], label='XYZ'[k], lw=1)
            ax.set_xlim(0, 30); ax.set_xlabel('Hz (nominal sample rate)'); ax.set_ylabel('Windowed |FFT| / N')
        else:
            for k in range(3): ax.plot(np.arange(len(values))/hz, values[:, k], color=COLORS[k], label='XYZ'[k], lw=.9)
            ax.set_xlabel('Seconds (nominal sample rate)'); ax.set_ylabel('Acceleration (g)')
        ax.set_title(name + ' — one training recording'); ax.legend(loc='upper right', ncol=3, fontsize=7)
    return save(fig, output, '06_frequency_examples.png' if frequency else '05_time_examples.png')


def train_statistics(points, metadata, output):
    stats = metadata['normalization']
    normalized = (np.clip(points, stats['clip_low'], stats['clip_high']) - stats['mean']) / stats['std']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for k in range(3):
        axes[0].hist(points[:,k], bins=70, density=True, histtype='step', color=COLORS[k], label='XYZ'[k])
        axes[1].hist(normalized[:,k], bins=70, density=True, histtype='step', color=COLORS[k], label='XYZ'[k])
    axes[0].set(title='Raw sampled training points', xlabel='Acceleration (g)', ylabel='Density')
    axes[1].set(title='After train-only normalization', xlabel='Standardized acceleration', ylabel='Density')
    for ax in axes[:2]: ax.legend()
    corr = np.corrcoef(points.T)
    axes[2].imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    axes[2].set_xticks(range(3), list('XYZ')); axes[2].set_yticks(range(3), list('XYZ'))
    axes[2].set_title('Axis correlation (sampled train)')
    for i in range(3):
        for j in range(3): axes[2].text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center')
    return save(fig, output, '07_train_statistics.png')


def adapter_examples(previews, metadata, output):
    """Initial identity-channel adapter, not learned model output."""
    fig, axes = plt.subplots(2, 8, figsize=(17, 5), constrained_layout=True)
    stats = metadata['normalization']
    for col, gesture in enumerate(CLASSES):
        values = previews[gesture][:metadata['window_size']]
        values = ((np.clip(values, stats['clip_low'], stats['clip_high']) - stats['mean']) / stats['std']).T
        for row, signal in enumerate([values, np.log1p(np.abs(np.fft.fft(values, axis=-1)))]):
            signal = (signal - signal.mean(axis=-1, keepdims=True)) / np.maximum(signal.std(axis=-1, keepdims=True), 1e-6)
            # PyTorch interpolate align_corners=False uses these source centers.
            centers = np.clip((np.arange(64) + .5) * signal.shape[-1] / 64 - .5, 0, signal.shape[-1]-1)
            rgb = np.stack([np.interp(centers, np.arange(len(trace)), trace) for trace in signal])
            rgb = (np.tanh(rgb / 3) + 1) / 2
            image = np.repeat(rgb.T[None, :, :], 64, axis=0)
            axes[row,col].imshow(image); axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row == 0: axes[row,col].set_title(NAMES[col], fontsize=8)
    axes[0,0].set_ylabel('Time'); axes[1,0].set_ylabel('Frequency')
    fig.suptitle('Initial signal-to-RGB adapter (before image normalization; not learned output)')
    return save(fig, output, '08_adapter_examples.png')


def validate_counts(frame, metadata):
    for split, key in [('train', 'train'), ('validation', 'validation'), ('test', 'test')]:
        actual = frame[frame.split == split].groupby('gesture').windows.sum().reindex(CLASSES, fill_value=0)
        expected = metadata[key + '_windows_by_class']
        if any(int(actual[c]) != expected[c] for c in CLASSES):
            raise ValueError(f'{split} counts differ from preprocessing metadata')


def sample_pool(root, frame, metadata, per_class=12, seed=42):
    """One seeded recording per training subject, up to per_class subjects.

    All paired figures use this same pool. Global counts still use the full
    inventory. Selection is independent of model predictions or signal shape.
    """
    rng, pool, manifest = np.random.default_rng(seed), {}, []
    for gesture in CLASSES:
        available = frame[(frame.split == 'train') & (frame.gesture == gesture) & (frame.windows > 0)]
        subjects = np.array(sorted(available.subject.unique()))
        chosen_subjects = rng.choice(subjects, min(per_class, len(subjects)), replace=False)
        pool[gesture] = []
        for subject in chosen_subjects:
            paths = sorted(available.loc[available.subject == subject, 'file'])
            relative = paths[int(rng.integers(len(paths)))]
            raw = pd.read_csv(Path(root)/relative, usecols=['x','y','z'])
            values = raw.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).interpolate(limit_direction='both').dropna().to_numpy(dtype=np.float32)
            if len(values) < metadata['window_size']:
                raise ValueError('Pool file no longer matches inventory: ' + relative)
            pool[gesture].append(dict(file=relative, subject=subject, values=values, gesture=gesture))
            manifest.append(dict(file=relative, subject=subject, gesture=gesture, label_id=CLASSES.index(gesture)))
    return pool, pd.DataFrame(manifest)


def sample_format(root, pool, metadata):
    first = pool[CLASSES[0]][0]
    raw = pd.read_csv(Path(root)/first['file'])
    print('Example file:', first['file'])
    print('Full CSV shape:', raw.shape)
    print('Columns / dtypes:', raw.dtypes.astype(str).to_dict())
    print(raw.head(3).to_string(index=False))
    print('X/Y/Z:', first['values'].shape, first['values'].dtype)
    print('One model window:', first['values'][:metadata['window_size']].T.shape)
    print('Annotation granularity: one file-level label inferred from segmented filename.')
    print('No frame-level onset/offset annotation is inferred or fabricated.')


def label_alignment(pool, metadata, output):
    selected = [1, 3, 5, 7]
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), constrained_layout=True)
    stats = metadata['normalization']
    hz, window = metadata['sampling_rate_hz'], metadata['window_size']
    for row, index in enumerate(selected):
        item = pool[CLASSES[index]][0]
        values = item['values'][:1200]
        domain = np.arange(len(values))/hz
        for k in range(3): axes[row,0].plot(domain, values[:,k], color=COLORS[k], lw=.8, label='XYZ'[k])
        axes[row,0].set_title(f"{NAMES[index]} | {item['subject']}")
        axes[row,0].set_ylabel('Acceleration (g)'); axes[row,0].set_xlabel('Seconds (nominal Hz)')
        axes[row,0].legend(ncol=3, fontsize=7)
        axes[row,1].axvspan(0, domain[-1], color=plt.cm.tab10(index), alpha=.35)
        axes[row,1].text(.5,.5, f'ID {index}: {NAMES[index]}\nFile-level annotation', ha='center', va='center', transform=axes[row,1].transAxes, fontsize=10)
        axes[row,1].set_xlim(0, domain[-1]); axes[row,1].set_ylim(0,1); axes[row,1].set_yticks([])
        axes[row,1].set_xlabel('Seconds (nominal Hz)'); axes[row,1].set_title('Label belongs to the recording')
        normalized = (np.clip(values[:window],stats['clip_low'],stats['clip_high'])-stats['mean'])/stats['std']
        for k in range(3): axes[row,2].plot(np.arange(window),normalized[:,k],color=COLORS[k],lw=.8,label='XYZ'[k])
        axes[row,2].set_title(f'First {window}-sample window | inherited label {index}')
        axes[row,2].set_xlabel('Sample index'); axes[row,2].set_ylabel('Standardized acceleration')
    fig.suptitle('Signal / file-level label / model window — no onset or offset ground truth')
    return save(fig, output, '09_signal_label_alignment.png')


def multisubject_examples(pool, metadata, output):
    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharey='row', constrained_layout=True)
    for row, index in enumerate([1,3,5,7]):
        for col, item in enumerate(pool[CLASSES[index]][:3]):
            values = item['values'][:1200]
            axes[row,col].plot(np.arange(len(values))/metadata['sampling_rate_hz'],np.linalg.norm(values,axis=1),color='#2563eb',lw=.9)
            axes[row,col].set_title(f"{NAMES[index]} | {item['subject']}")
            axes[row,col].set_xlabel('Seconds (nominal Hz)')
            if col==0: axes[row,col].set_ylabel('Acceleration magnitude (g)')
    fig.suptitle('Same gesture, different training subjects (shared y scale per row)')
    return save(fig, output, '10_multi_subject_examples.png')


def gesture_profiles(pool, metadata, output):
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), constrained_layout=True)
    progress = np.linspace(0, 1, 200)
    for ax, gesture, name in zip(axes.flat,CLASSES,NAMES):
        traces=[]
        for item in pool[gesture]:
            magnitude=np.linalg.norm(item['values'],axis=1)
            traces.append(np.interp(progress,np.linspace(0,1,len(magnitude)),magnitude))
        traces=np.stack(traces); mean,std=traces.mean(axis=0),traces.std(axis=0)
        ax.fill_between(progress*100,mean-std,mean+std,alpha=.2,color='#2563eb',label='±1 SD across recordings')
        ax.plot(progress*100,mean,color='#2563eb',lw=1.5,label='Mean')
        ax.set_title(f'{name} | n={len(traces)} recordings from distinct subjects')
        ax.set_xlabel('Relative progress in recording (%)'); ax.set_ylabel('Magnitude (g)')
        ax.legend(fontsize=7)
    fig.suptitle('Gesture profiles — resampled only for display, not onset-aligned or model input')
    return save(fig, output, '11_gesture_profiles.png')


def pool_features(pool, metadata):
    """One fixed-length normalized window per selected training recording."""
    stats=metadata['normalization']; windows=[]; labels=[]
    for gesture in CLASSES:
        for item in pool[gesture]:
            values=item['values'][:metadata['window_size']]
            values=(np.clip(values,stats['clip_low'],stats['clip_high'])-stats['mean'])/stats['std']
            windows.append(values); labels.append(gesture)
    windows=np.stack(windows)
    features=np.concatenate([windows.mean(axis=1),windows.std(axis=1),
                            np.sqrt(np.mean(windows**2,axis=1)),
                            np.quantile(windows,.95,axis=1)-np.quantile(windows,.05,axis=1)],axis=1)
    return windows,features,np.array(labels)


def pca_features(pool, metadata, output):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    _,features,labels=pool_features(pool,metadata)
    pca=PCA(n_components=2)
    embedding=pca.fit_transform(StandardScaler().fit_transform(features))
    fig,ax=plt.subplots(figsize=(10,7),constrained_layout=True)
    for index,gesture in enumerate(CLASSES):
        selected=labels==gesture
        ax.scatter(embedding[selected,0],embedding[selected,1],label=NAMES[index],color=plt.cm.tab10(index),s=35,alpha=.7)
    ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
           title='PCA of 12 statistical features — selected training windows only')
    ax.legend(fontsize=8)
    return save(fig,output,'12_pca_features.png')


def channel_rms(pool, metadata, output):
    windows,_,labels=pool_features(pool,metadata)
    rms=np.sqrt(np.mean(windows**2,axis=1))
    matrix=np.stack([rms[labels==gesture].mean(axis=0) for gesture in CLASSES])
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
    image=ax.imshow(matrix,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(3),list('XYZ'));ax.set_yticks(range(8),NAMES)
    ax.set_title('Mean channel RMS by gesture — selected normalized training windows')
    for i in range(8):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center')
    fig.colorbar(image,ax=ax,label='RMS (standardized units, not physical energy)')
    return save(fig,output,'13_channel_rms.png')


def time_frequency(pool, metadata, output):
    fig,axes=plt.subplots(4,2,figsize=(14,12),constrained_layout=True)
    for ax,gesture,name in zip(axes.flat,CLASSES,NAMES):
        item=pool[gesture][0];values=item['values'][:1200,0]
        ax.specgram(values-values.mean(),NFFT=64,Fs=1.0,noverlap=48,cmap='magma')
        ax.set(title=f"{name} | {item['subject']} | Acceleration X",xlabel='Sample index',ylabel='Frequency (cycles/sample)')
    fig.suptitle('STFT visualization only — not the frequency image used by the deployed model')
    return save(fig,output,'14_time_frequency.png')


def duplicate_audit(frame, metadata, output):
    duplicates=frame[frame.duplicated('recording_hash',keep=False)]
    across=duplicates.groupby('recording_hash').split.nunique()
    report={'scope':'Exact cleaned float32 X/Y/Z recording arrays; not window-level or approximate duplicates',
            'recordings':len(frame),'constant_recordings':int(frame.constant_recording.sum()),
            'duplicate_recordings_after_first':int(frame.duplicated('recording_hash').sum()),
            'unique_hashes_shared_across_subject_splits':int((across>1).sum()),
            'note':'Matching constant signals may occur naturally; inspect provenance before interpreting leakage.'}
    Path(output).mkdir(parents=True,exist_ok=True)
    Path(output,'duplicate_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    duplicates.to_csv(Path(output,'duplicate_recordings.csv'),index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data/external/InGesture'))
    parser.add_argument('--metadata', type=Path, default=Path('datasets/InGestureOffline/metadata.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/data_visualization'))
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    root = find_root(args.input_dir)
    frame, previews, points = inventory(root, metadata)
    validate_counts(frame, metadata)
    pool, manifest = sample_pool(root, frame, metadata)
    previews = {name: items[0]['values'][:1200] for name,items in pool.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / 'recording_inventory.csv', index=False)
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for fn in [class_distribution, split_distribution, subject_coverage, quality]: print(fn(frame, metadata, args.output_dir))
    print(signal_examples(previews, metadata, args.output_dir))
    print(signal_examples(previews, metadata, args.output_dir, True))
    print(train_statistics(points, metadata, args.output_dir))
    print(adapter_examples(previews, metadata, args.output_dir))
    manifest.to_csv(args.output_dir/'sample_pool.csv', index=False)
    for fn in [label_alignment, multisubject_examples, gesture_profiles]: print(fn(pool, metadata, args.output_dir))
    for fn in [pca_features,channel_rms,time_frequency]:print(fn(pool,metadata,args.output_dir))
    print(json.dumps(duplicate_audit(frame,metadata,args.output_dir),indent=2))
    print(f'Validated {len(frame)} recordings and {int(frame.windows.sum())} windows.')


if __name__ == '__main__':
    main()
