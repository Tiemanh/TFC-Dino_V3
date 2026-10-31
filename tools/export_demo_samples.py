"""Export real labelled train examples for functional demo, not accuracy testing."""
import json
from pathlib import Path
import zipfile
import pandas as pd
from visualize_ingesture import CLASSES

ROOT=Path(__file__).resolve().parents[1]


def main():
    pool=pd.read_csv(ROOT/'reports/data_visualization/sample_pool.csv')
    meta=json.loads((ROOT/'datasets/InGestureOffline/metadata.json').read_text())
    output=ROOT/'sample_data/examples';output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for gesture in CLASSES:
        item=pool[pool.gesture==gesture].iloc[0]
        values=pd.read_csv(ROOT/'data/external/InGesture'/item.file,usecols=['x','y','z']).head(meta['window_size'])
        if len(values)!=meta['window_size'] or values.isna().any().any():
            raise ValueError('Demo source must contain a complete finite window')
        values.insert(0,'timestamp',[i/meta['sampling_rate_hz'] for i in range(len(values))])
        destination=output/(gesture+'.csv');values.to_csv(destination,index=False)
        rows.append(dict(file='examples/'+destination.name,source_file=item.file,source_split='train',
                         label=gesture,label_id=CLASSES.index(gesture),samples=len(values),
                         timestamp='generated_seconds_at_nominal_200Hz',purpose='functional_demo_not_accuracy_evaluation'))
    pd.DataFrame(rows).to_csv(ROOT/'sample_data/example_manifest.csv',index=False)
    with zipfile.ZipFile(ROOT/'sample_data/gesture_examples.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in output.glob('*.csv'):archive.write(path,'examples/'+path.name)
        archive.write(ROOT/'sample_data/example_manifest.csv','example_manifest.csv')
    print('Exported 8 real train examples; filenames provide labels, not guaranteed predictions.')


if __name__=='__main__':main()
