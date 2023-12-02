# %%
"""
This file contains functions for processing labels.
"""
from pathlib import Path
import pandas as pd
label_path = Path("/home/siyuan/research/PoseFall/data/MoCap/Mocap_data/MoCap_label.csv")
output_path = Path("/home/siyuan/research/PoseFall/data/MoCap/Mocap_processed_data")
if not output_path.exists():
    raise ValueError(f"Output path {output_path} does not exist.")
if not label_path.exists():
    raise ValueError(f"Label path {label_path} does not exist.")
# %%
# read from csv
df = pd.read_csv(label_path)
print(df.columns)
# convert categorical labels to one-hot labels
label_dummies = pd.get_dummies(df[['Impact Location', 'Impact Attribute', 'Impact Force',
       'Glitch Attribute', 'Glitch Speed', 'Fall Attribute', 'End Postion']]).astype('int')
# Merge trial number
label_dummies['Trial Number'] = df['Trial Number']
# move trial number to the first column
cols = list(label_dummies)
cols.insert(0, cols.pop(cols.index('Trial Number')))
label_dummies = label_dummies.loc[:, cols]
print(label_dummies.head())
# save to csv
label_dummies.to_csv(output_path / 'label.csv', index=False)
