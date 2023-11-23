# %%
from pathlib import Path
import pandas as pd
import numpy as np
import re
# %%
# load label
label_paty = "/home/siyuan/research/PoseFall/data/all_data_with_clapper_and_annotation.csv"
label_paty = Path(label_paty)

if not label_paty.is_file():
    raise FileNotFoundError("Label file not found.")

df = pd.read_csv(label_paty)
df.head()
# %%
impact_df = df["Impact"]
# use regex to extract the impact start and end time
impact_df = impact_df.str.extract(r"[\s\S]*(\d{4,}),[\s\S]*(\d{4,})[\s\S]*'point of impact: ([a-zA-Z0-9_ ]*)[\s\S]*force: (\d)[\s\S]*attribute: (.+)[\s\S]*'")
# print out the rows that have nan values
print(impact_df[impact_df.isna().any(axis=1)])
# columns name
col_name = ["impact_start", "impact_end", "impact_loc", "impact_force", "impact_attribute"]
impact_df.columns = col_name
impact_df.head()

# map impact_loc into 4 categories: head, torso, arms, legs
impact_df["impact_loc"] = impact_df["impact_loc"].map({"chest":"torso",
                                                       "guts":"torso",
                                                       "top of head":"head",
                                                       "cheek":"head",
                                                       "right shoulder":"arms",
                                                       "right hand":"arms",
                                                       "right side of chest":"torso",
                                                       "knees"  :"legs",
                                                       "back of knee":"legs",
                                                       "chin":"head",
                                                       "right foot":"legs",
                                                       "hip":"torso",
                                                       "side of face":"head",
                                                       "right arm":"arms",
                                                       "upper back":"torso",
                                                       "bottoms of feet":"legs",
                                                       "kneex":"legs",
                                                       "back of neck":"head",
                                                       "back of head":"head",
                                                       "tops of shoulders":"arms",
                                                       "palms of hands and shoulders":"arms",
                                                       "left hand and cheek":"arms",
                                                       "left hand":"arms",
                                                       "armpits":"arms",
                                                       "neck":"head",
                                                       "left hand": "arms",
                                                       "gut":"torso",
                                                       "hip":"torso",    
                                                       })

# %%
glitch_df = df["Glitch"]
print(glitch_df.head())
print("-"*10)
glitch_time = glitch_df.str.extract(r"\[(\d{4,}), (\d{4,})[\s\S]*'")
glitch_time.columns = ["glitch_start", "glitch_end"]
# check if there is any nan value
nan_idx = glitch_time[glitch_time.isna().any(axis=1)].index
print(f'time Nan rows: \n{glitch_df[nan_idx]}')
print("-"*10)
glitch_att = glitch_df.str.extract(r"[\s\S]*attribute: (\w+);?[\s\S]*'")
glitch_att.columns = ["glitch_attribute"]
# check if there is any nan value
nan_idx = glitch_att[glitch_att.isna().any(axis=1)].index
print(f'attribute Nan rows: \n{glitch_df[nan_idx]}')
print("-"*10)
glitch_spread = glitch_df.str.extract(r"[\s\S]*spread: (\w+);?[\s\S]*'")
glitch_spread.columns = ["glitch_spread"]
# check if there is any nan value
nan_idx = glitch_spread[glitch_spread.isna().any(axis=1)].index
print(f'attribute Nan rows: \n{glitch_df[nan_idx]}')
print("-"*10)

# combine them together
glitch_df = pd.concat([glitch_time, glitch_att, glitch_spread], axis=1)
glitch_df.head()
# %%
fall_df = df["Fall"]
print(fall_df.head())
print("-"*10)
fall_time = fall_df.str.extract(r"\[(\d{4,}), (\d{4,})[\s\S]*'")
fall_time.columns = ["fall_start", "fall_end"]
# check if there is any nan value
nan_idx = fall_time[fall_time.isna().any(axis=1)].index
print(f'time Nan rows: \n{fall_time[nan_idx]}')
# %%
fall_att = fall_df.str.extract(r"[\s\S]*attribute: ([a-zA-Z0-9_ ]+);?[\s\S]*'")
fall_att.columns = ["fall_attribute"]
# check if there is any nan value
nan_idx = fall_att[fall_att.isna().any(axis=1)].index
print(f'attribute Nan rows: \n{fall_df[nan_idx]}')

fall_df = pd.concat([fall_time, fall_att], axis=1)
# %%
print(df.head())
all_df = pd.concat([df.iloc[:, 1:8], impact_df, glitch_df, fall_df], axis=1)
# %%
all_df.head()
# %%
attributes = ["impact_loc", "impact_force", "impact_attribute", "glitch_attribute", "glitch_spread", "fall_attribute"]
# for each attribute, print out the unique values and their counts
for att in attributes:
    counts = all_df[att].value_counts()
    all_counts = [f'{str(key)}: {count}' for key, count in zip(counts.keys(), counts.values)]
    print(f'{att}: \n{all_counts}')
    print("-"*10)
# %%P

# create a bar plot for each attribute. Show count for each value

import matplotlib.pyplot as plt
import seaborn as sns
for att in attributes:
    counts = all_df[att].value_counts()
    sns.barplot(x=counts.keys(), y=counts.values)
    plt.title(att)
    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.show()
# %%
