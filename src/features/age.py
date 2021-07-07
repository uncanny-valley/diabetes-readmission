import pandas as pd
import re

def age_to_index(age: pd.Series) -> pd.Series:
    age_mappings = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    age_mappings = dict(zip(age_mappings, range(0, len(age_mappings))))
    return age.map(age_mappings)

def age_to_midpoints(age: str) -> int:
    m = re.search(r'[(,[](\d+)-(\d+)[],)]', age)
    x = int(m[1])
    y = int(m[2])
    return (x + y) / 2