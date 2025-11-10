import pandas as pd
import argparse
import os


# Init args
parser = argparse.ArgumentParser()
parser.add_argument('-i','--path_input', required=True, nargs='+')
parser.add_argument('-o','--path_out', required=True)
args = vars(parser.parse_args())

df_paths = args['path_input']
path_out = args['path_out']

if len(df_paths) == 0:
    print(f"Warning: No files found")
    # Create empty dataframe
    df = pd.DataFrame(columns=['name', 'tp', 'fp', 'fn'])
else:
    # Read and concatenate all files
    dfs = []
    for file in df_paths:
        if os.path.exists(file):
            tmp = pd.read_csv(file)
            dfs.append(tmp)
        else:
            print(f"Warning: File not found: {file}")
    
    if len(dfs) > 0:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame(columns=['name', 'tp', 'fp', 'fn'])

# Write
os.makedirs(os.path.dirname(path_out), exist_ok=True)
df.to_csv(path_out, index=False)
print(f"Aggregated {len(df)} confusion matrices to {path_out}")
