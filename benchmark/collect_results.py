from glob import glob

import pandas as pd
from tqdm.auto import tqdm

from config import RESULTS_PATH


def main():
    for path_with_task in tqdm(glob(RESULTS_PATH + "*"), desc="TASKs", position=0, leave=False):
        task = path_with_task.split("/")[-1].split("\\")[-1]
        dfs = []

        for path_with_model in tqdm(glob(path_with_task + "/*"), desc="MODELs", position=1, leave=False):
            model = path_with_model.split("/")[-1].split("\\")[-1]

            for path_with_dataset in tqdm(glob(path_with_model + "/*"), desc="DATASETs", position=2, leave=False):
                dataset = path_with_dataset.split("/")[-1].split("\\")[-1]

                for result_path in tqdm(glob(path_with_dataset + "/*"), desc="EXPERIMENTs", position=3, leave=False):
                    df = pd.read_csv(result_path)
                    df["task"] = task
                    df["model"] = model
                    df["dataset"] = dataset

                    dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(path_with_task + "/results.csv", index=False)

if __name__ == '__main__':
    main()