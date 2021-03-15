import pandas as pd
import matplotlib.pyplot as plt
import os
import statistics
import numpy as np

'''
1. Collect required measurements (final, back, edge)
2. Reformat as time taken for each iteration (2nd-1st etc)
'''
# Data Variables
FINAL = 'final'
BACK = 'back'
EDGE = 'edge'

# Static Vars
TIMES = 'times'


# METHODS

# Returns paths of files to be collected
def collectFiles(cur):
    if cur == FINAL:
        path = 'time_measurements_final/'
    elif cur == BACK:
        path = 'time_measurements_background/'
    elif cur == EDGE:
        path = 'time_measurements_edge/'
    fs = os.listdir(path)
    return [path + f for f in fs if '.csv' in f]


# Collects dataframes from given files
def collectData(files):
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    return dfs


# Reformat data as time taken between measurement
def reformat(dfs):
    new_dfs = []
    for df in dfs:
        new_times = []
        i = 1

        while i < df.shape[0]:
            diff = df.iloc[i][TIMES] - df.iloc[i-1][TIMES]
            new_times.append(diff)
            i += 1

        new_df = pd.DataFrame()
        new_df[TIMES] = new_times
        new_dfs.append(new_df)

    return new_dfs


# Plot
def custom_plot(dfs, min, max):
    xs = []
    for i in range(1, dfs[0].shape[0]):
        cur = [d[TIMES][i] for d in dfs]
        xs.append(cur)

    means = np.array([statistics.mean(x) for x in xs])
    stds = np.array([statistics.stdev(x) for x in xs])

    plt.errorbar(range(len(means)), means, yerr=stds, capsize=4)
    #plt.errorbar(range(len(means)), means, yerr=stds)
    #plt.plot(range(len(means)), means)
    plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)
    plt.title('BS with Feature Tracking: Errobar of the processing time per frame')
    plt.xlabel('Frame number')
    plt.ylabel('Time taken (s)')
    plt.show()

# MAIN
def main():

    cur = FINAL  # SET WHICH DATASET WE NEED

    # Collect data
    files = collectFiles(cur)
    print("Collected Files:")
    print(files)
    dfs_raw = collectData(files)
    print("Collected Raw Data")

    # Format data
    dfs = reformat(dfs_raw)
    print("Reformatted data")

    # Plot 1st as test
    #df_plot = dfs[0]
    custom_plot(dfs, 1, 30)
    # for i in range(030):
    #     plt.plot(range(dfs[i].shape[0]), dfs[i][TIMES], label=str(i))
    #
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
