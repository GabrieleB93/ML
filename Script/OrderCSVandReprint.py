import pandas as pd
from config import *

data_path = FIRST_GRID_ETR


def main():
    data = pd.read_csv(data_path, sep=',', index_col=False, header='infer')
    saveOnCSV(data, data_path)


def saveOnCSV(results_records, nameResult):
    results = results_records
    results = results.sort_values('mee', ascending=True)
    filepath = "../DATA/" + nameResult
    file = open(filepath, mode='w')
    results.to_csv(file, sep=',', header=True, index=False)
    file.close()


if __name__ == '__main__':
    main()
