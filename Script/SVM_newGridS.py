from os.path import exists
import pandas as pd
from utils import *

data_path_POLY = "../DATA/grid_search_result_SVR_POLY"
data_path_RBF = "../DATA/grid_search_result_SVR_RBF"


def main():
    if exists(data_path_POLY and data_path_RBF):
        data_RBF = pd.read_csv(data_path_RBF, sep=',', index_col=False)
        data_POLY = pd.read_csv(data_path_POLY, sep=',', index_col=False)

        # df_corr = (data.corr())
        #
        #
        # plt.figure(figsize=(15, 10))
        # seaborn.heatmap(df_corr)
        # seaborn.set(font_scale=2)
        # plt.title('Heatmap correlation')
        # plt.show()

        sorted = data_RBF.sort_values('mee')

        best_row_POLY = data_POLY[data_POLY.mee == data_POLY.mee.min()]
        best_row_RBF = data_RBF[data_RBF.mee == data_RBF.mee.min()]

        C_RBF_2 = sorted[sorted['C'] != float(best_row_RBF['C'])].iloc[0]['C']
        gamma_RBF_2 = sorted[sorted['gamma'] != float(best_row_RBF['gamma'])].iloc[0]['gamma']
        C_POLY_2 = sorted[sorted['C'] != float(best_row_POLY['C'])].iloc[0]['C']

        print(C_RBF_2)

        C_RBF = best_row_RBF.iloc[0]['C']
        C_POLY = best_row_POLY.iloc[0]['C']
        epsilon_RBF = best_row_RBF.iloc[0]['epsilon']
        epsilon_POLY = best_row_POLY.iloc[0]['epsilon']
        degree = best_row_POLY.iloc[0]['degree']
        gamma_RBF = best_row_RBF.iloc[0]['gamma']

        print(C_RBF)
        print(C_POLY)
        print(epsilon_RBF)
        print(epsilon_POLY)
        print(degree)
        print(gamma_RBF)

        X, Y = getTrainData()

if __name__ == '__main__':
    main()