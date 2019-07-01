from config import *
from utils import *


def main():
    XTS1,YTS1 = getTrainData(Monk1TS, '2:8', '1:2', ' ')
    XTR1,YTR1 = getTrainData(Monk1TR, '2:8', '1:2', ' ')
    XTS2,YTS2 = getTrainData(Monk2TS, '2:8', '1:2', ' ')
    XTR2,YTR2 = getTrainData(Monk2TR, '2:8', '1:2', ' ')
    XTS3,YTS3 = getTrainData(Monk3TS, '2:8', '1:2', ' ')
    XTR3,YTR3 = getTrainData(Monk3TR, '2:8', '1:2', ' ')



if __name__ == '__main__':
    main()