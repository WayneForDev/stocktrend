import pandas as pd
import itertools
import numpy as np
import stockstats as sts
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

pd.options.display.float_format = '${:,.2f}'.format

def readCSV(file):
    ds = pd.read_csv(file)
    df = pd.DataFrame(ds)
    return df

def generateStockDf(df):
    dfStock = sts.StockDataFrame.retype(df)
    dfStock['macd']
    dfStock['rsi_6']
    dfStock['rsi_12']
    dfStock['kdjk']
    dfStock['kdjd']
    dfStock['kdjj']
    dfStock['close_5_sma']
    dfStock['close_10_sma']
    dfStock['close_20_sma']
    dfStock['close_40_sma']
    dfStock['close_60_sma']
    dfStock['close_80_sma']
    dfStock['close_100_sma']
    dfStock['vr']
    dfStock['wr_6']
    dfStock['wr_10']
    dfStock['cci']
    return dfStock

def getLabels(df): # get labels for one day stock trend
    labels = []
    df_ = df['change']
    for change in df_:
        if change <= 0:
            labels.append('Down')
        else:
            labels.append('Up')
    return labels

def getLabelsWithInput(df, period):
    labels = []
    df_ = df['adj close']
    # Checking if the period is already set
    if period == 0:
        j = int(input('Please choose the time period in term of days of the trend you want to forcast (example: 5): '))
    else:
        j = period

    for i in range(len(df_)-j):
        change = df_.iloc[i+j] - df_.iloc[i]
        if change <= 0:
            labels.append('Down')
        else:
            labels.append('Up')
    return labels, j

def runPCA(df):
    standardized_df = preprocessing.StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(standardized_df)
    principalDf = pd.DataFrame(data = principalComponents)
    return principalDf

def runSVC(df,labels):
    clf = SVC()
    clf.fit(df,labels)
    return clf

def calculateEigen(a):
    eigValue, eigVectors = np.linalg.eig(a)
    for x in range(len(eigValue)):
        # print("Eigen Values: ", eigValue[x])
        # print("Eigen Vectors", eigVectors[x])
        return eigValue, eigVectors

######################################### Algorithms #############################################

def runRF(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(max_depth=5,n_estimators=30)
    clf.fit(X_train,y_train)
    return accuracy_score(y_test,clf.predict(X_test)),clf

def runKNN(X_train, y_train, X_test, y_test):
    bestk = 0
    bestclf = []
    bestacc = 0
    for n in range(5,8):
        clf=KNeighborsClassifier(n_neighbors=n)
        clf.fit(X_train,y_train)
        # y_predict = clf.predict(x_test)
        # accuracy = clf.score(X_test, y_test)
        accuracy = accuracy_score(y_test,clf.predict(X_test))
        if(accuracy > bestacc):
            bestk = n
            bestclf = clf
            bestacc = accuracy
    
    #print("bestk ",bestk, "bestclf ",bestclf,"bestacc ",bestacc)
    return bestacc, bestclf

def runNB(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    # y_predict = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    return accuracy,clf

def runSVM(X_train, y_train, X_test, y_test):
    clf = runSVC(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    # report = classification_report(y_test,y_predict,target_names = ['Down','Up'])
    return accuracy, clf

######################################### Call Algorithm ###################################
def callAlg(algorithm, X_train, y_train, X_test, y_test):
    if algorithm == 'Naive-Bayes':
        acc,clf = runNB(X_train, y_train, X_test, y_test)
    elif algorithm == 'Random Forest':
        acc,clf = runRF(X_train, y_train, X_test, y_test)
    elif algorithm == 'KNN':
        acc,clf = runKNN(X_train, y_train, X_test, y_test)
    else:
        acc,clf = runSVM(X_train, y_train, X_test, y_test)
    return acc, clf

########################################### Main ################################################

def main():
    
    fileNames = ['AAPL_DAILY','NVDA_DAILY','SNAP_DAILY','AMZN_DAILY']
    algorithms = ['Naive-Bayes','Random Forest','KNN','SVM']
    period = 0
    accuracyReport = []
    feature_used = []
    for i in range(4):
        if algorithms[i] == 'SVM':
            pcaReduce = True
        else:
            pcaReduce = False
        accuracy = []
        feature_list = []
        for k in range(4):
            feature_list.append('')
        for ind in range(4):
            name = fileNames[ind]
            fileName = (name+'.csv')
            df = readCSV(fileName)
            df = generateStockDf(df)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(axis = 0, how = 'any')
            # df.to_csv('stockdf.csv')
            # labels = getLabels(df)
            labels, period = getLabelsWithInput(df,period)

            #Initial feature drop
            df = df.drop(['open','high','low','close','adj close','middle','change','middle_14_sma', 'close_60_sma', 'close_80_sma', 'close_100_sma'], axis = 1)
            
            feature_names = df.columns
            print(feature_names)

            # print(getLabelsWithInput(df))
            # if name == 'SNAP_MONTHLY':
            #     p = 10
            # else:
            #     p = 21
            p = 21
            y_train = labels[6-period:-p]
            df = df.drop(df.index[0:5])


            if pcaReduce == True:
                reducedDf = runPCA(df)
                X_train = reducedDf[:-p-1]
                X_test = reducedDf[-p:-1]
                y_test = labels[-p+1:]
                acc,clf = callAlg(algorithms[i],X_train, y_train, X_test, y_test)
                print(algorithms[i], 'feature selected: PCA')
                print(fileName,': \nTraining set size: ', len(X_train), '\nTesting set size: ', (p-1))
                accuracy.append(acc)
                feature_list[ind] = ['PCA']
                print(name, acc)
            else:
                acc = -1
                feature_selected = []
                feature_number = [2]
                #feature_number = [1, 2]
                for j in feature_number:
                    for sub_feature in itertools.combinations(feature_names, j):
                        list_drop = []
                        for features in sub_feature:
                            list_drop.append(str(features))
                        #list_drop = ['macd']
                        df_drop = pd.DataFrame(df,columns = list_drop)
                        X_train = df_drop[:-p-1]
                        X_test = df_drop[-p:-1]
                        y_test = labels[-p+1:]
                        acc_temp,clf = callAlg(algorithms[i],X_train, y_train, X_test, y_test)
                        if acc_temp > acc:
                            acc = acc_temp
                            feature_selected = list_drop
                print(algorithms[i], 'feature selected: ', feature_selected)
                print(fileName,': \nTraining set size: ', len(X_train), '\nTesting set size: ', (p-1))
                accuracy.append(acc)
                feature_list[ind] = feature_selected
                print(name, acc)
        accuracyReport.append(accuracy)
        feature_used.append(feature_list)
    for accuracy in accuracyReport:
        print(accuracy)
    for i in range(4):
        print(algorithms[i])
        print(fileNames)
        print(accuracyReport[i])
        print(feature_used[i])


main()
