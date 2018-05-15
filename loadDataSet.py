import numpy as np 
import matplotlib.pyplot as plt

'''
读取数据集
'''
def loadDataSet(filename):
    dataMat=[];lableMat=[]
    fr=open(filename,'r',encoding="utf-8")
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        lableMat.append(float(lineArr[2]))
    return dataMat,lableMat
'''
数据可视化
'''
def showDataSet(dataMat,labelMat):
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):                   #正负样本分开
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
    plt.show()

if __name__=='__main__':
    dataMat,labelMat=loadDataSet(r'C:\Users\user\Desktop\Machine-Learning-master\SVM\testSet.txt')
    showDataSet(dataMat,labelMat)