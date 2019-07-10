# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/25 19:42
 @Author  : hanzi5
 @Email   : hanzi5@yeah.net
 @File    : SVM.py
 @Software: PyCharm
"""
import numpy as np
#import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_tnc
#from scipy.optimize import fmin_bfgs 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle

matplotlib.rcParams['font.family']='SimHei'  # 用来正常显示中文
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

# 定义目标函数，w是未知数据，args是已知数，求w的最优解
def func(w,*args):
    X,Y,c=args
    yp=np.dot(X,w)  # w*x，注意已经将b作为x的第一列进行了计算，w*x就是我们现在预测的y
    idx=np.where(yp*Y<1)[0] # 找到分错的数据索引位置
    e=yp[idx]-Y[idx]        # y预测值-y真实值，误差
    cost=np.dot(e,e)+c*np.dot(w,w)  # 平方和损失，c：学习率，加w的二范式惩罚
    grand=2*(np.dot(X[idx].T,e)+c*w)# 梯度下降？？
    return cost,grand

def plotResult(w):
    margin=2/np.sqrt(np.dot(w[1:3],w[1:3]))
    plot_x=np.append(np.min(x,0)[0]-0.2,np.max(x,0)[0]+0.2)
    plot_y=-(plot_x*w[1]+w[0])/w[2]
    plt.figure()
    pos=(Y==1) # 正类
    neg=(y==-1) # 负类
    plt.plot(x[pos][:,0],x[pos][:,1],"r+",label="正类")
    plt.plot(x[neg][:,0],x[neg][:,1],"bo",label="负类")
    plt.plot(plot_x,plot_y,"r-",label="分割超平面")
    plt.plot(plot_x,plot_y+margin/2,"g-.",label="")
    plt.plot(plot_x,plot_y-margin/2,"g-.",label="")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('SVM Demo')
    plt.legend()
    plt.show()

# 简易smo算法开始####################################################################
# 随机选择第2个alpha
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j

# 调整大于H或小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

# 公共函数，根据公式求w，简易smo算法及完整smo算法通用
def calcWs(alphas,dataArr,labelArr):
    w=sum( np.array(alphas) * np.array(labelArr.reshape((-1,1))) * np.array(np.array(dataArr))  )
    return w


# 输入变量：x、y、c：常数c、toler：容错率、maxIter：最大循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    #dataMatIn, classLabels, C, toler, maxIter=dataArr,lableArr,0.6,0.001,40
    dataMatrix = np.mat(dataMatIn)             # 数据x转换为matrix类型
    labelMat = np.mat(classLabels).transpose() # 标签y转换为matrix类型，转换为一列
    b = 0                                      # 截距b
    m,n = np.shape(dataMatrix)                 # 数据x行数、列数
    alphas = np.mat(np.zeros((m,1)))           # 初始化alpha，有多少行数据就产生多少个alpha
    iter = 0                                   # 遍历计数器
    while (iter < maxIter):
        #print( "iteration number: %d" % iter)
        alphaPairsChanged = 0                  # 记录alpha是否已被优化，每次循环都重置
        for i in range(m):                     # 按行遍历数据，类似随机梯度下降
            # i=0
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b # 预测值y，g(x)函数，《统计学习方法》李航P127，7.104
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions  # 误差，Ei函数，P127，7.105
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 找第一个alphas[i]，找到第一个满足判断条件的，判断负间隔or正间隔，并且保证0<alphas<C
                j = selectJrand(i,m)            # 随机找到第二个alphas[j]
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b # 计算预测值
                Ej = fXj - float(labelMat[j])   # 计算alphas[j]误差
                alphaIold = alphas[i].copy()    # 记录上一次alphas[i]值
                alphaJold = alphas[j].copy()    # 记录上一次alphas[j]值
                if (labelMat[i] != labelMat[j]):# 计算H及L值，《统计学习方法》李航，P126
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: 
                    #print( "L==H")
                    continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                # 《统计学习方法》李航P127，7.107，这里的eta与李航的一致，这里乘了负号
                if eta >= 0: 
                    #print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta     # 《统计学习方法》李航P127，7.107，更新alphas[j]
                alphas[j] = clipAlpha(alphas[j],H,L)       # alphas[j]调整大于H或小于L的alpha值
                if (abs(alphas[j] - alphaJold) < 0.00001): # 调整后过小，则不更新alphas[i]
                    #print( "j not moving enough")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j]) #更新alphas[i]，《统计学习方法》李航P127，7.109
                # 更新b值，《统计学习方法》李航P130，7.115，7.116
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): # 判断符合条件的b
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else: 
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                #print( "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
    return b,alphas

# 画图
def plot_smoSimple(dataArrWithAlpha,b,w):
    type1_x1 = []
    type1_x2 = []
    type2_x1 = []
    type2_x2 = []
    dataSet=dataArrWithAlpha 
    # 取两类x1及x2值画图
    type1_x1=dataSet[dataSet[:,-2]==-1][:,:-2][:,0].tolist() 
    type1_x2=dataSet[dataSet[:,-2]==-1][:,:-2][:,1].tolist()
    type2_x1=dataSet[dataSet[:,-2]==1][:,:-2][:,0].tolist()
    type2_x2=dataSet[dataSet[:,-2]==1][:,:-2][:,1].tolist()
    
    # 画点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(type1_x1,type1_x2, marker='s', s=90)
    ax.scatter(type2_x1,type2_x2, marker='o', s=50, c='red')
    plt.title('Support Vectors Circled')
    
    # 获取支持向量值，画椭圆
    dataVectors=dataArrWithAlpha[dataArrWithAlpha[:,-1]>0]
    for d in dataVectors:
        circle = Circle(d[0:2], 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)
        
    # 画分割超平面
    b=b1.getA()[0][0] # 获得传入的b
    w0= w[0]#0.8065
    w1= w[1]#-0.2761
    x = np.arange(-2.0, 12.0, 0.1)
    y = (-w0*x - b)/w1
    ax.plot(x,y)
    ax.axis([-2,12,-8,6])
    plt.show()
# 简易smo算法结束####################################################################

# 完整smo算法开始####################################################################
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2))) #first column is valid flag
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

# 核函数
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

# 计算误差
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 寻找第2个步长最大的alphas[j]
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 计算误差存入缓存中
def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

# 内循环，寻找第2个步长最大的alphas[j]
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            #print ("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: 
            #print( "eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): 
            #print( "j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

# SMO主函数
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                #print( "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                #print( "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        #print( "iteration number: %d" % iter)
    return oS.b,oS.alphas

# 测试Rbf数据
def testRbf(dataArrTrain,labelArrTrain,dataArrTest,labelArrTest,k1=1.3):
    b,alphas = smoP(dataArrTrain, labelArrTrain, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=np.mat(dataArrTrain); labelArrTrain = np.mat(labelArrTrain).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelArrTrain[svInd];
    #print( "there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArrTrain[i]): errorCount += 1
    print( "the training error rate is: %f" % (float(errorCount)/m))
    errorCount = 0
    datMat=np.mat(dataArrTest)
    #labelMat = np.mat(labelArrTest).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArrTest[i]): errorCount += 1    
    print( "the test error rate is: %f" % (float(errorCount)/m)   )
    return b,alphas

# 画图，rbf核函数数据
def plot_smoCompletion ():
    xcord0 = []; ycord0 = []; xcord1 = []; ycord1 = []
    fw = open('./testSetRBF2.txt', 'w')#generate data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xcord0 = []; ycord0 = []; xcord1 = []; ycord1 = []
    for i in range(100):
        [x,y] = np.random.uniform(0,1,2)
        xpt=x*np.cos(2.0*np.pi*y); ypt = x*np.sin(2.0*np.pi*y)
        if (x > 0.5):
            xcord0.append(xpt); ycord0.append(ypt)
            label = -1.0
        else:
            xcord1.append(xpt); ycord1.append(ypt)
            label = 1.0
        fw.write('%f\t%f\t%f\n' % (xpt, ypt, label))
    ax.scatter(xcord0,ycord0, marker='s', s=90)
    ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')
    plt.title('Non-linearly Separable Data for Kernel Method')
    plt.show()
    fw.close()
# 完整smo算法结束####################################################################

if __name__ == '__main__':
    ##1、SVM直接求参数值#############################################################
    print('\n1、SVM直接求参数值，开始')
    # 生成数据，《统计学习方法》李航，P103，例7.1
    dataSet=np.array([[3,3,1],[4,3,1],[1,1,-1]]) # ,[0,0,-1],[0,1,-1]
    m, n = dataSet.shape
    x=dataSet[:,:-1]
    y=dataSet[:,-1] #.reshape((-1,1))
    # 数据定义
    X=np.append(np.ones([x.shape[0],1]),x,1) # x新增一列全1值，作为截距b
    Y=y
    c=0.001 # 学习率
    w=np.zeros(X.shape[1]) # 初始化一组w系数，全0，也可随机产生：np.random.rand(X.shape[1])
    # bfgs_b方法求最优化问题
    REF=fmin_l_bfgs_b(func,x0=w,args=(X,Y,c),approx_grad=False) #x0=np.random.rand(X.shape[1]) [0,0,0]
    # 采用scipy.optimize其他包夜可以求得
    REF2=fmin_tnc(func,x0=w,args=(X,Y,c),approx_grad=False)
    # 求得最优化计算后的w
    w=REF[0].round(2)           # 取得w值
    print('w:',w[1:],'b:',w[0]) # 与《统计学习方法》李航，P103，例7.1计算结果一致
    # 画图
    plotResult(w)
    print('\n1、SVM直接求参数值，结束')
    
    ##2、SVM简易SMO算法#############################################################
    print('\n2、SVM简易SMO算法，开始')
    fileIn = './testSet.txt'
    #dataSet=pd.read_table(fileIn,names=['x1','x2','y']).values
    dataSet=np.loadtxt(fileIn)
    dataArr=dataSet[:,:-1] # x
    labelArr=dataSet[:,-1] # y
    b1,alphas1=smoSimple(dataArr,labelArr,0.6,0.001,50) # 输入变量：x、y、c：常数c、toler：容错率、maxIter：最大循环次数
    dataArrWithAlpha1=np.array(np.concatenate((dataSet,alphas1),axis=1)) # 把alphas1与原始数据合并
    w1=calcWs(alphas1,dataArr,labelArr)                                    # 根据alpha求w
    print('b:',b1,'\nw:',w1,'\ndata，alphas，支撑向量:\n',dataArrWithAlpha1[dataArrWithAlpha1[:,-1]>0] )# 注意这里的筛选方式与pd.DataFrame筛选方式一致，array类型的才可以这样写，np.ndarray及np.matrix类型不可以使用
    plot_smoSimple(dataArrWithAlpha1,b1,w1)  # 画图
    print('2、SVM简易SMO算法，结束')
    
    ##3、SVM完整SMO算法#############################################################
    print('\n3、SVM完整SMO算法，开始')
    dataSetTrain = np.loadtxt('./testSetRBF.txt')
    dataSetTest  = np.loadtxt('./testSetRBF2.txt')
    # 训练集
    dataArrTrain=dataSetTrain[:,:-1] # 训练集x
    labelArrTrain=dataSetTrain[:,-1] # 训练集y
    # 测试集
    dataArrTest=dataSetTest[:,:-1]   # 测试集x
    labelArrTest=dataSetTest[:,-1]   # 测试集y
    # 调用主函数
    b2,alphas2=testRbf(dataArrTrain,labelArrTrain,dataArrTest,labelArrTest,k1=1.3)
    w2=calcWs(alphas2,dataArrTrain,labelArrTrain)                                    # 根据alpha求w
    dataArrWithAlpha2=np.array(np.concatenate((dataSetTrain,alphas2),axis=1)) # 把alphas1与原始数据合并
    print('b:',b1,'\nw:',w1,'\ndata，alphas，支撑向量:\n',dataArrWithAlpha2[dataArrWithAlpha2[:,-1]>0] )# 注意这里的筛选方式与pd.DataFrame筛选方式一致，array类型的才可以这样写，np.ndarray及np.matrix类型不可以使用
    plot_smoCompletion() # 画图，训练集
    print('3、SVM完整SMO算法，结束')
