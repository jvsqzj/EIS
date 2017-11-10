
import numpy as np
import numpy.linalg as lalg
import plotly
plotly.tools.set_credentials_file(username='jvsqzj', api_key='xl7rKrwa7iAfeT2SjWhG')
import plotly.plotly as py
from plotly.graph_objs import *
import csv

def CSVmatrix(filename):
    matrix = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            matrix.append(row[0].split(','))
    return matrix

def inv(A):
    inverse = [[A[1][1],-1*A[0][1]],[-1*A[1][0],A[0][0]]]
    det = A[0][0]*A[1][1]-A[0][1]*A[1][0]
    for i in range(len(inverse)):
        for j in range(len(inverse[i])):
            inverse[i][j] = inverse[i][j]/det
    return inverse

def complexMatrix(iM):
    cM = []
    for i in range(1,len(iM)):
        re = []
        im = []
        for k in range(1,len(iM[i]),2):
            re.append(float(iM[i][k]))
            im.append(float(iM[i][k+1]))
        cM.append([[re[0]+1j*im[0],re[1]+1j*im[1]],[re[2]+1j*im[2],re[3]+1j*im[3]]])
    return cM

#This function gets Y_Lpad from Y admitance matrix
def Y_getLpad(Y):
    Y_Lpad =  [[ Y[0][0] - Y[1][0] ,  2*Y[1][0] ],
               [     2*Y[1][0]     , -2*Y[1][0] ]]
    return Y_Lpad

#This function gets Y_Rpad from Y admitance matrix
def Y_getRpad(Y):
    Y_Rpad =  [[ -2*Y[0][1] ,     2*Y[0][1]     ],
               [  2*Y[0][1] , Y[1][1] - Y[0][1] ]]
    return Y_Rpad

def getTfromY(Y):
    T = [[-Y[1][1] / Y[1][0],   -1/Y[1][0]   ],
         [ -(Y[0][0]*Y[1][1]-Y[0][1]*Y[1][0])/Y[1][0]  ,-Y[0][0]/Y[1][0]]]
    return T

def T_Deembed(TLinv, TMeas, TRinv):
    a = TMeas[0][0]
    b = TMeas[0][1]
    c = TMeas[1][0]
    d = TMeas[1][1]
    L = TLinv
    R = TRinv
    T_dut = [[(a*L[0][0]+c*L[0][1])*R[0][0]+(b*L[0][0]+d*L[0][1])*R[1][0],(a*L[0][0]+c*L[0][1])*R[0][1]+(b*L[0][0]+d*L[0][1])*R[1][0]],
             [(a*L[1][0]+c*L[1][1])*R[0][0]+(b*L[1][0]+d*L[1][1])*R[1][0],(a*L[1][0]+c*L[1][1])*R[0][1]+(b*L[0][0]+d*L[0][1])*R[1][0]]]
    return T_dut

def SfromT(T):
    s11 = T[0][1]/T[1][1]
    s12 = (T[0][0]-T[1][1])/T[1][1]
    s21 = 1/T[1][1]
    s22 = -T[1][0]/T[1][1]
    S = [[s11,s12],
         [s21,s22]]
    return S

def TfromS(S):
    A = -(S[0][0]*S[1][1]-S[0][1]*S[1][0])/S[1][0]
    B = S[0][0]/S[1][0]
    C = -S[1][1]/S[1][0]
    D = 1/S[1][0]
    T = [[A,B],[C,D]]
    return T

#M y Thru deben ser matrices del mismo tamano para utilizar esta funcion
def DeEmbeddingSweep(tM,thru):
    S = []
    for i in range(len(tM)):
        YL = Y_getLpad(thru[i])
        YR = Y_getRpad(thru[i])
        TL = inv(getTfromY(YL))
        TR = inv(getTfromY(YR))
        x = T_Deembed(TL,tM[i],TR)
        y = SfromT(x)
        S.append(y)
    return S



#def Rsweep(A):

thru = CSVmatrix('thru.csv')  ##This is the Y matrix of the thru DeEmbedding fixture
sM = CSVmatrix('DUT.csv')    ##this is the S matrix of the Measured device

freq = []
for i in range(1,len(thru)):
    freq.append(thru[i][0])

thru = complexMatrix(thru)
sM = complexMatrix(sM)

for i in range(len(sM)):
    sM[i] = TfromS(sM[i])

Sparam = DeEmbeddingSweep(sM, thru)


#THIS SECTION PLOTS MAGNITUDES OF S11 IN FREQ

s11 = []
s12 = []
s21 = []
s22 = []
for i in range(len(Sparam)):
    s11.append(Sparam[i][0][0])

s11 = np.absolute(s11)
s12 = np.absolute(s12)
s21 = np.absolute(s21)
s22 = np.absolute(s22)

trace0 = Scatter(
    x=freq,
    y=s11
)

trace1 = Scatter(
    x=freq,
    y=s12
)

trace2 = Scatter(
    x=freq,
    y=s21
)

trace3 = Scatter(
    x=freq,
    y=s22
)

data = Data([trace0])

plotly.offline.plot({
    "data": [trace0,trace1,trace2,trace3],
    "layout": Layout(title="hello world")
})

#data = Data([freq,s11])

#py.plot(data, filename='s11')

'''
x = CSVmatrix('thru.csv')
cM = complexMatrix(x)
#two = [[cM[0][1][1],-1*cM[0][0][1]],[0,1]]
#two = [x*(cM[0][]-cM[0]cM[0])]
one = inv(cM[0])
print cM[0]
print one
identity = np.array(cM[0])*np.array(one)
print np.absolute(identity)
'''
