
import numpy as np
import numpy.linalg as lalg
import plotly
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
def Y_getLpad_A(Y):
    Y_Lpad =  [[ Y[1][1] - Y[2][1] ,  2*Y[2][1] ],
               [     2*Y[2][1]     , -2*Y[2][1] ]
    return Y_Lpad


#This function gets Y_Rpad from Y admitance matrix
def Y_getRpad_A(Y):
    Y_Rpad =  [[ -2*Y[1][2] ,     2*Y[1][2]     ],
               [  2*Y[1][2] , Y[2][2] - Y[1][2] ]
    return Y_Rpad

def getTfromY(Y):
    T = [[,],[,]]

#def Rsweep(A):


matrix = CSVmatrix('thru.csv')

freq = []
for i in range(1,len(matrix)):
    freq.append(matrix[i][0])

matrix = complexMatrix(matrix)

s11 = []
for i in range(len(matrix)):
    s11.append(matrix[i][0][0])


#THIS SECTION PLOTS MAGNITUDES OF S11 IN FREQ

s11 = np.absolute(s11)

trace0 = Scatter(
    x=freq,
    y=s11
)

data = Data([trace0])

plotly.offline.plot({
    "data": [trace0],
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
