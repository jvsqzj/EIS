
import numpy as np
import numpy.linalg as lalg
import plotly.plotly as py
import plotly.graph_objs as go
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

x = CSVmatrix('thru.csv')
cM = complexMatrix(x)
#two = [[cM[0][1][1],-1*cM[0][0][1]],[0,1]]
#two = [x*(cM[0][]-cM[0]cM[0])]
one = inv(cM[0])
print cM[0]
print one
identity = np.array(cM[0])*np.array(one)
print np.absolute(identity)
