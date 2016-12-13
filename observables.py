from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import initialize as init

blue = (31/255.0, 119/255.0, 180/255.0)
lightBlue = (114/255.0, 158/255.0, 206/255.0)
orange = (255/255.0, 127/255.0, 14/255.0)
purple = (148/255.0, 103/255.0, 189/255.0)
green = (103/255.0, 191/255.0, 92/255.0)
red = (237/255.0, 102/255.0, 93/255.0)
violet = (173/255.0, 139/255.0, 201/255.0)

def MSD(pos, pos0, pbc, L):
    #Calculate the Mean squared displacement of the system
    N = pos.shape[0]
    dr = (pos - pos0) + L*pbc
    MSD = (dr*dr).sum()
    return MSD/N

def RDF(dim, natom, pos, L):
    return rdf

def plotMSD():
    MSD = np.loadtxt('msdTest.txt')
    MSDc = np.loadtxt('msdColloid.txt')
    t = MSD[:,0]
    msd = MSD[:,1]
    msd_col = MSDc[:,1]

    m, c = Fit(np.log10(t), np.log10(msd))
    mc, cc = Fit(np.log10(t), np.log10(msd_col))
    #m, c = Fit(t,msd)
    #mc, cc = Fit(t, msd_col)
    print "Solute Slope:", m
    print "Colloid Slope:", mc

    m_exp = 1.0
    ce = -3.6
    fit = pow(10,m_exp*np.log10(t)+ce)

    fig = plt.figure()
    ax = fig.add_subplot(111, xscale='log', yscale='log', xlim=[1,10000])
    ax.plot(t, msd, lw='4', color=blue, label='Suspension') #: slope={0:.3f}'.format(m))
    ax.plot(t, msd_col, lw='4', color=red, label='Colloid') #: slope={0:.3f}'.format(mc))
    ax.plot(t, fit, label='Thermal Expectation', lw=3, color=lightBlue, alpha=0.6)

    ax.set_xlabel('MD Timesteps', fontsize=20)
    ax.set_ylabel('MSD', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.5)
    ax.tick_params(axis='both', which='minor', length=4, width=1)
    ax.legend(loc='upper left', prop = {'size':14})

    fig.tight_layout()
    #fig.savefig('./Results/colloidMSD335_noslope.pdf')
    plt.show()

def Fit(x,y):
    x = np.asarray(x)
    y = np.asarray(y)
    y = y[(x > np.log10(1000))]
    x = x[(x > np.log10(1000))]
    #print len(x), len(y)
    N = len(x)

    Ex = x.sum() /N
    Ey = y.sum() / N
    Exx = (x**2).sum() / N
    Exy = (x*y).sum() / N

    m = (Exy - Ex*Ey)/(Exx - Ex**2)
    c = (Exx*Ey - Ex*Exy)/(Exx - Ex**2)

    return abs(m), c


if __name__ == '__main__':
    plotMSD()
