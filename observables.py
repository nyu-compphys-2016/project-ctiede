from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

def plotLJ():

    def LJ(r):
        return 4.0*(r**(-12) - r**(-6))

    x = np.linspace(0.1, 2.5, num=1000)
    lj = LJ(x)
    print min(lj)

    steric = x**(-12)
    att = -4*x**(-6)

    fig = plt.figure()
    ax = fig.add_subplot(111, ylim=[-1.25,1.25], xlim=[0.5,2.5])
    ax.plot(x, lj, lw='4', color=blue, label='Lennard-Jones')
    ax.plot(x, steric, lw='2.5', ls='--', color=red, label=r'Repulsive: $\frac{A}{r^{12}}$')
    ax.plot(x, att, lw='2.5', ls='--', color=green, label=r'Attractive: $\frac{-B}{r^6}$')


    ax.set_xlabel('pair-separation ($\sigma$)', fontsize=20)
    ax.set_ylabel('Energy ($\epsilon$)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.5)
    ax.tick_params(axis='both', which='minor', length=4, width=1)
    ax.legend(loc='upper right', prop = {'size':18})


    fig.tight_layout()
    #fig.savefig('./Presentation/LJplot.pdf')
    plt.show()


def plotEnergy():
    E = np.loadtxt('EnergiesTest.txt')
    t = E[:,0]
    potE = E[:,1]
    kinE = E[:,2]
    totE = E[:,3]

    fig = plt.figure()
    ax = fig.add_subplot(111, ylim=[-10.0,10.0])
    ax.plot(t, potE, lw=4, ls='-', color=blue, label='Potential') #: slope={0:.3f}'.format(m))
    ax.plot(t, kinE, lw=4, ls='-', color=green, label='Kinetic') #: slope={0:.3f}'.format(mc))
    ax.plot(t, totE, lw=4, ls='--', color=red, label='Total')

    ax.set_xlabel('MD Timesteps', fontsize=20)
    ax.set_ylabel('Energy/N ($\epsilon$)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.5)
    ax.tick_params(axis='both', which='minor', length=4, width=1)
    ax.legend(loc='upper right', prop = {'size':18})

    fig.tight_layout()
    #fig.savefig('./Presentation/RescaleEnergy_T=2.5.png')
    plt.show()
    return

def plotMSD():
    MSD = np.loadtxt('msdTest.txt')
    MSDc1 = np.loadtxt('msdColloid_temp.txt')
    MSDc2 = np.loadtxt('msdColloid_temp2.txt')
    MSDc3 = np.loadtxt('msdColloid_temp3.txt')
    MSDc4 = np.loadtxt('msdColloid_temp4.txt')
    MSDc5 = np.loadtxt('msdColloid_temp5.txt')

    ts = MSD[:,0]
    msd = MSD[:,1]
    msd_col1 = MSDc1[:,1]
    msd_col2 = MSDc2[:,1]
    msd_col3 = MSDc3[:,1]
    msd_col4 = MSDc4[:,1]
    msd_col5 = MSDc5[:,1]
    msd_ave = (msd_col1+msd_col2+msd_col3+msd_col4+msd_col5)/5

    m, c = Fit(np.log10(ts), np.log10(msd))
    mc, cc = Fit(np.log10(ts), np.log10(msd_col1))
    #m, c = Fit(t,msd)
    #mc, cc = Fit(t, msd_col)
    print "Solute Slope:", m
    print "Colloid Slope:", mc

    m_exp = 1.0
    m_b = 2.0
    ce = -1.8
    cb = -3.5
    fit = pow(10,m_exp*np.log10(ts)+ce)
    fit2 = pow(10,m_b*np.log10(ts)+cb)

    fig = plt.figure()
    ax = fig.add_subplot(111, xscale='log', yscale='log', xlim=[1,10000], ylim=[1e-4,1e4])
    ax.plot(ts, msd, lw='4', color=blue, label='Surrounding Particles') #: slope={0:.3f}'.format(m))
    ax.plot(ts, msd_col1, lw='2', color=lightBlue, alpha=0.25) #: slope={0:.3f}'.format(mc))
    ax.plot(ts, msd_col2, lw='2', color=violet, alpha=0.25)
    ax.plot(ts, msd_col3, lw='2', color=orange, alpha=0.25)
    ax.plot(ts, msd_col4, lw='2', color=green, alpha=0.25)
    ax.plot(ts, msd_col5, lw='2', color=red, alpha=0.25)
    ax.plot(ts, msd_ave, lw='4', color=orange, label='Colloid Average')
    ax.plot(ts, fit, label='Thermal Expectation', ls='--', lw=3, color=blue, alpha=0.8)
    ax.plot(ts, fit2, label='Ballistic Expectation', ls='--', lw=3, color=violet, alpha=0.8)

    ax.set_xlabel(r'MD Timesteps ($\tau$)', fontsize=20)
    ax.set_ylabel(r'MSD ($\sigma$)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.5)
    ax.tick_params(axis='both', which='minor', length=4, width=1)
    ax.legend(loc='upper left', prop = {'size':14})

    #left, bottom, width, height = [0.62,0.21,0.28,0.28]
    #ax2 = fig.add_axes([left, bottom, width, height])
    #ax2.plot(ts, msd_ave, lw=3, color=orange)
    #ax2.plot(ts, msd_col1, lw='2', color=lightBlue, alpha=0.25) #: slope={0:.3f}'.format(mc))
    #ax2.plot(ts, msd_col2, lw='2', color=violet, alpha=0.25)
    #ax2.plot(ts, msd_col3, lw='2', color=orange, alpha=0.25)
    #ax2.plot(ts, msd_col4, lw='2', color=green, alpha=0.25)
    #ax2.plot(ts, msd_col5, lw='2', color=red, alpha=0.25)
    #ax2.tick_params(axis='both', which='major', labelsize=8, length=2, width=0.5)
    #ax2.tick_params(axis='both', which='minor', length=4, width=1)

    fig.tight_layout()
    #fig.savefig('./Results/colloidMSD335_noslope.pdf')
    plt.show()
    return

def spatialDensity():
    control = np.loadtxt('./Test/baseDens200.txt')
    dens = np.loadtxt('./Test/tempSplit_Dens_GeoffWay.txt')

    x = np.linspace(-0.5,8.5,1000)
    flat = np.full((len(x)), 25, dtype='float')

    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=[-0.5,8.5])
    ax.hist(control[:,2],bins=8, label='Constant $T$', color=blue,histtype='stepfilled')
    ax.hist(dens[:,2], bins=8, label='Split $T$', color=orange, histtype='step', lw='4')
    ax.plot(x, flat, color=red, ls='--', lw=2)

    ax.set_xlabel(r'$z \ (\sigma)$', fontsize=24)
    ax.set_ylabel('Particles', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16, length=7, width=1.5)
    ax.tick_params(axis='both', which='minor', length=4, width=1)
    ax.legend(loc='upper right', prop = {'size':16})

    fig.tight_layout()
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
    #plotMSD()
    #plotEnergy()
    #plotLJ()
    spatialDensity()
