
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import initialize as init
import observables as obs


def md(dim, natom, L, position, velocity, mass, sigma, steps, tau, fileName):
    print 'Molecular Dynamics Simulation'
    print 'Dimensions:', dim
    print 'Size of Box:', L
    print 'Number of particles:', natom
    print 'Volume Fraction:', np.pi/6.0 * natom/(L[0]*L[1]*L[2])
    print 'Number of Timesteps:', steps
    print 'Timestep size', tau

    print_step = 25
    dof = dim*natom - dim
    pos0 = np.copy(position)
    pos = np.copy(position)
    vel = np.copy(velocity)
    acc = np.zeros((natom,dim))
    pbc = np.zeros((natom,dim))

    force, potE, kinE = compute_LJ(dim, natom, L, pos, vel, mass, sigma)
    totE = potE + kinE

    print "%8s %14s %14s %14s %14s" % ('step', 'potE', 'kinE', 'totE', 'temp')
    open(fileName, 'w').close() #clears the file if it exists already
    #print "%8d %14g %14g %14g" % (0, potE, kinE, totE)
    #dumpPos(pos, 0, 'test.xyz')

    for step in range(0,steps+1):
        pos, vel, acc, pbc = evolve(dim, natom, L, pos, vel, acc, pbc, mass, sigma, tau)

        potE, kinE = compute_LJ(dim, natom, L, pos, vel, mass, sigma)[1:]
        totE = potE + kinE
        T = 2*kinE/(natom*dof)

        if (step+print_step) % print_step == 0:
            print "%8d %14g %14g %14g %14g" % (step, potE/natom, kinE/natom, totE/natom, T)
            dumpPos(pos, step, fileName)
            dumpVel(vel, step)
            dumpAcc(acc, step)
            solMSD(pos, step, pos0, pbc, L[0])
            bigMSD(pos, step, pos0, pbc, L[0])
    return

def compute_LJslow(dim, natom, L, pos, vel, mass, sigma):
    #Calculates forces on particles according to 12-6 LJ potential
    #Uses periodic boundary conditions
    force = np.zeros([natom, dim])
    rij = np.zeros((dim))
    potE = 0.0
    kinE = 0.0
    rc = 5.0

    sig2 = 2.0 #double the radius because sig is the sum of the 2 particls' radii

    for i in range (natom):
        for j in range(natom):
            if i != j:
                rij = pos[i,:] - pos[j,:]
                #Periodic boundary
                rij = rij - L[0] *np.round(rij/L[0])
                rhat = rij/np.linalg.norm(rij)

                d2rij = (rij*rij).sum()
                drij = np.sqrt(d2rij)
                #print drij
                #time.sleep(0.05)
                potE += 4*(np.power(d2rij,-6) - np.power(d2rij,-3))
                if drij < rc:
                    if i==0:
                        force[i,:] = force[i,:] + 48*drij*(np.power( sig2/d2rij ,7) - 0.5*np.power( sig2/d2rij ,4)) *rhat
                    elif j==0:
                        force[i,:] = force[i,:] + 48*drij*(np.power( sig2/d2rij ,7) - 0.5*np.power( sig2/d2rij ,4)) *rhat
                    else:
                        force[i,:] = force[i,:] + 48*drij*(np.power(d2rij,-7) - 0.5*np.power(d2rij,-4)) *rhat
                #LJ Potential Force with sig=1 and eps=1-->LJ units


    kinE = 0.5*np.sum(mass[:,None]*vel[::]*vel[::])

    return force, potE, kinE

def compute_LJ(dim, natom, L, pos, vel, mass, sig2):
    #Calculates forces on particles according to 12-6 LJ potential
    #Uses periodic boundary conditions
    potE = 0.0
    kinE = 0.0
    rc = 2.5
    l = L[0]

    #Build array of separations: distance between particle i and j is Rij[i,j,:]
    #where the third dimension stores the 3 cartesian coords of r_ij
    Rij = pos[None,:] - pos[:,None]

    #Periodic Boundaries
    Rij[Rij>(l/2.0)] = Rij[Rij>(l/2.0)] - l
    Rij[Rij<(-l/2.0)] = Rij[Rij<(-l/2.0)] + l
    #print "Rij:\n", Rij

    #Get the sum of the squares in each direction for each r_ij
    R2ij = np.sum(Rij*Rij,axis=2)

    #Get the norms
    Rnorm = np.sqrt(R2ij)

    #Get the unit vectors for each pair-distance
    Rhat = Rij/Rnorm[:,:,None]
    Rhat[np.isnan(Rhat)] = 0.0

    #Calculate the LJ forces between each particle in each direction
    Fij = np.copy(Rnorm)
    Fij[R2ij!=0] = 48*Rnorm[R2ij!=0]*(np.power(R2ij[R2ij!=0],-7) - 0.5*np.power(R2ij[R2ij!=0],-4))
    #print Fij[0]

    #Polydispers
    #Fij[0,1:] = 48*Rnorm[0,1:]*(np.power( R2ij[0,1:]/sig2, -7) - 0.5*np.power( R2ij[0,1:]/sig2, -4))
    #Fij[1:,0] = 48*Rnorm[1:,0]*(np.power( R2ij[1:,0]/sig2, -7) - 0.5*np.power( R2ij[1:,0]/sig2, -4))
    #print Fij[0]
    #time.sleep(5)

    Fvec = Fij[:,:,None]*Rhat
    #Add up all the forces on particle i due to all particles j
    force = np.sum(Fvec,axis=0)

    #Calculate potE
    potential = np.copy(R2ij)
    potential[R2ij!=0] = 4*(np.power(R2ij[R2ij!=0],-6) - np.power(R2ij[R2ij!=0],-3))
    potE = potential.sum()

    #Calculate kinE
    kinE = 0.5*np.sum(mass[:,None]*vel[::]*vel[::])

    return force, potE, kinE

def evolve(dim, natom, L, pos0, vel0, acc0, pbc0, mass, sigma, tau):
    vel = np.copy(vel0)
    pos = np.copy(pos0)
    acc = np.copy(acc0)
    pbc = np.copy(pbc0)

    #Velocity Verlet--first half
    vel = vel + 0.5*tau*acc
    pos = pos + tau*vel #+ 0.5*tau*tau*acc
    #PBCs
    pbc[(pos>L[0])] = pbc[(pos>L[0])] + 1.0
    pbc[(pos<0.0)] = pbc[(pos<0.0)] - 1.0
    pos[(pos>L[0])] = pos[(pos>L[0])] - L[0]
    pos[(pos<0.0)] = pos[(pos<0.0)] + L[0]

    #Compute accelerations from updated positions
    force = compute_LJ(dim, natom, L, pos, vel, mass, sigma)[0]
    acc = force / mass[:,None] #\
                #- Gauss_Thermostat(natom, force, vel, mass)

    #Velocity Verlet--Last Step: Get full step velocities
    vel = vel + 0.5*tau*acc

    return pos, vel, acc, pbc

def mxwl(dim, natom, vel, T):
    total_v_components = dim*natom
    dof = dim*natom - dim

    #Assign Gaussian dist to each velocity component
    for i in range(natom):
        for k in range(dim):
            vel[i][k] = np.random.standard_normal()

    #Scale to satisfy equipartition theorem
    ek = (vel*vel).sum()
    vs = np.sqrt(ek/(dof*T))
    vel = vel/vs

    #Calculate COM velocity
    vcm = vel.sum(axis=0) #1d array with [vcm_x,vcm_y,vcm_z]
    #Move to COM frame to negate any inadvertent flow
    vel = vel - vcm
    return vel

def Gauss_Thermostat(natom, F, vel, mass):
    mom = vel*mass[:,None]
    #Calcualte weight parameter alpha
    num = np.sum(F*mom)
    denom = np.sum(mom*mom)
    if denom < 0.01:
        print "momentum sum to zer0"
        quit()
    alpha = num/denom
    print alpha, num

    return alpha*mom


def swap(pos, i, j):
    #Swap positions and velocitiees of particle i with particle j
    temp = np.copy(pos[i])
    pos[i] = pos[j]
    pos[j] = temp
    return

def initialize(dim, natom, L, T, type):
    if (type=="Random" or type =="random" or type=="rand" or type=="Rand"):
        pos, vel = init.initialize_random(dim, natom, L, 1.2, T)
    if (type=="Cube" or type =="cube"):
        pos, vel = init.initialize_cube(natom, L, 1.2, T)
    if (type=="Square" or type =="square" or type=="sq" or type=="Sq"):
        pos, vel = init.initialize_square(natom, L, 1.2)
    if (type=="Basic" or type =="basic"):
        pos, vel = init.basic_init(dim, natom, L)
    return pos, vel

def init_fromFile(fileName_pos, fileName_vel):
    pos = np.loadtxt(fileName_pos)
    vel = np.loadtxt(fileName_vel)
    return pos, vel

def dumpPos(pos, step, fileName):
    f = open(fileName, 'a')
    N = pos.shape[0]
    line = "{0:d} \nAtoms. Timestep: {1:g} \n".format(N,step)
    f.write(line)

    type = np.full((N),1, dtype=float)
    type[0] = 2
    for i in range(N):
        line = "{0:g} {1:g} {2:g} {3:g} \n".format(type[i], pos[i,0], pos[i,1], pos[i,2])
        f.write(line)
    return

def dumpVel(vel, step):
    fileName = './Test/velComp.txt'
    if step == 0:
        h = open(fileName,'w')
    else:
        h = open(fileName,'a')
    N = vel.shape[0]
    line = "{0:d} \nAtoms. Timestep: {1:g} \n".format(N,step)
    h.write(line)

    type = np.full((N),1, dtype=float)
    type[0] = 2
    for i in range(N):
        line = "{0:g} {1:g} {2:g} {3:g} \n".format(type[i], vel[i,0], vel[i,1], vel[i,2])
        h.write(line)
    return

def dumpAcc(acc, step):
    fileName = './Test/accBad.txt'
    if step == 0:
        k = open(fileName,'w')
    else:
        k = open(fileName,'a')
    N = acc.shape[0]
    line = "{0:d} \nAtoms. Timestep: {1:g} \n".format(N,step)
    k.write(line)

    type = np.full((N),1, dtype=float)
    type[0] = 2
    for i in range(N):
        line = "{0:g} {1:g} {2:g} {3:g} \n".format(type[i], acc[i,0], acc[i,1], acc[i,2])
        k.write(line)
    return


def solMSD(pos, step, pos0, pbc, l):
    #Calculates MSD at timestep 'step' and writes to a file
    fileName = 'msdTest.txt'
    if step == 0:
        g = open(fileName,'w')
    else:
        g = open(fileName,'a')
    msd = obs.MSD(pos[1:], pos0[1:], pbc[1:], l)
    line = "{0:g} {1:g}\n".format(step, msd)
    g.write(line)
    return

def bigMSD(pos, step, pos0, pbc, l):
    #Calculates MSD at timestep 'step' for the colloid
    fileName = 'msdColloid.txt'
    if step == 0:
        g2 = open(fileName,'w')
    else:
        g2 = open(fileName,'a')
    msd = obs.MSD(pos[0], pos0[0], pbc[0], l)
    line = "{0:g} {1:g}\n".format(step, msd)
    g2.write(line)
    return


if __name__ == '__main__':
    dim = 3
    natom = 100
    steps = 1000
    dt = 0.01
    l = 6.786           #Phi=0.4
    L = [l,l,l]

    mass = np.full((natom),1.0, dtype=float)
    mass[0] = 25

    sigma = np.full((natom), 1.0, dtype=float)
    sig2 = 2.0
    sigma[0] = sig2

    fileName = 'test.xyz'
    pos, vel = initialize(dim, natom, L, 0.1, "cube")
    #pos, vel = initialize(dim, natom, L, 1.0, 'basic')
    vel[0] = 0.0 #for bigger particle
    a = 0; b = np.int(natom/2)
    swap(pos,a,b)
    print "vel sum:", vel.sum()
    md(dim, natom, L, pos, vel, mass, sig2, steps, dt, fileName)
