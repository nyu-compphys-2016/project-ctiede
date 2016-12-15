
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import initialize as init
import observables as obs


def md(dim, natom, L, position, velocity, mass, sigma, T, steps, tau, fileName):
    print 'Molecular Dynamics Simulation'
    print 'Dimensions:', dim
    print 'Size of Box:', L
    print 'Number of particles:', natom
    print 'Volume Fraction:', np.pi/6.0 * natom/(L[0]*L[1]*L[2])
    print 'Number of Timesteps:', steps
    print 'Timestep size', tau

    print_step = 25
    rescale_step = 500
    dof = dim*natom - dim
    pos0 = np.copy(position)
    pos = np.copy(position)
    vel = np.copy(velocity)
    acc = np.zeros((natom,dim))
    pbc = np.zeros((natom,dim))

    binNum = int(2*L[2])
    hist = np.zeros(binNum)
    rescales = 0

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
        temp = 2*natom*kinE/(dof)

        Tmax = 1.5
        Tmin = 1.0
        #if step > 1000:
        #    vel = velRescaleGradient(dim, natom, L, pos, vel, Tmax, Tmin, rescale_step)
        #else:
        vel = velRescaleUniform(dim, natom, vel, T, rescale_step)

        #if (step+rescale_step) % rescale_step == 0:
        #    if step > 1000:
        #        Tmax = 1.5
        #        Tmin = 1.0
        #        hist = (hist + np.histogram(pos, bins=binNum, range=(0,L[2]))[0])
        #        vel = velRescaleGradient(dim, natom, L, pos, vel, Tmax, Tmin)
        #        rescales+=1
        #    elif step > 0.0:
        #        vel = velRescaleUniform(dim, natom, vel, T)


        if (step+print_step) % print_step == 0:
            print "%8d %14g %14g %14g %14g" % (step, potE, kinE, totE, temp)
            dumpPos(pos, step, fileName)
            dumpVel(vel, step)
            dumpAcc(acc, step)
            dumpEnergy(potE, kinE, totE, step)
            solMSD(pos, step, pos0, pbc, L[0])
            bigMSD(pos, step, pos0, pbc, L[0])
        if step < 25:
            solMSD(pos, step, pos0, pbc, L[0])
            bigMSD(pos, step, pos0, pbc, L[0])

    createRestart(dim, natom, L, pos, vel, step)
    np.savetxt("densityHistDat_16bins.txt", hist/rescales)
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

    return force, potE/natom, kinE/natom

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

    return force, potE/natom, kinE/natom

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
    acc = force / mass[:,None]
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

def velRescaleUniform(dim, natom, vel0, T, rescaleStep):
    total_v_components = dim*natom
    dof = dim*natom - dim
    vel = np.copy(vel0)

    #Scale velocities to equipartition theorem
    ek = np.sum(vel*vel)
    vs = np.sqrt(ek/(dof*T))
    vel = vel/(vs**(1/rescaleStep))
    return vel

def velRescaleGradient(dim, natom, L, pos0, vel0, Tmax, Tmin, rescaleStep):
    total_v_components = dim*natom
    dof = dim*natom - dim
    pos = np.copy(pos0)
    vel = np.copy(vel0)
    l = L[2]

    #offset = 2.0
    #vout = (pos[:,2] < offset) & (pos[:,2] > (l-offset) )
    #vin = (pos[:,2] > offset) & (pos[:,2] < (l-offset))


    vbot = (pos[:,2] < l/2)
    vtop = (pos[:,2] > l/2)


    #Bottom Cold Region
    v0 = vel[vbot]              #velocity of each particle in region
    N0 = v0.shape[0]            #Number of particles in this region
    z0 = pos[vbot][:,2]         #z-positions of each particle in region
    dof0 = dim*N0 - dim         #Total number of dof in this region
    ek0 = np.sum(v0*v0)         #Kinetic Energy of this region
    vs0 = np.sqrt(ek0/(dof0*Tmin))
    vel[vbot] = vel[vbot]/(vs0**(1/rescaleStep))
    #print pos[vtop]
    #print "Low Temp:", ek0/dof0
    #print "low temp after:", np.sum(vel[vbot]*vel[vbot])/dof0

    #Top Hot region
    v1 = vel[vtop]
    N1 = v1.shape[0]
    dof1 = dim*N1 - dim
    ek1 = np.sum(v1*v1)
    vs1 = np.sqrt(ek1/(dof1*Tmax))
    vel[vtop] = vel[vtop]/(vs1**(1/rescaleStep))
    #print "High Temp:", ek1/dof1
    #print "high temp after:", np.sum(vel[vtop]*vel[vtop])/dof1
    print 'Top Particles:', N1
    print 'Bot Particles:', N0

    return vel

def velRescaleGradient4(dim, natom, L, pos0, vel0, Tmax, Tmin, rescaleStep):
    total_v_components = dim*natom
    dof = dim*natom - dim
    pos = np.copy(pos0)
    vel = np.copy(vel0)
    l = L[2]

    #offset = 2.0
    #vout = (pos[:,2] < offset) & (pos[:,2] > (l-offset) )
    #vin = (pos[:,2] > offset) & (pos[:,2] < (l-offset))

    divs = 4
    vbot = (pos[:,2] < l/4)
    vtop = (pos[:,2] > 3*l/4)
    vmid1 = (pos[:,2] > l/4) & (pos[:,2] < l/2)
    vmid2 = (pos[:,2] > l/2) & (pos[:,2] < 3*l/4)

    Tdiff = (Tmax - Tmin)/3
    Tmid1 = Tmax - Tdiff
    Tmid2 = Tmid1 - Tdiff

    #Bottom Cold Region
    v0 = vel[vbot]              #velocity of each particle in region
    N0 = v0.shape[0]            #Number of particles in this region
    z0 = pos[vbot][:,2]         #z-positions of each particle in region
    dof0 = dim*N0 - dim         #Total number of dof in this region
    ek0 = np.sum(v0*v0)         #Kinetic Energy of this region
    vs0 = np.sqrt(ek0/(dof0*Tmin))
    vel[vbot] = vel[vbot]/(vs0**(1/rescaleStep))
    #print pos[vtop]
    #print "Low Temp:", ek0/dof0
    #print "low temp after:", np.sum(vel[vbot]*vel[vbot])/dof0

    #Top Hot region
    v1 = vel[vtop]
    N1 = v1.shape[0]
    dof1 = dim*N1 - dim
    ek1 = np.sum(v1*v1)
    vs1 = np.sqrt(ek1/(dof1*Tmax))
    vel[vtop] = vel[vtop]/(vs1**(1/rescaleStep))
    #print "High Temp:", ek1/dof1
    #print "high temp after:", np.sum(vel[vtop]*vel[vtop])/dof1
    #print 'Top Particles:', N1
    #print 'Bot Particles:', N0

    #Upper Middle Region
    v3 = vel[vmid1]
    N3 = v1.shape[0]
    dof3 = dim*N3 - dim
    ek3 = np.sum(v3*v3)
    vs3 = np.sqrt(ek3/(dof3*Tmid1))
    vel[vmid1] = vel[vmid1]/(vs3**(1/rescaleStep))

    #Lower Middle Region
    v2 = vel[vmid2]
    N2 = v1.shape[0]
    dof2 = dim*N2 - dim
    ek2 = np.sum(v2*v2)
    vs2 = np.sqrt(ek2/(dof2*Tmid2))
    vel[vmid2] = vel[vmid2]/(vs2**(1/rescaleStep))

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

def PosBin(pos0, hist):
    pos = np.copy(pos)
    #hist +=
    return

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

def dumpEnergy(potE, kinE, totE, step):
    fileName = 'EnergiesTest.txt'
    if step == 0:
        k = open(fileName,'w')
    else:
        k = open(fileName,'a')

    line = "{0:g} {1:g} {2:g} {3:g} \n".format(step, potE, kinE, totE)
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
    fileName = 'msdColloid_temp5.txt'
    if step == 0:
        g2 = open(fileName,'w')
    else:
        g2 = open(fileName,'a')
    msd = obs.MSD(pos[0], pos0[0], pbc[0], l)
    line = "{0:g} {1:g}\n".format(step, msd)
    g2.write(line)
    return

def createRestart(dim, natom, L, pos, vel, step):
    #Creates restart files for positions and velocities from the state of our
    #system at 'step' with positions and velocities 'pos' and 'vel'
    np.savetxt("restart_pos.txt", pos)
    np.savetxt("restart_vel.txt", vel)
    return



if __name__ == '__main__':
    dim = 3
    natom = 500
    steps = 5000
    dt = 0.01
    l = 8.0           #Phi=0.4
    L = [l,l,l]

    mass = np.full((natom),1.0, dtype=float)
    mass[0] = 20

    sigma = np.full((natom), 1.0, dtype=float)
    sig2 = 2.0
    sigma[0] = sig2

    T = 1.0

    fileName = 'test.xyz'
    pos, vel = initialize(dim, natom, L, T, "cube")
    #pos, vel = initialize(dim, natom, L, 1.0, 'basic')
    #vel[0] = 0.0 #for bigger particle
    a = 0; b = np.int(natom/2)
    swap(pos,a,b)
    print "vel sum:", vel.sum()
    md(dim, natom, L, pos, vel, mass, sig2, T, steps, dt, fileName)
