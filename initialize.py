from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time


def basic_init(dim, natom, L):
    sep = 1.5
    off = 0.1
    l = L[0]
    line = np.int(np.floor((l-2*off)/sep))
    if natom > line**3:
        print "Too many atoms for this initialization"
        quit()

    pos = np.zeros((natom,3))
    vel = np.zeros((natom,3))
    n = 0
    x=off; y=off; z=off
    dim = 0
    while n < natom:
        pos[n,:] = np.array([x,y,z])
        x += sep
        if x > l:
            y += sep
            x = off
        if y > l:
            z += sep
            y = off
        n += 1
    return pos, vel

def initialize_random(dim, natom, L, minDist):
    #Randomly puts particles in box of side length L and checks to make sure
    #they aren't overlapping within some distance 'minDist'
    pos = np.random.rand(natom, dim)*L[0]
    vel = np.zeros((natom, dim))

    overlapAtoms = Overlap(dim, natom, pos, minDist)
    nOverlap = len(overlapAtoms)

    count = 0
    maxCount = 1000
    while nOverlap > 0:
        count+=1
        print "Count:", count, "Overlap:", nOverlap
        if count % maxCount == 0:
            print "After", count, "iterations, still have", nOverlap, \
            "overlapping pairs of atoms; try fewer particles or a bigger box"
        for i in overlapAtoms:
            pos[i,:] = np.random.rand(dim)*L[0]
        overlapAtoms = Overlap(dim, natom, pos, minDist)
        nOverlap = len(overlapAtoms)

    np.savetxt('test_randomInit_pos.dat',pos)
    np.savetxt('test_randomInit_vel.dat',vel)
    return pos, vel

def initialize_random(dim, natom, L, minDist, T):
    #Randomly puts particles in box of side length L and checks to make sure
    #they aren't overlapping within some distance 'minDist'
    pos = np.random.rand(natom, dim)*L[0]
    vel = np.zeros((natom, dim))

    overlapAtoms = Overlap(dim, natom, pos, minDist)
    nOverlap = len(overlapAtoms)

    count = 0
    maxCount = 1000
    while nOverlap > 0:
        count+=1
        print "Count:", count, "Overlap:", nOverlap
        if count % maxCount == 0:
            print "After", count, "iterations, still have", nOverlap, \
            "overlapping pairs of atoms; try fewer particles or a bigger box"
        for i in overlapAtoms:
            pos[i,:] = np.random.rand(dim)*L[0]
        overlapAtoms = Overlap(dim, natom, pos, minDist)
        nOverlap = len(overlapAtoms)

    #Call function to "thermalize atoms"--give random velocities accoring to
    #Boltzmann distribution, and move to COM frame to negate overall flow
    vel = mxwl(dim, natom, vel, T)

    np.savetxt('test_randomInit_pos.dat',pos)
    np.savetxt('test_randomInit_vel.dat',vel)
    return pos, vel

def initialize_square(natom, L, minDist):
    #Puts atoms into a square box of side length L as a square lattice
    pos = np.zeros((natom,2))
    vel = np.zeros((natom,2))

    gridSize = int(np.ceil(np.sqrt(natom)))
    print "L:", L[0]
    print "GridSize:", gridSize
    gridSpacing = L[0]/float(gridSize)
    print "GridSpacing:", gridSpacing
    if gridSpacing < minDist:
        print "Too many particle for box size. Use less atoms or bigger box"

    startPos = gridSpacing/2.0
    atnum = 0
    for i in range(gridSize):
        for j in range(gridSize):
            if atnum < natom:
                x = startPos + i*gridSpacing
                y = startPos + j*gridSpacing
                pos[atnum,0] = x
                pos[atnum,1] = y
                atnum += 1
    #Call function to "thermalize atoms"--give random velocities accoring to
    #Boltzmann distribution, and move to COM frame to negate overall flow

    np.savetxt('test_randomSq_pos.dat',pos)
    np.savetxt('test_randomSq_vel.dat',vel)
    return pos, vel

def initialize_cube(natom, L, minDist):
    #Puts atoms into a cubic box of side length L as a square lattice
    pos = np.zeros((natom,3))
    vel = np.zeros((natom,3))

    gridSize = int(np.ceil(float(natom)**(1.0/3.0)))
    print "L:", L[0]
    print "GridSize:", gridSize
    gridSpacing = L[0]/float(gridSize)
    print "GridSpacing:", gridSpacing
    if gridSpacing < minDist:
        print "Too many particle for box size. Use less atoms or bigger box"

    startPos = gridSpacing/2.0
    atnum = 0
    for i in range(gridSize):
        for j in range(gridSize):
            for k in range(gridSize):
                if atnum < natom:
                    x = startPos + i*gridSpacing
                    y = startPos + j*gridSpacing
                    z = startPos + k*gridSpacing
                    pos[atnum,:] = np.array([x,y,z])
                    atnum += 1

    np.savetxt('test_randomCube_pos.dat',pos)
    np.savetxt('test_randomCube_vel.dat',vel)
    return pos, vel

def initialize_cube(natom, L, minDist, T):
    #Puts atoms into a cubic box of side length L as a square lattice
    pos = np.zeros((natom,3))
    vel = np.zeros((natom,3))

    gridSize = int(np.ceil(float(natom)**(1.0/3.0)))
    print "L:", L[0]
    print "GridSize:", gridSize
    gridSpacing = L[0]/float(gridSize)
    print "GridSpacing:", gridSpacing
    if gridSpacing < minDist:
        print "Too many particle for box size. Use less atoms or bigger box"

    startPos = gridSpacing/2.0
    atnum = 0
    for i in range(gridSize):
        for j in range(gridSize):
            for k in range(gridSize):
                if atnum < natom:
                    x = startPos + k*gridSpacing
                    y = startPos + j*gridSpacing
                    z = startPos + i*gridSpacing
                    pos[atnum,:] = np.array([x,y,z])
                    atnum += 1

    #Call function to "thermalize atoms"--give random velocities accoring to
    #Boltzmann distribution, and move to COM frame to negate overall flow
    vel = mxwl(3, natom, vel, T)

    np.savetxt('test_randomCube_pos.dat',pos)
    np.savetxt('test_randomCube_vel.dat',vel)
    return pos, vel

def Overlap(dim, natom, pos, minDist):
    #For use in the random initialization
    #Returns an array of particle numbers which are part of an overlapping pair
    overlap = []
    for i in range(natom):
        for j in range(i):
            rij = pos[i,:] - pos[j,:]
            d2rij = (rij*rij).sum()
            drij = np.sqrt(d2rij)

            #If i and j overlap in any dimension, note i and move on
            #to next particle--break
            if drij < minDist:
                overlap.append(i)
                #print "Break"
                #time.sleep(5)
                break
    overlap = np.asarray(overlap)
    return overlap

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
    vcm = vel.sum(axis=0)/natom #1d array with [vcm_x,vcm_y,vcm_z]
    #Move to COM frame to negate any inadvertent flow
    vel = vel - vcm
    return vel

if __name__ == '__main__':
    l = 50
    L = [l,l,l]
    #pos, vel = initialize_cube(1000, L, 1.2)
    pos, vel = initialize_random(3,1000,L,1.2)
    print pos
