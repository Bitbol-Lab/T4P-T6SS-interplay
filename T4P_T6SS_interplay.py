#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from random import*
from itertools import product
import numba as nb
from numba import njit
from numba.np.unsafe.ndarray import to_fixed_tuple
import copy


# Initialize the system
def initialization():
    """
    Returns
    -------
    grid: NumPy array
        One entry corresponds to one site of the lattice
    """

    grid = np.zeros((L, L, L))

    # We uniformly choose at random the coordinates of the n initial particles:
    initial_coodinates = sample([(i1, i2, i3) for i1 in range(0, L) for i2 in range(0, L) for i3 in range(0, L) if grid[i1][i2][i3] == 0], n)

    for particle in range(n):
        if particle < n_prey:
            type = 1 # Prey are denoted by 1
        else:
            type = 2 # Predators are denoted by 2
        grid[initial_coodinates[particle][0]][initial_coodinates[particle][1]][initial_coodinates[particle][2]] = type

    return grid






# Compute the size of aggregates
@njit
def identify_agregate(grid, particle_type, start_pos, E_cross, vis):
    """
    Returns
    -------
    connected_particles: set
        set of particles forming an aggregate
    """

    # Initialize a set to store the connected particles' coordinates
    connected_particles = set()

    # We put in queue the sites whose neighbors still need to be inspected
    queue = [to_fixed_tuple(np.array([start_pos[0], start_pos[1], start_pos[2]]), 3)]

    # Particles are associated with a number from which we can recover their coordinates:
    connected_particles.add(queue[0][0] * (L+1) ** 2 + queue[0][1] * (L+1) + queue[0][2])

    while len(queue) > 0: # The loop keeps going until there is no more sites to inspect

        # We inspect the neighbors
        for d in range(8):

            neighbor_coord_1, neighbor_coord_2, neighbor_coord_3 = (queue[0][0]+move_list[d][0])%L, (queue[0][1]+move_list[d][1])%L, (queue[0][2]+move_list[d][2])%L
            neighbor_type = abs(grid[neighbor_coord_1][neighbor_coord_2][neighbor_coord_3])

            # If E_cross = 0, this function yields all the connected particles OF SAME TYPE than the particle we put in argument
            # If E_cross > 0, this function yields all the particles connected to the particle we put in the arguments
            # If E_prey = 0 (we assume E_prey = E_predator), in which case we always set E_cross = 0 and there is no aggregate
            if (neighbor_type  == abs(particle_type) and E_prey != 0) or (neighbor_type != 0 and (E_cross != 0)):

                # 'vis' lists the sites we already inspected in the past
                if neighbor_coord_1*(L+1)**2 + neighbor_coord_2*(L+1) + neighbor_coord_3 not in vis and neighbor_coord_1*(L+1)**2 + neighbor_coord_2*(L+1) + neighbor_coord_3 not in connected_particles:
                    connected_particles.add(neighbor_coord_1*(L+1)**2 + neighbor_coord_2*(L+1) + neighbor_coord_3)
                    queue.append(to_fixed_tuple(np.array([neighbor_coord_1, neighbor_coord_2, neighbor_coord_3]), 3))


        queue.remove(queue[0]) # Once inspected, the considered site is removed from the queue

    return connected_particles



@njit
def aggregated_particles_detection(grid, particle_type, start_pos, moving_particles, directions, main_aggregate):
    """
    Returns
    -------
    new_moving_particles: list of list
        list of lists (1 per direction) of all the particles that move if the considered aggregate moves in a given direction
    """
    
    # "moving_particles[i]" lists all the particles that move in case the aggregate (the original particle we've drawn is in) 
    # move in the direction i
    new_moving_particles = moving_particles.copy() # Given that we recursively call "aggregated_particles_detection",
                                                   # this avoids unexpected memory-related issues

    # "aggregate" lists all the particles forming the aggregate the particle of coordinates "start_pos" is in
    aggregate = identify_agregate(grid, particle_type, start_pos, E_cross, new_moving_particles[directions[-1]])

    for m in directions: # Update "new_moving_particles"
        new_moving_particles[m] += list(aggregate)

    if E_cross != 0: # If E_cross > 0, all the connected particles form a single aggregate
        return new_moving_particles

    if main_aggregate: # main_aggregate is True if "aggregate" only lists the particles forming the aggregate the drawn particle is part of
                       # (meaning that we didn't look at the surroundings yet)
        for pos in aggregate: # We look at the neighbors of the particles of the aggregate in order to inspect the surroundings
            neighbors = [to_fixed_tuple(np.array([(pos // (L + 1) ** 2 + m[0]) % L, ((pos % (L + 1) ** 2) // (L + 1) + m[1]) % L, (pos % (L + 1) + m[2]) % L]), 3) for m in move_list]

            for ind in range(8):
                n_1, n_2, n_3 = neighbors[ind][0], neighbors[ind][1], neighbors[ind][2]
                state = grid[n_1][n_2][n_3]
                if state != 0:
                    if n_1*(L+1)**2 + n_2*(L+1) + n_3 not in aggregate and n_1*(L+1)**2 + n_2*(L+1) + n_3 not in new_moving_particles[ind]: # If not already considered
                        new_moving_particles = aggregated_particles_detection(grid, int(grid[n_1][n_2][n_3]), np.array([n_1, n_2, n_3]), new_moving_particles, np.array([ind]), False)

    else:
        for pos in aggregate:  # For each element of the aggregate, we inspect its surroundings in a given direction given in the arguments
            coord1, coord2, coord3 = to_fixed_tuple(np.array([(pos // (L + 1) ** 2 + move_list[directions[0]][0]) % L, ((pos % (L + 1) ** 2) // (L + 1) + move_list[directions[0]][1]) % L, (pos % (L + 1) + move_list[directions[0]][2]) % L]),3)
            site_state = grid[coord1][coord2][coord3]
            if site_state != 0:  # if not empty
                if coord1 * (L + 1) ** 2 + coord2 * (L + 1) + coord3 not in new_moving_particles[directions[0]]:  # If not already considered
                    new_moving_particles = aggregated_particles_detection(grid, site_state, np.array([coord1, coord2, coord3]), new_moving_particles, directions, False)

    return new_moving_particles

# Main
@njit
def main(initial_state_of_the_lattice):
    """
    Parameters
    ----------

    initial_state_of_the_lattice: Numpy array
        An entry corresponds to a site of the lattice. a 1 entry denotes a prey, a 2 a predator and a -1 a lysing prey

    k_kill, k_d, k_hop, k_lysis, E_prey, E_predator, E_cross: floats
        See the comments below

    L: Integer
        size of the system (length of an edge of the rhombohedral lattice)
    
    Returns
    -------

    number_of_prey: list
        Successive numbers of prey over time

    number_of_predator: list
        Successive numbers of predators over time

    number_of_lysing_cells: list
        Successive numbers of lysing prey over time

    times: list
        Values of time
    """


    grid_state = initial_state_of_the_lattice


    t = 0
    number_of_preys = [len(np.where(grid_state == 1)[0])] # The 1 entries represent prey
    number_of_predators = [len(np.where(grid_state == 2)[0])] # The 2 entries represent predators
    number_of_lysing_prey = [len(np.where(grid_state == -1)[0])] # The -1 entries represent lysing prey
    times = [0.0]


    while t <= tmax:

        occupied_sites = list(zip(*np.where(grid_state != 0)))
        random_site = occupied_sites[np.random.randint(len(occupied_sites))] # We draw a site directly among the occupied ones
                                                                             # where events can occur to speed up the simulation

        particle_type = grid_state[random_site[0]][random_site[1]][random_site[2]]

        free_neighboring_sites =  np.array([dir for dir in range(8) if grid_state[(random_site[0] + move_list[dir][0]) % L][(random_site[1] + move_list[dir][1]) % L][(random_site[2] + move_list[dir][2]) % L] == 0])
        number_of_free_neighbors = len(free_neighboring_sites)
        occupied_neighbors = [[(random_site[0] + move_list[dir][0]) % L, (random_site[1] + move_list[dir][1]) % L, (random_site[2] + move_list[dir][2]) % L] for dir in range(8) if grid_state[(random_site[0] + move_list[dir][0]) % L][(random_site[1] + move_list[dir][1]) % L][(random_site[2] + move_list[dir][2]) % L] != 0]


        # Determination of the rates of the events that can potentially occur in the site we've drawn
        # The multiplication by L^3 accounts for the fact that, if we randomly draw a site and then try to make an event x to occur there,
        # the overall probability is k_x and the probability given that we draw the right site k_x*L^3
        if particle_type == 1:
            energy, killing_proba, lysis_proba = E_prey, k_kill*L**3, 0
            same_type_neighbors = np.sum(np.array([1 for n in occupied_neighbors if abs(grid_state[n[0]][n[1]][n[2]]) == 1]))
            neighboring_predators = np.sum(np.array([1 for n in occupied_neighbors if grid_state[n[0]][n[1]][n[2]] == 2]))
        elif particle_type ==  - 1:
            energy, killing_proba, lysis_proba = E_prey, 0, k_lysis*L**3
            same_type_neighbors = np.sum(np.array([1 for n in occupied_neighbors if abs(grid_state[n[0]][n[1]][n[2]]) == 1]))
            neighboring_predators = np.sum(np.array([1 for n in occupied_neighbors if grid_state[n[0]][n[1]][n[2]] == 2]))
        elif particle_type == 2:
            energy, killing_proba, lysis_proba = E_predator, 0, 0
            same_type_neighbors = np.sum(np.array([1 for n in occupied_neighbors if abs(grid_state[n[0]][n[1]][n[2]]) == 2]))
            neighboring_predators = 0
        if number_of_free_neighbors > 0:
            hopping_proba = number_of_free_neighbors*k_hop*L**3*np.exp(-energy*same_type_neighbors - E_cross*(8-number_of_free_neighbors-same_type_neighbors))
            if particle_type > 0:
                division = k_d*L**3
            else:
                division = 0
        else:
            hopping_proba = 0
            division = 0

        random_event = np.random.random() # Random number used to determine the occurrence and type of event

        # Lysis
        if  random_event <= lysis_proba:
            grid_state[random_site[0]][random_site[1]][random_site[2]] = 0




        # Diffusion
        if lysis_proba < random_event <= lysis_proba + hopping_proba:
            random_direction = np.random.choice(free_neighboring_sites)
            new_coordinates = [(random_site[0] + move_list[random_direction][0]) % L, (random_site[1] + move_list[random_direction][1]) % L, (random_site[2] + move_list[random_direction][2]) % L]
            grid_state[new_coordinates[0]][new_coordinates[1]][new_coordinates[2]] = particle_type
            grid_state[random_site[0]][random_site[1]][random_site[2]] = 0



        # Division
        elif lysis_proba + hopping_proba < random_event <= lysis_proba + division + hopping_proba:
            random_direction = np.random.choice(free_neighboring_sites)
            new_coordinates = [(random_site[0] + move_list[random_direction][0]) % L, (random_site[1] + move_list[random_direction][1]) % L, (random_site[2] + move_list[random_direction][2]) % L]
            grid_state[new_coordinates[0]][new_coordinates[1]][new_coordinates[2]] = particle_type



        # killing
        elif lysis_proba + division + hopping_proba < random_event <= lysis_proba + division + hopping_proba + killing_proba * neighboring_predators:
            grid_state[random_site[0]][random_site[1]][random_site[2]] = -1 # Lysing cells are denoted by -1 entries



        # Aggregate diffusion
        elif number_of_free_neighbors < 8: # If there is no aggregate and no surrounding particles/aggregates to push, we don't enter this loop

            # We define empty lists to avoid issues with Numba
            overall_connected_particles = [[nb.float32(0)] for _ in range(8)]
            for m in range(8):
                overall_connected_particles[m].pop()

            # "moving_particles_list[i]" lists all the particles that move in case the aggregate (the original particle we've drawn is in)
            # move in the direction i
            moving_particles_list = [[nb.float32(0)]]
            moving_particles_list.pop()

            moving_particles_list += aggregated_particles_detection(grid_state, particle_type, random_site, overall_connected_particles, np.arange(8), True)


            aggregate_hopping_rates = np.zeros(8)
            for ele in range(len(moving_particles_list)):
                if len(moving_particles_list[ele]) > 1:
                    aggregate_hopping_rates[ele] += k_hop * L ** 3 / len(moving_particles_list[7-ele]) # We divide by the number of particles
                                                                                             # which can be drawn to make the aggregate move,
                                                                                             # namely the particles of the aggregate itself and the ones that can push it


            if  lysis_proba + division + hopping_proba + killing_proba * neighboring_predators < random_event <= lysis_proba + division + hopping_proba + killing_proba * neighboring_predators + np.sum(aggregate_hopping_rates):

                cumulative_dist = np.cumsum(aggregate_hopping_rates/np.sum(aggregate_hopping_rates))

                random_direction2 = np.random.random()
                choice = np.argmax(random_direction2 < cumulative_dist)

                coordinates = [[int(particle // (L+1)**2), int((particle % (L+1)**2) // (L+1)), int(particle % (L+1))] for particle in moving_particles_list[choice]]
                new_coordinates_list = [[(coordinates[particle][0] + move_list[choice][0]) % L, (coordinates[particle][1] + move_list[choice][1]) % L, (coordinates[particle][2] + move_list[choice][2]) % L] for particle in range(len(coordinates))]

                all_types = []
                # We remove the particles that move and relocate them
                for site in coordinates:
                    all_types.append(int(grid_state[site[0]][site[1]][site[2]]))
                    grid_state[site[0]][site[1]][site[2]] = 0
                for particle in range(len(coordinates)):
                    grid_state[new_coordinates_list[particle][0]][new_coordinates_list[particle][1]][new_coordinates_list[particle][2]] = all_types[particle]

        if t <= tmax:

            # Time is updated based on the mean number of waiting iterations for drawing an occupied site
            # when selecting sites uniformly at random, regardless of occupation status
            t += 1 * L**3/len(occupied_sites)


            if t in recorded_time_points or t > recorded_time_points[len(times)+1]:

                number_of_preys += [len(np.where(grid_state == 1)[0])]
                number_of_predators += [len(np.where(grid_state == 2)[0])]
                number_of_lysing_prey += [len(np.where(grid_state == -1)[0])]
                times.append(t)


    return  number_of_preys, number_of_predators, number_of_lysing_prey, times



if __name__ == "__main__":
    # Working example:

    # Arguments
    L = 40 # With a body-centred cubic (bcc) lattice, the system is a rhombohedron with edges of size L
    n = 100 # Initial number of particles (bacteria)
    n_prey = int(n/2) # Initial number of prey among the n particles

    R = (3/(4*np.pi)*0.692)**(1/3)*10**-6 # Radius of a particle
    D = 10**-11*0.3 # Diffusion coefficient
    k_hop = 3*D / (4 * (2 * R) ** 2) # Hopping rate

    E_cross = 3 # Binding energy between a prey and a predator
    E_prey = 3 # Binding energy between 2 preys
    E_predator = 3 # Binding energy between 2 predators

    k_d = 1.58 / 3600 # Division rate

    k_kill = 50 / 3600 / 8
    k_lysis = 1 / (75 * 60)

    # Here we set the length of the time window dt corresponding to one iteration in our kinetic MC algorithm:
    c = (k_d + 8 * k_hop) * L ** 3 / 0.99 # Given that during an iteration we draw an isolated lysing cell,
                                     # we set to 0.99 the probability for an event to occur there

    k_d /= c
    k_kill /= c
    k_lysis /= c
    k_hop /= c



    # Given the primitive translation vectors of the bcc lattice, "move_list" gives, from a site (0, 0, 0), where are its neighbors
    move_list = np.array([item for item in product([-1, 0, 1], repeat=3) if not str(item).count('0') in [1, 3] and not (str(item).count('0') == 0 and abs(np.sum(item)) != 3)])
    all_coordinates = np.array([list(ele) for ele in list(product(range(L), repeat=3))]) # array of the coordinates of all the sites of the grid

    tmax = c*600 # Duration of the simulation (in "real" time units, not in number of iterations)
    recorded_time_points = np.array([int(i) for i in np.linspace(0, tmax, 10)])

    # Simulation
    simulation = main(initialization())

    # Outcome
    print("Number of prey over time = " + str(simulation[0]))
    print("Number of predators over time = " + str(simulation[1]))
    print("Number of lysing prey over time = " + str(simulation[2]))
    print("Time (min) = " + str(np.array(simulation[3])/c/60))
