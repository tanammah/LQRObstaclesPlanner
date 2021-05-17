import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import figure as fig
from LQRPlanner import LQRPlanner

# generate map
map_data = np.zeros((20,20))

map_data[5][5] = 1
map_data[5][6] = 1
map_data[6][5] = 1
map_data[6][6] = 1

map_data[9][9] = 1
map_data[9][10] = 1
map_data[10][9] = 1
map_data[10][10] = 1

map_data[4][4] = 1
map_data[7][3] = 1
#map_data[9][5] = 1
map_data[10][3] = 1
map_data[8][6] = 1
map_data[7][6] = 1
map_data[7][5] = 1

###############
"""
map_data[8][5] = 1
map_data[6][4] = 1
map_data[4][5] = 1
map_data[3][6] = 1
map_data[4][7] = 1
map_data[8][6] = 1
map_data[3][3] = 1
map_data[4][3] = 1
map_data[6][3] = 1
map_data[4][6] = 1
"""
continuos_obs = np.zeros((2,2))
continuos_obs[0][0] = 5.5
continuos_obs[1][0] = 8.5
continuos_obs[0][1] = 8
continuos_obs[1][1] = 4.5

A = np.zeros((4,4))
B = np.zeros((4,2))
C = np.zeros((2,4))

A[0][2] = 1
A[1][3] = 1

B[2][0] = 1
B[3][1] = 1

C[0][0] = 1
C[1][1] = 1

planner = LQRPlanner(map_data, A, B, C)
planner.add_obs(continuos_obs)
#x = np.array([5, 8, 0, 0]).reshape((4,1))
#print("lqr obstacles for x is ", planner.LQR_obstacles_test(x, continuos_obs))


start = (1, 1)
goal = (7, 10)
#goal = (7, 6)

path = planner.search(start, goal)

#tracked_traj = planner.track(path)
tracked_traj = planner.approach(start, goal)
print("tracked traj is of length ", len(tracked_traj))

"""
if path is not None:
	print("found path of length ", len(path))
	print("path is ", path)
"""

def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, occupancy_map.shape[1]-1, 0, occupancy_map.shape[0]-1])

def visualize_obstacles(map):
	xs = []
	ys = []

	for i in range(planner.obstacles.shape[1]):
		obs = planner.obstacles[:,i]
		x, y = obs[0], obs[1]
		xs.append(x)
		ys.append(y)

	scat = plt.scatter(xs, ys, c="black", marker='o', s=8)

def visualize_goal(goal):
	scat = plt.scatter([goal[0]], [goal[1]], c="cyan", marker='o', s=2)


def visualize_timestep(loc):
    scat = plt.scatter([loc[0]], [loc[1]], c='r', marker='o', s=2)
    #plt.show()
    #plt.pause(0.00001)
    plt.pause(0.001)
    #scat.remove()

visualize_map(np.zeros((20, 20)))
visualize_goal(goal)
visualize_obstacles(map_data)
for i in range(len(tracked_traj)):
	visualize_timestep(tracked_traj[i])

