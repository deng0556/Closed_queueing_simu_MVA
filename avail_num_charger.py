# test for queue length

import numpy as np
import matplotlib.pyplot as plt

num_vehicle = 763 # max num of vehicles
num_station = 60 # number of stations
pro_charging = 1.0/3.0 # probability of charging at FS
num_nodes = num_station * num_station + num_station #num of nodes

lamb = np.zeros(num_nodes + 1, dtype = float) -1
#print(lamb)

#normalization factor 
summation = num_station*pro_charging + num_station + num_station

# number of visits
for i in range(1, num_nodes + 1):
	if i >= 1 and i <= num_station:
		lamb[i] = pro_charging/summation
	elif i >= num_station + 1 and i <= 2 *num_station:
		lamb[i] = 1.0/summation
	elif i >= 2 * num_station + 1 and i <= num_nodes:
		lamb[i] = 1.0/(num_station-1)/summation
print('lambda \n', lamb)

mu = np.zeros(num_nodes + 1, dtype = float) -1
#service rate of one unit
for i in range(1, num_nodes + 1):
	if i >= 1 and i <= num_station:
		mu[i] = 2
	elif i >= num_station + 1 and i <= 2 *num_station:
		mu[i] = 10
	elif i >= 2 * num_station + 1 and i <= num_nodes:
		mu[i] = 3
print('mu \n', mu)

rever_per_service = np.zeros(num_nodes + 1, dtype = float) -1
for i in range(2 *num_station + 1, num_nodes + 1):
	rever_per_service[i] = 30

cost_per_charger = 2

'''
num_server = np.zeros(num_station +1, dtype = int) -1 
for i in range(1, num_station +1):
	num_server[i] = 2
print('num_server \n', num_server)
max_num_server = max (num_server)
#print(max_num_server)
'''

avail_thre = 0.1

avail__ = np.zeros(10, dtype = float)
num_server__= np.zeros(10, dtype = float)
num_server__[0] = 2

for k in range(10):

	num_server = np.zeros(num_station +1, dtype = int) -1 
	for i in range(1, num_station +1):
		num_server[i] = num_server__[k]
	print('num_server \n', num_server)
	max_num_server = max (num_server)
		#print(max_num_server)

	qlength = np.zeros([num_nodes + 1, num_vehicle + 1],  dtype = float) -1
	#queue length
	for i in range(1, num_nodes + 1):
		qlength[i,0] = 0
	#print('qlength \n', qlength)

	waittime = np.zeros([num_nodes + 1, num_vehicle + 1],  dtype = float) -1
	#average time spent at each node

	p = np.zeros((num_station + 1, max_num_server + 1 +1, num_vehicle + 1), dtype = float)
	#marginal distribution of FS : 0 default

	for i in range(1, num_station + 1):
		p[i,0,0] = 1

	throu = np.zeros((num_vehicle + 1),dtype = float) -1

	#update qlenth
	for m in range(1, num_vehicle + 1):
		for i in range(1, num_nodes + 1):
			if i >= 1 and i <= num_station:
				temp_j = 0.0
				for j in range(1, num_server[i]):
					temp_j += (num_server[i] - j) * p[i, j-1, m-1]
				waittime[i,m] = (1 + qlength[i,m-1] +temp_j) \
				/ mu[i] /num_server[i]
			elif i >= num_station + 1 and i <= 2 *num_station:
				waittime[i,m] = (1 + qlength[i,m-1]) / mu[i]
			elif i >= 2 * num_station + 1 and i <= num_nodes:
				waittime[i,m] = 1 / mu[i]

		temp_i = 0
		for i in range(1, num_nodes + 1):
			temp_i += lamb[i] * waittime[i,m]

		throu[m] = m  / temp_i

		for i in range(1, num_nodes + 1):
			qlength[i,m] = throu[m] * lamb[i] * waittime[i,m]

			#print(qlength[i,m])
			#print('p')

			if i >= 1 and i <= num_station:
				for j in range(1, num_server[i]):
					p[i,j,m] = lamb[i] * throu[m] * p[i, j-1, m-1] / j / mu[i]
				temp_jj = 0
				for jj in range(1, num_server[i]):
					temp_jj += (num_server[i] - jj) * p[i, jj, m]
				p[i,0,m] = 1 - (lamb[i] * throu[m] /mu[i] + temp_jj) / num_server[i]
	print('qlength \n', qlength[num_station + 1, num_vehicle])


	avail = np.zeros([num_nodes + 1, num_vehicle + 1],  dtype = float) -1
	for m in range(0, num_vehicle + 1):
		for i in range(1, num_nodes + 1):
			avail[i,m] = lamb[i] * throu[m] /mu[i]
	print('avail', avail[num_station + 1, num_vehicle])


	min_avail = np.zeros(num_vehicle + 1) -1
	min_avail_node = np.zeros(num_vehicle + 1) -1
	#for i in range(num_station + 1, 2 *num_station + 1):
		#print('avail' , avail[i, ]) 

	for m in range(0, num_vehicle + 1):
		for i in range(num_station + 1, 2 *num_station + 1):
			if avail[i,m] > min_avail[m]:
				min_avail_node[m] = i
				min_avail[m] = avail[i,m]
	print('min_avail \n', min_avail[num_vehicle])
	print('min_avail_node \n', min_avail_node[num_vehicle])

	profit = np.zeros(num_vehicle + 1)
	for m in range(0, num_vehicle + 1):
		for i in range(2 *num_station + 1, num_nodes + 1):
			profit[m] += rever_per_service[i] * lamb[i] * throu[m]
		for i in range(1, num_station + 1):
			profit[m] = profit[m]- cost_per_charger * num_server[i]
	print('profit \n', profit[num_vehicle])

	avail__[k] = avail[num_station + 1, num_vehicle]
	if k <= 8:
		num_server__[k+1] = num_server__[k] + 1 

	print('\n')

plt.figure()
plt.plot(num_server__, avail__,'-b',linewidth=5.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Number of Charger per Station',fontsize = 40)
plt.ylabel('Availability',fontsize = 40)
plt.show()