# test for queue length

import numpy as np
import matplotlib.pyplot as plt

num_vehicle = 40 # max num of vehicles
num_station = 3 # number of stations
pro_charging = 1.0/3.0 # probability of charging at FS
num_nodes = num_station * num_station + num_station #num of nodes

lamb = np.zeros(num_nodes + 1, dtype = float) -1
#print(lamb)

# number of visits
for i in range(1, num_nodes + 1):
	if i >= 1 and i < 2:
		lamb[i] = 3.0/56.0
	elif i >= 2 and i <= 3:
		lamb[i] = 5.0/112.0
	elif i >= 4 and i < 5:
		lamb[i] = 9.0/56.0
	elif i >= 5 and i <= 6:
		lamb[i] = 15.0/112.0
	elif i >= 7 and i <= 10:
		lamb[i] = 9.0/112.0
	elif i >= 11 and i <= 12:
		lamb[i] = 3.0/56.0
print('lambda \n', lamb)

num_server = np.zeros(num_station +1, dtype = int) -1 
num_server[1] = 3
for i in range(2, num_station +1):
	num_server[i] = 2
print('num_server \n', num_server)
max_num_server = max (num_server)

mu = np.zeros([num_nodes + 1, num_vehicle + 1], dtype = float) -1
#service rate 
for i in range(1, num_nodes + 1):
	for j in range(1, num_vehicle + 1):
		if i >= 1 and i <= num_station:
			mu[i, j] = 2 * min(num_server[i],j)
		elif i >= num_station + 1 and i <= 2 *num_station:
			mu[i, j] = 10
		elif i >= 2 * num_station + 1 and i <= num_nodes:
			mu[i, j] = 3 * j
print('mu \n', mu)


f = np.zeros([num_nodes + 1, num_vehicle + 1], dtype = float) -1
for i in range(1, num_nodes + 1):
	f[i, 0] = 1
	for j in range(1, num_vehicle + 1):
		product = 1 
		for k in range(1, j + 1):
			product *= mu[i, k]
		f[i, j]= lamb[i] ** j / product
print('f\n', f)

g = f[1]
print('g\n',g)

for l in range(2, num_nodes + 1):
	g = np.convolve(g, f[l])
	#print(g) 

print()
g_m_1 = g[num_vehicle - 1]
g_m = g[num_vehicle]


'''
#g_without_FS1
g_wo_fs = f[2]
for l in range(3, num_nodes + 1):
	g_wo_fs = np.convolve(g_wo_fs, f[l])
	#print(g) 

p_fs = np.zeros(num_vehicle + 1) - 1
mean_fs = 0
for j in range(0, num_vehicle + 1):
	p_fs[j] = f[1, j] * g_wo_fs[num_vehicle - j] / g_m
	mean_fs += p_fs[j] * j

print('p_fs\n', p_fs)
print('mean_fs', mean_fs)

p_fss = np.zeros(16) - 1
for j in range(0, 16):
	p_fss[j] =p_fs[j]


plt.figure()
plt.plot(range(0, 16), p_fss,'-b',markersize=5.0,linewidth=5.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.xlabel('n: Number of Vehicles at FS1', fontsize = 40)
plt.ylabel('P(n): Probability', fontsize = 40)
plt.show()
'''


#g_without_SS1
g_wo_ss = f[1]
for l in range(2, num_station + 1):
	g_wo_ss = np.convolve(g_wo_ss, f[l])

for l in range(num_station + 2, num_nodes + 1):
	g_wo_ss = np.convolve(g_wo_ss, f[l])


p_ss = np.zeros(num_vehicle + 1) - 1
mean_ss = 0
for j in range(0, num_vehicle + 1):
	p_ss[j] = f[num_station + 1, j] * g_wo_ss[num_vehicle - j] / g_m
	mean_ss += p_ss[j] * j

print('mean_ss', mean_ss)




plt.figure()
plt.plot(range(0, num_vehicle + 1), p_ss,'-b',markersize=5.0,linewidth=5.0)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('n: Number of Vehicles at SS1', fontsize = 40)
plt.ylabel('P(n): Probability', fontsize = 40)
plt.show()


'''
#g_without_IS
g_wo_is = f[1]
for l in range(2, num_nodes + 1):
	if l == 2 * num_station + 1 :
		g_wo_is = g_wo_is
	else:
		g_wo_is = np.convolve(g_wo_is, f[l])
	#print(g) 

p_is = np.zeros(num_vehicle + 1) - 1
mean_is = 0
for j in range(0, num_vehicle + 1):
	p_is[j] = f[2 * num_station + 1, j] * g_wo_is[num_vehicle - j] / g_m
	mean_is += p_is[j] * j

print('mean_is', mean_is)

plt.figure()
plt.plot(range(0, num_vehicle + 1), p_is)
plt.xlabel('$n_i$, infinite server node', fontsize = 20)
plt.ylabel('$P(n_i)$, Prob of $n_i$ vehicles at IS', fontsize = 20)
plt.show()
'''
