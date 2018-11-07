import math

a_array = [(10,1),(9,6),(3,2),(7,4),(8,5)]
p_array = [(8,3),(7,4),(5,9),(2,10),(6,1)]

def euclidean_distance(a_array, p_array):
	result = 0
	for i in range(len(a_array)):
		x_1 = a_array[i][0]
		x_2 = p_array[i][0]
		y_1 = a_array[i][1]
		y_2 = p_array[i][1]
		result += (x_1 - x_2)**2 + (y_1 - y_2)**2

	return math.sqrt(result) #math.sqrt performs the square root of the argument
pass

print(euclidean_distance(a_array,p_array))