
'''
We consider a scenario in which the independet variable represets the INITIAL COST of a product and 
the dependent variable is the PUBLIC PRICE of the same product.
In order to make it is to verify the results, very simple data are used in this example: every product's intial cost is doubled.
'''

independent_var = [2.5, 8.5, 5.5, 4, 7, 10, 1, 4, 5.5, 5]
independent_mean = 5.3
dependent_var = [5, 17, 11, 8, 14, 20, 2, 8, 11, 10]
dependent_mean = 10.6

#Formula: {SUM[(xi - xM)(yi - yM)] / SUM[(xi - xM)^2]}
def calculateSlope(independent_var, independent_mean, dependent_var, dependent_mean):

	d = []
	x = []
	top = 0
	y = 0 


	#Calculate difference between independent variable and its mean
	for i in range(len(independent_var)):
		x.append(independent_var[i] - independent_mean)

	#Calculate difference between dependent variable and its mean
	for i in range(len(dependent_var)):
		d.append(dependent_var[i]-dependent_mean)

	#Multiply the two differences (Residuals) vectors and add all the results
	for i in range(len(x)):
		top += x[i] * d[i]

	#Sum the squared residuals of the independent varible
	for i in range(len(independent_var)):
		y += (independent_var[i] - independent_mean)**2

	return top/y
pass

#Formula: {A = yM - BxM} where B: slope, A: y intercept
def calculateYintercept(independent_mean, dependent_mean, line_slope):
	return (dependent_mean - (line_slope*independent_mean))
pass


line_slope = calculateSlope(independent_var, independent_mean, dependent_var, dependent_mean)
print("Line's slope: " + str(line_slope))

y_intercept = calculateYintercept(independent_mean, dependent_mean, line_slope)
print("Y intercept:" + str(y_intercept))

print("Linear regression for the two variables: y = " +  str(line_slope) + "x + " + str(y_intercept))

