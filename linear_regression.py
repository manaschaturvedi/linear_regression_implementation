import numpy as np
import matplotlib.pyplot as plt


X = np.array([0,1,2,3,4,5,6,7,8,9,10])
Y = np.array([0,2,4,6,8,10,12,14,16,18,20])
m = float(len(X))   # number of data points

# initializing theta parameters
theta_0 = 0
theta_1 = 4

# predicting y values using hypothesis/model: theta_0 + theta_1*x
def generate_predicted_values(X,theta_0,theta_1):
	Y_predicted = np.array([])
	for x in X:
		Y_predicted = np.append(Y_predicted,theta_0+(theta_1*x))

	# plotting actual points wrt predicted values
	# plt.scatter(X,Y)
	# plt.plot(X,Y_predicted)
	# plt.show()

	return Y_predicted


def compute_cost_function(Y,Y_predicted):
	m = float(len(Y))
	J = (1/(2*m))*(np.sum((Y_predicted-Y)**2))

	return J

# applying gradient descent
iterations = 100
learning_rate = 0.01
cost_functions = []
theta_ones = []
for _ in range(iterations):
	Y_predicted = generate_predicted_values(X,theta_0,theta_1)
	cost_functions.append(compute_cost_function(Y,Y_predicted))
	theta_0 = theta_0 - (float(learning_rate)*((1/m)*np.sum(Y_predicted-Y)))
	theta_1 = theta_1 - (float(learning_rate)*((1/m)*np.sum((Y_predicted-Y)*X)))
	theta_ones.append(theta_1)

print('After gradient descent-\n','theta_0:',theta_0,'\ntheta_1:',theta_1,'\nCost function:',cost_functions[-1])

# plotting cost function wrt number of iterations in gradient descent
plt.plot(range(iterations),cost_functions)
plt.show()