# Basic linear regression implementation in Python

We start off by defining the values in X and Y-axis by storing them inside two variables:

```
X = np.array([0,1,2,3,4,5,6,7,8,9,10])
Y = np.array([0,2,4,6,8,10,12,14,16,18,20])
```
We have also initialized our parameter value `theta_1` to initially be equal to 4:

```
# initializing theta parameters
theta_0 = 0
theta_1 = 4
```

Based on the initial values of parameters, this is how our best fit line looks like when plotted against the actual values of X and Y-axis:
![alt text](https://github.com/manaschaturvedi/linear_regression_implementation/blob/master/plot_xy_1.png)

Now, we will run the Gradient Descent algorithm to find the optimal values of our parameters `theta_0` and `theta_1` by running the algorithm 10 times and using learning rate's value to be 0.01:

```
iterations = 10
learning_rate = 0.01
```

Now, at each iteration of running Gradient Descent, we will perform the following tasks:
- predict values for our label (Y) using our hypothesis/model: `theta_0 + theta_1*x`
- compute cost function using the actual and our predicted values of Y
- simultaneously update the values of `theta_0` and `theta_1`

```
for _ in range(iterations):
	Y_predicted = generate_predicted_values(X,theta_0,theta_1)
	cost_functions.append(compute_cost_function(Y,Y_predicted))
	theta_0 = theta_0 - (float(learning_rate)*((1/m)*np.sum(Y_predicted-Y)))
	theta_1 = theta_1 - (float(learning_rate)*((1/m)*np.sum((Y_predicted-Y)*X)))
```

After running Gradient Descent 10 times and updating the values of `theta_0` and `theta_1` at each iteration, we can compare the values of our parameters and the corresponding cost functions before and after running Gradient Descent:

```
Before Gradient Descent:
theta_0 = 0
theta_1 = 4
cost function: 70.0

After Gradient Descent-
theta_0: -0.2709674247575545 
theta_1: 2.0631089174216615 
cost function: 0.03538502177709652
```
The final plot of our best fit line after finding the values of our parameters using Gradient Descent:
![alt text](https://github.com/manaschaturvedi/linear_regression_implementation/blob/master/plot_xy_10.png)

Plotting the cost function wrt number of iterations in Gradient Descent to verify whether the cost function of our model decreases with each iteration:
![alt text](https://github.com/manaschaturvedi/linear_regression_implementation/blob/master/cost_plot.png)
