import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

import random

# set random seed, generate some example data 
np.random.seed(1989)
randomX = np.random.rand(10, 3) 

true_weights = np.array([2.0, -1.5, 0.5])
true_bias = 1.0

randomY = np.dot(randomX,true_weights) + true_bias 

# add some variability
randomY += np.random.normal(0, 0.1, size=randomY.shape)

# combine into one df - probably an unnecessary step, but treating this as if starting with a df to begin with
df = pd.DataFrame(randomX, columns=['X1', 'X2', 'X3'])
df['Target'] = randomY

################################

# Separate target and features
X = df.drop(columns=['Target']).copy()
Y = df['Target'].copy()

# Standardizing 
Y = np.array((Y-Y.mean())/Y.std())
# Standardizing X is more difficult, bc its shape [n,X], where X=number of features
X = X.apply(lambda rec:(rec-rec.mean())/rec.std(),axis=0)

###########################
# Initialize bias and weights
def initialize(dimensions): 
    np.random.seed(1989)
    b = np.random.rand(1) 
    theta = np.random.rand(dimensions) 
    return b, theta 

b, theta = initialize(X.shape[1])
print(
    f"Initialized: \n"
    f"  Bias: {b} \n"
    f"  Weights: {theta} \n"
    f"Actuals: \n"
    f"  Bias: {true_bias} \n"
    f"  Weights: {true_weights}"
)

###########################
# Forward pass example
def predict_Y(b, theta, X):
    # use initialized bias and weights (plus X) to predict Y
    Y_hat = b + np.dot(X, theta)
    return Y_hat

Y_hat = predict_Y(b, theta, X) 

###########################
# Create cost function
def get_cost(Y, Y_hat):
    # residuals
    Y_resd = Y_hat - Y
    # Mean Squared Error
    MSE = np.dot(Y_resd.T, Y_resd)/len(Y)
    return MSE

MSE = get_cost(Y, Y_hat)
print(f"MSE: {MSE}")

###########################
# Backpropogation to get more accurate weights and bias 
def update_theta(x, y, y_hat, b_0, theta_0, learning_rate):

    # get residuals again
    Y_resd = Y_hat - Y

    # derivative of bias is easy
    db = np.mean(Y_resd)
    # update bias
    b_1 = b_0 + (learning_rate*(-db)) 

    # derivative of weights is harder, 
    # but there is no activation function so still pretty straightforward...
    dw = (np.dot(x.T,Y_resd))/len(y) 
    
    # update weights
    theta_1 = theta_0 + (learning_rate*(-dw))

    return b_1, theta_1

b, theta = update_theta(X, Y, Y_hat, b, theta, learning_rate=0.01)
new_MSE = get_cost(Y=Y, Y_hat=predict_Y(b, theta, X))
print(
    f"After first update: \n"
    f"  Bias: {b} \n"
    f"  Weights: {theta} \n"
    f"  New Cost: {new_MSE} \n"
    f"Actuals: \n"
    f"  Bias: {true_bias} \n"
    f"  Weights: {true_weights}"
)


##################################
# Now just make it a loop!
def run_gradient_descent(X, Y, alpha, num_iterations):

    # use initialize function from earlier
    b, theta = initialize(X.shape[1]) 
    cost_log = []

    # for loop
    for each_iter in range(num_iterations):

        # use predict function to get predictions from w and b
        Y_hat = predict_Y(b, theta, X)

        # calculate cost 
        this_cost = get_cost(Y, Y_hat)

        # update weights and bias with backprop
        b, theta = update_theta(X, Y, Y_hat, b, theta, alpha)

        # keep track of the cost
        cost_log.append({'iteration': each_iter, 'cost': this_cost})

    # turn cost log into df so i can plot it later 
    gd_iterations_df = pd.DataFrame(cost_log)
    
    print(
        f"Number of iterations: {num_iterations} \n"
        f"alpha: {alpha} \n"
        f"Final Estimate of b and w: {b, theta} \n"
    )

    return gd_iterations_df, b, theta

# run it!
gd_iterations_df, b, theta = run_gradient_descent(
                                X,
                                Y,
                                alpha = 0.001,
                                num_iterations = 200)

gd_iterations_df.head()

###############################
# Plot cost function over gd iterations
plt.figure()
plt.plot(gd_iterations_df['iteration'], gd_iterations_df['cost'])
plt.xlabel('Number of iterations')
plt.ylabel('Cost or MSE')
plt.show()


##############################
# make custom grid search, just for fun
alpha_grid = [0.0001, 0.001, 0.005, 0.010]

plt.figure(figsize=(10,6))
for a in alpha_grid:
    # run gradient descent
    cost_df, b, theta = run_gradient_descent(X, Y, alpha = a, num_iterations = 150)
    # plot results
    plt.plot(cost_df['iteration'], cost_df['cost'], label=f'alpha={a}')
plt.legend()
plt.ylabel('Cost or MSE')
plt.xlabel('Number of iterations')
plt.title('Cost vs. Iterations for different alpha values')
plt.savefig("Quick Gradient Descent Example/figs/gradient_descent_for_alpha_levels.png")
plt.show()