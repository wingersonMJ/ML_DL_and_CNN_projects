# Simple gradient descent for network without activation function:

The example Python script demonstrates the proccess of gradient descent, from weight initializations and forward passes, through the derivative calculations and weight updates 
that make up backpropogation. I end the example by iterating through gradient descent for 150 iteration, with varying learning rates (0.0001 to 0.010). The resulting cost vs 
iteration figure is shown below. This approach did not include an activation function, thus derivative calculations are fairly straightforward.  
<br>

**Figure.** Cost (mean squared error in this example) vs iterations plot, with a line for various learning rates (alphas). 
<img src = "./figs/gradient_descent_for_alpha_levels.png" size=400>