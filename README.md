# Classification_using_CNN
Experimenting several ways to improve the accuracy of the CNN using CIFAR dataset. 

Modifying the existing code given at - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

1 - Running the training longer
Before the for loop runs for 2 loops. Increasing the for loop for 12 loops increases the accuracy to
60% and the loss decreased from 1.7 to 0.7. But the loss also increases in the middle and then again 
decreases. Increasing the for loop iterations to 22 loops, there is no change in accuracy. The 
accuracy remains constant at 60% and the loss decreases from 2.2 to 0.54. The loss does not 
continuously decrease, but it decreases overall. Increasing the for loop iterations to 32 loops, 
decreases the accuracy from 60 to 59% and the loss decreases from 2.2 to 0.6 
Thus we can see that increasing the training time for a much longer time does not help in increasing 
the accuracy. What happens when we try to increase the training time is the model goes from 
underfitting to optimal to overfit after some time. As we can see above, the model goes from 54% to 
60% but stays constant at 60% when you from 12 to 22 loops. And we should train the model longer 
only when we have a huge amount of data. 

2 - Changing the learning rate and other hyperparameters
When we change the learning rate to 0.01 and keeping the momentum same, the accuracy 
decreases drastically to 10% It is the same when we make the learning rate 0.00001, the accuracy is 
11% Here the loss decreases and immediately keeps on increasing. When we decrease the learning 
rate the accuracy drops drastically. When we decrease the learning rate to 0.00001 and momentum 
to 0.2, the accuracy remains low at 10% With learning rate (lr) = 0.001 and momentum = 0.5 the 
accuracy jumps from 10 to 50%
Thus, keeping a small learning rate and large momentum, decreases the accuracy of the 
model(around 10%). In this case, the model becomes better at classifying only a single class label, in 
our case we can see that at first only cat has an accuracy of 95%, at second only dog has an accuracy 
of 94% A medium range momentum and a small learning rate gives somewhat better accuracy 
(around 50%). 

3 - Change the number or sizes of the fully connected layers
When we change the number of the fully connected layers to 4, the accuracy decreases from 54% to 
47% and when we change the number of fully connected layers to 5 the accuracy increases back to 
54% Changing the number of fully connected layers to 10, the accuracy increases a little, it becomes 
56% Thus increasing the number of the fully connected layers, increases the accuracy of the model 
because as we increase the layers, the feature extraction becomes more specific.

4 – Nonlinear function other than ReLU
Using ReLU we got the model accuracy as 54% and using tanh as the activation function, the model 
accuracy is 55% and with sigmoid it is 10% The model performs the worst with sigmoid activation as 
it just learns to classify a single class, giving us the accuracy of class bird as 100% Thus we can 
observe that ReLU is the better option to choose as an activation function.

5 – Adding a dropout layer
Adding a dropout layer, the accuracy of the model remains constant at 54% The accuracy of the 
model does not increase because our model is not overfitting. It has yet to reach the optimal stage. 
Thus adding a dropout layer does not help in increasing the accuracy of the model. 

Integrating everything together: 
In the final model, we increased the fully connected layers, added a dropout layer, changed the 
momentum to 0.5 and increased the training time to 12 epochs. We see that the accuracy of the 
model increases from 54% to 63% Adding the dropout layer does not help, because the size of our 
dataset is very small. But increasing the training period significantly (not too much) and adding the 
fully connected layers helps in increasing the accuracy of the model. The learning rate of 0.001 is 
appropriate, as we had seen that decreasing the learning rate decreased the accuracy. Comparing 
the accuracy of each class from before and after we observe that the accuracy is increased for all 
classes but dog, and we can conclude that the model is performing better than before.
