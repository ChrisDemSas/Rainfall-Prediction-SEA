# Rainfall-Prediction-SEA
A neural network prediction on the monthly rainfall of Singapore. 

This project was inspired by the paper "Rainfall Monthly Prediction Based on Artificial Neural Network – A Case Study Tenggarong Station, 
East Kalimantan – Indonesia" by Mislan et. al and contains an attempt to implement the neural network provided by them. There was a lot of problems with the Levenberg-Marquardt Descent in the project because of vanishing gradients. Since I currently lack the knowledge on how to resolve this issue, this implementation of the Levenberg-Marquardt descent represents my best effort 
until I learn more about numerical methods. Otherwise, the project was extremely fun to do and I enjoyed a few successes on learning how to read papers, designing my own optimisers and writing a custom neural network.

To use the code:

1) Download the dataset from: https://www.kaggle.com/datasets/kelvinchow1979/monthly-rainfall-in-singapore?resource=download
2) Download all the code in the repository.
3) Put the dataset inside the folder containing all the code in the repository.
4) Change the arguments at the top of the page and then run 'train.py'.

Note that the most optimal settings I used are:

1) Lambda = 100 (Original used 0.1)
2) Epochs = 200
3) Train-Test split: 80:20 ratio

Note that because of vanishing gradients, the lambda seems to vanish quicker at any other value. I attribute this to the normalizing transformations, whereby at higher epochs, the further backpropagation seem to lead to the weights vanishing and intereferes with the calculation of the inverse during Levenberg-Marquardt descent. The next step would be to learn more about numerical methods and how to prevent this from happening again.
