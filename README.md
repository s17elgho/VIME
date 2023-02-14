# VIME
This is a PyTorch implementation of the paper VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain on MNIST DATA

I use some codes from the original repo that uses tensorflow https://github.com/jsyoon0823/VIME ; I also used some functions from the repo https://github.com/aladdinpersson/Machine-Learning-Collection <br>

The notebook that compares supervised learning, semi-supervised learning and self-supervised learning is VIME_training.ipynb <br>

The semi-supervised learning is not working correctly and I'm still working on debugging It.
So far Here are the accuracies that I get on the test set :
| Method                             | Accuray |
|------------------------------------|---------|
| Supervised on 1K examples          | ~88.16%  |
| Self sup.only + Sup.on 1K examples | ~90.14%  |
| Semi-supervised                    | ~88.89%   | 

In the notebook VIME_income_data.ipynb I tried self supervised learning . However, I still get a better accuracy for the supervised xgboost model. This means that either I need to work on optimizing the hyperparameters or I need to change the loss definition on categorical variables.


| Method                             | Accuray |
|------------------------------------|---------|
| Supervised on 10% of the data      | ~86.09%  |
| Self sup.only + Sup.on 10% | ~83.65%  |