# VIME
This is a PyTorch implementation of the paper VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain on MNIST DATA

I use some codes from the original repo that uses tensorflow https://github.com/jsyoon0823/VIME ; I also used some functions from the repo https://github.com/aladdinpersson/Machine-Learning-Collection <br>

The notebook that compares supervised learning, semi-supervised learning and self-supervised learning is VIME_training.ipynb <br>

The semi-supervised learning is not working correctly and I'm still working on debugging It.
So far Here are the accuracies that I get on the test set : <br>
	| Method | Accuracy |
    |--- | --- |
    |Supervised trained on only 1K examples| 76.76% |
    |Self supervised only + Supervised on 1K examples | 83.36% |
    |Semi-supervised | ? |