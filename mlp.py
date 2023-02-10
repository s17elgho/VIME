import torch
import torch.nn.functional as F 

class MLP(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        """
        Here we define the layers of the network. We create two fully connected layers
        Parameters:
            input_size: the size of the input, in this case 784 (28x28)
            num_classes: the number of classes we want to predict, in this case 10 (0-9)
        """
        super(MLP, self).__init__()
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.fc1 = torch.nn.Linear(input_size, 50)
        self.fc2 = torch.nn.Linear(50, num_classes)

    def forward(self, x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        Parameters:
            x: mnist images
        Returns:
            out: the output of the network
        """

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x