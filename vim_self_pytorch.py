import torch

class VimPretext(torch.nn.Module):
    def __init__(self,dim = 28*28):
        super().__init__()
        self.dim = dim
		# Building an linear encoder to predict feature and mask
        self.fc0 = torch.nn.Sequential(
			torch.nn.Linear(self.dim,self.dim),
			torch.nn.ReLU()
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(self.dim,self.dim),
            torch.nn.Sigmoid()
        )
        self.fc2 = torch.nn.Sequential(
			torch.nn.Linear(self.dim,self.dim),
			torch.nn.Sigmoid()
		)

    def forward(self, x):
        encoder = self.fc0(x) #z
        mask = self.fc1(encoder)
        feature = self.fc2(encoder)
        return mask, feature, encoder

if __name__ == "__main__":
    x = torch.rand(3, 28*28)
    model = VimPretext()
    mask, feature,_ = model(x)
    print(mask.shape, feature.shape)
