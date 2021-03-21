from torch import nn


# The Siamese Network (Core component) implementation
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.size_inputs = [400, 200, 100, 2]
        self.fc_layers = self.create_fc_layers()

    def create_fc_layers(self):
        layers = []

        for i in range(len(self.size_inputs) - 1):
            if self.size_inputs[i + 1] is not None:
                n_out_features = self.size_inputs[i + 1]
                n_in_features = self.size_inputs[i]
                layers += [
                    nn.Linear(in_features=n_in_features, out_features=n_out_features)]
                # If not the last layer
                if i != len(self.size_inputs) - 2:
                    layers += [nn.ReLU()]

        return nn.Sequential(*layers)


    def forward(self, input_features):
        result = self.fc_layers(input_features)
        return result




if __name__ == "__main__":
    siamese_net = SiameseNet()
    print(siamese_net)
    print("test")