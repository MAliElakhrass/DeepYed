from torch import nn

INPUT_FEATURES = 773


# The DBN(Pos2Vec) implementation
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.input_features = INPUT_FEATURES
        self.size_inputs = [773, 600, 400, 200, 100]
        self.encoder_layers = self.create_encoder_layers()
        self.decoder_layers = self.create_decoder_layers()

    def create_encoder_layers(self):
        layers = []

        for i in range(len(self.size_inputs) - 1):
            if self.size_inputs[i + 1] is not None:
                n_out_features = self.size_inputs[i + 1]
                n_in_features = self.size_inputs[i]
                layers += [
                    nn.Linear(in_features=n_in_features, out_features=n_out_features),
                    nn.ReLU()]

        return nn.Sequential(*layers)

    def create_decoder_layers(self):
        layers = []
        for i, _ in reversed(list(enumerate(self.size_inputs))):
            if self.size_inputs[i - 1] is not None:
                n_out_features = self.size_inputs[i - 1]
                n_in_features = self.size_inputs[i]
                if n_in_features > n_out_features:
                    break
                layers += [
                    nn.Linear(in_features=n_in_features, out_features=n_out_features)]
                if i != 1:
                    layers += [nn.ReLU()]
                else:
                    layers += [nn.Sigmoid()]

        return nn.Sequential(*layers)

    def forward(self, input_features):
        encoder_result = self.encoder_layers(input_features.view(-1, 773))
        decoder_result = self.decoder_layers(encoder_result)
        return decoder_result, encoder_result


if __name__ == "__main__":
    autoencoder = AutoEncoder()
    print(autoencoder)

