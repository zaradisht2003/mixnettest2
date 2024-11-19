import json
import torch
import torch.nn as nn
from collections import OrderedDict


class newMixNet(nn.Module):
    """Neural Network to predict the future trajectory of a vehicle based on its history.
    It predicts mixing weights for mixing the boundaries, the centerline and the raceline.
    Also, it predicts section wise constant accelerations and an initial velocity to
    compute the velocity profile from.
    """

    def __init__(self, params):
        """Initializes a MixNet object."""
        super(newMixNet, self).__init__()

        self._params = params

        # Input embedding layer:
        self._ip_emb = torch.nn.Linear(2, params["encoder"]["in_size"])
    # the input embedding the first layer transforming from 2d to more complex 
    # higher deminsional to capture more complex relationship within the data
        
        # History encoder LSTM:
        self._enc_hist = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        # Boundary encoders:
        self._enc_left_bound = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        self._enc_right_bound = torch.nn.LSTM(
            params["encoder"]["in_size"],
            params["encoder"]["hidden_size"],
            1,
            batch_first=True,
        )
        # the first parameter is the input of the LSTM which is of size "in_size" as it will take
        # the input from the embedding layer, the second parameter is of size "hidden size"
        #  and enabling batching
        # makes them predict more than one trajectory to more than one car
        
        
        # Linear stack that outputs the path mixture ratios:
        self._mix_out_layers = self._get_linear_stack(
            in_size=params["encoder"]["hidden_size"] * 3,
            hidden_sizes=params["mixer_linear_stack"]["hidden_sizes"],
            out_size=params["mixer_linear_stack"]["out_size"],
            name="mix",
        )
        # the layer before the linear softmax that predicts the weights for the predefined paths
        # takes as an input the concatination of the three encoder "history" "left boundry"
        # "right boundry" and outputs number of weights equals to "out_size" which is 4,
        # the number of predefined paths that we want to give them weights (the right and
        # left boundries, the centerline, the raceline(the most optimal))

        # Linear stack for outputting the initial velocity:
        self._vel_out_layers = self._get_linear_stack(
            in_size=params["encoder"]["hidden_size"]*3,# new: make it by 3 to add the left and right boundries Q- why we predict the initial velocity, why not calculate it with our sensirs?
            hidden_sizes=params["init_vel_linear_stack"]["hidden_sizes"],
            out_size=params["init_vel_linear_stack"]["out_size"],
            name="vel",
        )
        # the layer that predicts the initial velocity which will be the part of 
        # velocity profile, it takes as input the history, left & right boundries encoder 
        # gives an output which is the initial veloctiy
        

        # dynamic embedder between the encoder and the decoder:
        self._dyn_embedder = nn.Linear(
            params["encoder"]["hidden_size"] * 3, params["acc_decoder"]["in_size"]
        )
        # the layer takes the concatinating output encoders and mapping them to less deminsion
        # to use it in the accelration prediction

        # acceleration decoder:
        self._acc_decoder = nn.LSTM(
            params["acc_decoder"]["in_size"],
            params["acc_decoder"]["hidden_size"],
            1,
            batch_first=True,
        )
        # the lstm used to predict the acceleration

        # output linear layer of the acceleration decoder:
        self._acc_out_layer = nn.Linear(params["acc_decoder"]["hidden_size"], 1)
        #make the output to one acceleration used in the velocity accelaration profile

        # migrating the model parameters to the chosen device:
        if params["use_cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using CUDA as device for MixNet")
        else:
            self.device = torch.device("cpu")
            print("Using CPU as device for MixNet")

        self.to(self.device)

    def forward(self, hist, left_bound, right_bound):
        """Implements the forward pass of the model.

        args:
            hist: [tensor with shape=(batch_size, hist_len, 2)]
            left_bound: [tensor with shape=(batch_size, boundary_len, 2)]
            right_bound: [tensor with shape=(batch_size, boundary_len, 2)]

        returns:
            mix_out: [tensor with shape=(batch_size, out_size)]: The path mixing ratios in the order:
                left_ratio, right_ratio, center_ratio, race_ratio
            vel_out: [tensor with shape=(batch_size, 1)]: The initial velocity of the velocity profile
            acc_out: [tensor with shape=(batch_size, num_acc_sections)]: The accelerations in the sections
        """

        # encoders:
        _, (hist_h, _) = self._enc_hist(self._ip_emb(hist.to(self.device)))
        _, (left_h, _) = self._enc_left_bound(self._ip_emb(left_bound.to(self.device)))
        _, (right_h, _) = self._enc_right_bound(
            self._ip_emb(right_bound.to(self.device))
        )
        #saving the outputs of the lstms of history and right and left boundries by first 
        # applying the input embeddence to map from 2 deminsions to more complex higher deminsion
        #then go to the lstm to generate hidden and cell state, the cell state is for the long memory
        # and the hidden state is the short memory and also the output
        """new adding the leaky relu activation function on the outputs of the lstms"""
        new_hist_h=torch.nn.functional.leaky_relu(hist_h)
        new_left_h=torch.nn.functional.leaky_relu(left_h)
        new_right_h=torch.nn.functional.leaky_relu(right_h)
        
        # concatenate and squeeze encodings:
        enc = torch.relu(torch.squeeze(torch.cat((new_hist_h, new_left_h, new_right_h), 2), dim=0))
        # concatinating the outputs of the lstms then they will be used in both the weights 
        # for the predefined curves and also the acceleration new for the velocity 
        """new: adding relu activation function after concatination of the lstms """
        """encoder is done"""
        
        
        # path mixture through softmax:
        mix_out = torch.softmax(self._mix_out_layers(enc), dim=1)
        # _mix_out_layers take the output of the encoder and make it to size of 4 equals to
        # the number of weights needed to the predefined lines then take that and give it to
        # softmax activation function to make the equation convex equation (the summation of
        # the weights will be 1)  

        # initial velocity:
        vel_out = self._vel_out_layers(enc)
        vel_out = torch.sigmoid(vel_out)
        vel_out = vel_out * self._params["init_vel_linear_stack"]["max_vel"]
        # it takes the output of the history encoder only and put it in _vel_out_layers that 
        # predicts the initial velocity and then give it to the segmoid to range it from 0 to 1
        # and then multiply the result by the max velocity, now we are sure there is no 
        # unexpected initial velocity will be predicted  

        # acceleration decoding:
        dec_input = torch.relu(self._dyn_embedder(enc)).unsqueeze(dim=1)
        dec_input = dec_input.repeat(
            1, self._params["acc_decoder"]["num_acc_sections"], 1
        )
        acc_out, _ = self._acc_decoder(dec_input)
        acc_out = torch.squeeze(self._acc_out_layer(torch.relu(acc_out)), dim=2)
        acc_out = torch.tanh(acc_out) * self._params["acc_decoder"]["max_acc"]
        # after combining the encoders outputs then it will be mapped to lower deminsional
        # then goes inside relu then repeat that by num_acc_sections to predict those number
        # of accelerations, _acc_out_layer maps that into one output which is the acceleration
        # then putting that in tanh to bound it from -1 to 1 and multiply it by the max 
        # acceleration  

        return mix_out, vel_out, acc_out

    def load_model_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path, map_location=self.device))
        print("Successfully loaded model weights from {}".format(weights_path))

    def _get_linear_stack(
        self, in_size: int, hidden_sizes: list, out_size: int, name: str
    ):
        """Creates a stack of linear layers with the given sizes and with relu activation."""

        layer_sizes = []
        layer_sizes.append(in_size)  # The input size of the linear stack
        layer_sizes += hidden_sizes  # The hidden layer sizes specified in params
        layer_sizes.append(out_size)  # The output size specified in the params

        layer_list = []
        for i in range(len(layer_sizes) - 1):
            layer_name = name + "linear" + str(i + 1)
            act_name = name + "relu" + str(i + 1)
            layer_list.append(
                (layer_name, nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            )
            layer_list.append((act_name, nn.ReLU()))

        # removing the last ReLU layer:
        layer_list = layer_list[:-1]

        return nn.Sequential(OrderedDict(layer_list))

    def get_params(self):
        """Accessor for the params of the network."""
        return self._params


if __name__ == "__main__":
    param_file = "mod_prediction/utils/mix_net/params/net_params.json"
    with open(param_file, "r") as fp:
        params = json.load(fp)

    batch_size = 32
    hist_len = 30
    bound_len = 50

    # random inputs:
    hist = torch.rand((batch_size, hist_len, 2))
    left_bound = torch.rand((batch_size, bound_len, 2))
    right_bound = torch.rand((batch_size, bound_len, 2))

    net = newMixNet(params)

    mix_out, vel_out, acc_out = net(hist, left_bound, right_bound)

    # printing the output shapes as a sanity check:
    print(
        "output shape: {}, should be: {}".format(
            mix_out.shape, (batch_size, params["mixer_linear_stack"]["out_size"])
        )
    )

    print("output shape: {}, should be: {}".format(vel_out.shape, (batch_size, 1)))

    print(
        "output shape: {}, should be: {}".format(
            acc_out.shape, (batch_size, params["acc_decoder"]["num_acc_sections"])
        )
    )

    print(vel_out)
