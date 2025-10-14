import torch
import torch.nn as nn
import math

class CustomLinear(nn.Module):
    def __init__(self, input_shape:tuple, output_shape:tuple):  # (input: channels, length)  ; output: (i_channels, length)
        super().__init__()
        # print(output_shape, input_shape)
        self.i_shape = input_shape
        self.o_shape = output_shape

        if (output_shape[1] == input_shape[1]):
            self.weights1 = nn.Parameter(torch.Tensor(output_shape[0], input_shape[0]))
            self.bias1 = nn.Parameter(torch.Tensor(output_shape[0], output_shape[1]))
            # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
            nn.init.normal_(self.weights1, mean=0.0, std=0.001)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias1, -bound, bound)  # bias init

        if (output_shape[1] != input_shape[1]):
            self.weights1 = nn.Parameter(torch.Tensor(output_shape[0], input_shape[0]))
            self.bias1 = nn.Parameter(torch.Tensor(output_shape[0], input_shape[1]))
            # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
            nn.init.normal_(self.weights1, mean=0.0, std=0.001)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias1, -bound, bound)  # bias init

            self.weights2 = nn.Parameter(torch.Tensor(input_shape[1], output_shape[1]))
            self.bias2 = nn.Parameter(torch.Tensor(output_shape[0], output_shape[1]))

            # nn.init.kaiming_uniform_(self.weights,  a=math.sqrt(5)) # weight init
            nn.init.normal_(self.weights2, mean=0.0, std=0.001)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias2, -bound, bound)  # bias init



    def forward(self, x):
        if (self.o_shape[1] != self.i_shape[1]):
            x = torch.add(torch.matmul(self.weights1, x ), self.bias1)
            x = torch.add(torch.matmul(x, self.weights2 ), self.bias2)
            return x

        x = torch.add(torch.matmul(self.weights1, x ), self.bias1)
        return x
