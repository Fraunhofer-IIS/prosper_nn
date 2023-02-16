import numpy as np
import random
import torch
import sys, os

sys.path.append(os.path.abspath("."))


def createModelData_T(rules, num_examples, num_inputs, num_outputs):
    inputs = np.random.uniform(
        size=(num_examples, num_inputs), low=-4, high=4
    )  # random inputs
    outputs = np.zeros(shape=(num_examples))  # empty labels -> tbd

    lower_border = np.random.uniform(
        -3, -0.5, num_inputs
    )  # borders for each input where the input should count as shifted negative
    upper_border = np.random.uniform(
        0.5, 3, num_inputs
    )  # borders for each input where the input should count as shifted positive
    print("lower borders: ", lower_border)
    print("upper borders: ", upper_border)

    for index, inp in enumerate(inputs):  # iterate Examples
        highest_output = np.zeros(
            num_outputs
        )  # shape of target vector -> tbd (1, 3, 2, 1, 2) -> (0, 1, 0, 0, 0)
        for i, output in enumerate(rules):
            # rules(5, 4, 3), output(4, 3)
            # rules (output 1
            # input 1
            # neg, const, pos
            # 0, 1, 0
            # input 2
            # neg, const, pos
            for ins, rule, lower, upper in zip(
                inp, output, lower_border, upper_border
            ):  # iterate input values
                # rule (3)
                if ins <= lower:  # input is low
                    highest_output[i] += rule[0]  # add rule for low input
                elif ins >= upper:  # input is high
                    highest_output[i] += rule[2]  # add rule for high
                else:  # input is constant
                    highest_output[i] += rule[1]  # add rule for constant

        outputs[index] = int(
            highest_output.argmax()
        )  # set label according to highest output

    return (
        torch.from_numpy(inputs).type(torch.DoubleTensor),
        torch.from_numpy(outputs).type(torch.LongTensor),
        lower_border,
        upper_border,
    )


def create_gaussian_data(num, sigma):
    inputs = torch.rand(size=(num, 1)) * 4 - 2
    noise = torch.rand(size=(num, 1)) * 2 - 1
    outputs_const = np.exp(-1 / 2 * (inputs / sigma) ** 2) + noise * 0.2
    return inputs, outputs_const


def create_normlog_data(num, slope):
    inputs = torch.rand(size=(num, 1)) * 4 - 2
    noise = torch.rand(size=(num, 1)) * 2 - 1
    outputs_const = 1 / (1 + torch.exp(-4 * (slope * inputs - 0.5))) + noise * 0.2
    return inputs, outputs_const


def linear_function(
    length,
    slope_factor=np.random.normal(0.2, 0.05, 1),
    noise_amplitude=0.05,
    sin=False,
    negative=False,
):
    x = np.arange(length)
    slope = np.random.random_sample() * slope_factor
    if negative:
        slope = -slope
    a = np.random.uniform(-2, 2, 1)
    noise = np.random.uniform(-noise_amplitude, noise_amplitude, length)
    y = slope * x + a + noise
    if sin and np.random.random_sample() < 0.3:
        y = y + 0.5 * np.random.random_sample() * np.sin(
            2 * np.random.random_sample() * x + np.random.random_sample()
        )
    return y


def createTimeSeriesData(batchsize, sequence_length, num_input, num_output, rules):
    inputs = np.zeros((batchsize, num_input, sequence_length))
    # outputs = np.zeros((batchsize, num_output)) uncomment for MSE Loss
    outputs = np.zeros(batchsize)
    for n in range(batchsize):
        err = random.randint(0, num_output - 1)
        # outputs[n, err] = 1 uncomment for MSE Loss
        outputs[n] = err
        if n == 0:
            print(rules[err])
        for i, rule in enumerate(rules[err]):
            if rule[0] == 1.0:
                inputs[n][i] = linear_function(sequence_length, negative=True, sin=True)
            if rule[1] == 1.0:
                inputs[n][i] = linear_function(
                    sequence_length, slope_factor=0, sin=True
                )
            if rule[2] == 1.0:
                inputs[n][i] = linear_function(sequence_length, sin=True)
    inputs = inputs.swapaxes(1, 2)
    return torch.from_numpy(inputs), torch.from_numpy(outputs)
