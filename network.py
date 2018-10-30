from ops import DenseBlock, conv, Transition, global_avg_pooling, fully_connected, batchnorm, relu, preprocess


def DenseNet(inputs, nums_out, growth_rate, train_phase, depth):
    inputs = preprocess(inputs)
    n = (depth - 4) // 3
    inputs = conv("conv1", inputs, nums_out=16, k_size=3)
    inputs = DenseBlock("DenseBlock1", inputs, n, growth_rate, train_phase)
    inputs = Transition("Transition_Layer1", inputs, nums_out=growth_rate, train_phase=train_phase)
    inputs = DenseBlock("DenseBlock2", inputs, n,  growth_rate, train_phase)
    inputs = Transition("Transition_Layer2", inputs, nums_out=growth_rate, train_phase=train_phase)
    inputs = DenseBlock("DenseBlock3", inputs, n, growth_rate, train_phase)
    inputs = batchnorm(inputs, train_phase, "BN")
    inputs = relu(inputs)
    inputs = global_avg_pooling(inputs)
    inputs = fully_connected("FC", inputs, nums_out)
    return inputs


