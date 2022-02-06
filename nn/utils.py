from nn.activation_functions import Relu, Sigmoid, Tanh, Linear
from nn.core import SequentialNetwork
from nn.layers import Flatten, Softmax, Dense
from nn.loss_functions import CrossEntropy, MSE
from nn.regularizers import L2, L1
from utils.value_checking import check_boundary_value, check_option_value


def network_from_config(config: dict) -> SequentialNetwork:
    global_params: dict = config.get("globals", {})
    layer_configs: list = config.get("layers", [])

    # parsing global parameters
    lr = global_params.get("lr", 0.001)
    epochs = check_boundary_value("epochs", global_params.get("epochs", 100), min_val=0)
    loss = check_option_value("loss", global_params.get("loss", "mse"), options=["mse", "cross_entropy"])
    reg_type = check_option_value("reg_type", global_params.get("reg_type", "none"), options=["none", "l1", "l2"])
    reg_rate = global_params.get("reg_rate", 0.0001)
    visualize = global_params.get("visualize", True)

    if loss == "cross_entropy":
        lf = CrossEntropy()
    else:
        lf = MSE()

    # init network
    model = SequentialNetwork(
        learning_rate=lr,
        loss_function=lf,
        epochs=epochs,
        visualize=visualize
    )

    # init regularizer
    if reg_type == "l1":
        regularizer = L1(reg_rate)
    elif reg_type == "l2":
        regularizer = L2(reg_rate)
    else:
        regularizer = None

    # add layers
    for layer_config in layer_configs:
        layer_type = check_option_value("layer", layer_config.get("type", "dense"), options=["dense", "flatten", "softmax"])
        act = check_option_value("act", layer_config.get("act", "relu"), options=["relu", "sigmoid", "tanh", "linear"])
        units = layer_config.get("units", None)
        wr = layer_config.get("wr", [-0.1, 0.1])

        if act == "relu":
            af = Relu()
        elif act == "sigmoid":
            af = Sigmoid()
        elif act == "tanh":
            af = Tanh()
        else:
            af = Linear()

        if layer_type == "flatten":
            layer = Flatten()
        elif layer_type == "softmax":
            layer = Softmax()
        else:
            layer = Dense(
                units=units,
                activation=af,
                wr=wr,
                regularizer=regularizer
            )
        model.add(layer)

    return model
