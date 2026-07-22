
def squared_error(
    prediction,
    target,
):
    """
    Возвращает квадрат ошибки для одного объекта.
    """
    return (prediction - target) ** 2


def squared_error_derivative(
    prediction,
    target,
):
    """
    Возвращает производную squared_error по prediction.
    """
    return 2 * (prediction - target)


def evaluate_neuron(
    features,
    weights,
    bias,
    target,
):
    """
    Возвращает prediction, loss и производную loss
    по prediction.
    """
    if len(features) != len(weights):
        raise ValueError("Arrays lenths are not equal")

    prediction = 0

    for i in range(0, len(features)):
        prediction += features[i] * weights[i]
    prediction += bias

    loss = squared_error(prediction, target)
    loss_derivative = squared_error_derivative(prediction, target)

    return (prediction, loss, loss_derivative)


    