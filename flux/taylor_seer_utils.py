import math


def approximate_derivative(Y, dY_prev, current_step, last_non_approximated_step):
    """
    Approximate the derivative of Y using the previous derivatives

    Args:
        Y: current value of the feature, i.e. Y=f(X) where f could be a transformer or linear layer
        dY_prev: the value of the derivative of Y t steps ago
        elapsed_steps: number of steps between Y and dY_prev
    """
    order = len(dY_prev)
    n_derivatives = order - 1

    dY_current = [None] * order
    dY_current[0] = Y

    finite_difference_window = current_step - last_non_approximated_step

    for i in range(n_derivatives):
        if dY_prev[i] is not None and current_step > 1:
            # equation (7) from the paper
            dY_current[i + 1] = (dY_current[i] - dY_prev[i]) / finite_difference_window
        else:
            break
    return dY_current


def approximate_value(dY_current, elapsed_steps):
    """
    Approximate the current value of Y using our current estimate of the derivative
    and the # of timesteps that have passed since the derivative was computed

    Args:
        dY_current: the value of the derivatives of Y
        elapsed_steps: number of steps between Y and dY_prev
    """
    # taylor series formula
    output = 0
    for i in range(len(dY_current)):
        if dY_current[i] is not None:
            output += (1 / math.factorial(i)) * dY_current[i] * (elapsed_steps**i)
        else:
            break
    return output
