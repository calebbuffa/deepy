def dice_coeff(
    y_hat: Tensor, y: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert y_hat.size() == y.size(), print(y_hat.size(), y.size())
    if y_hat.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f"Dice: asked to reduce batch but got tensor without batch dimension (shape {y_hat.shape})"
        )

    if y_hat.dim() == 2 or reduce_batch_first:
        inter = torch.dot(y_hat.reshape(-1), y.reshape(-1))
        sets_sum = torch.sum(y_hat) + torch.sum(y)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = sum(dice_coeff(y_hat[i, ...], y[i, ...]) for i in range(y_hat.shape[0]))

        return dice / y_hat.shape[0]


def multiclass_dice_coeff(
    y_hat: Tensor, y: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    # Average of Dice coefficient for all classes
    assert y_hat.size() == y.size()
    dice = sum(
        dice_coeff(
            y_hat[:, channel, ...], y[:, channel, ...], reduce_batch_first, epsilon,
        )
        for channel in range(y_hat.shape[1])
    )

    return dice / y_hat.shape[1]
