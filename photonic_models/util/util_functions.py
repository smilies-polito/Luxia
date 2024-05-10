import torch


def diag_matrix_alternating_tensors(tensor0: torch.tensor, tensor1: torch.tensor):
    """
    This function creates a diagonal matrix with the input tensors in
    alternated fashion
    Args:
        tensor (torch.tensor): Input tensor

    Returns:
        torch.tensor: Diagonal matrix with the input tensors in alternated
        fashion
        """
    # Stack the tensors along a new dimension
    stack_tensor = torch.stack((tensor0, tensor1), dim=1)
    # Reshape the stacked tensor to flatten it
    vec_tensor = stack_tensor.view(-1)
    # Create a diagonal matrix with the phases
    diag_matrix = torch.diag(vec_tensor)
    return diag_matrix


def diag_matrix_alternating_tensor_with_ones(tensor: torch.tensor):
    """
    This function creates a diagonal matrix with the input tensor and
    alternates the diagonal with ones.
    Args:
        tensor (torch.tensor): Input tensor

    Returns:
        torch.tensor: Diagonal matrix with the input tensor and alternated
        with ones
    """
    # Create a vector of ones with the same size as the phases
    vec_ones = torch.ones_like(tensor)
    # Stack the tensors along a new dimension
    stack_tensor = torch.stack((tensor, vec_ones), dim=1)
    # Reshape the stacked tensor to flatten it
    vec_tensor = stack_tensor.view(-1)
    # Create a diagonal matrix with the phases
    diag_matrix = torch.diag(vec_tensor)
    return diag_matrix


def check_and_resize_input(input: torch.tensor, capacity: int):
    """
    This function checks if the input tensor exceeds the given capacity and
    resizes it to the given capacity if it is smaller than the capacity.

    Args:
        input: Input tensor
        capacity: Capacity number for the comparison

    Raises:
        ValueError: Error Message when the number of features exceeds the given
        capacity

    Returns: reasized input tensor filled with zeros or the input tensor when
    the resize is not need it
    """
    input_size = input.shape[-1]
    if input_size > capacity:
        raise ValueError(
            f'The number of features exceeds the given capacity the maximum'
            f'allowed number is: {capacity} the number of features is: '
            f'{input_size}'
        )

    if input_size < capacity:
        # Create a tensor filled with zeros with the given capacity
        input_resized = torch.zeros(
            (input.shape[0], capacity), dtype=input.dtype, device=input.device)

        # Calculate the start and end limits for the input tensor
        start_limit = (capacity - input_size) // 2
        end_limit = start_limit + input_size
        # Fill the resized tensor with the input tensor
        input_resized[..., start_limit:end_limit] = input
        return input_resized
    return input
