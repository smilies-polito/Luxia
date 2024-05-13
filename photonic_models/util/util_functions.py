import torch


def diag_matrix_alternating_tensors(tensor0: torch.tensor, tensor1: torch.tensor):
    """
    This function creates a diagonal matrix with the input tensors in
    alternated fashion
    Args:
        tensor (torch.tensor): Input tensor (n,)

    Returns:
        torch.tensor: Diagonal matrix with the input tensors in alternated
        fashion
        """
    # Stack the tensors along a new dimension
    stack_tensor = torch.stack((tensor0, tensor1), axis=1) # (n, 2)
    # Reshape the stacked tensor to flatten it
    vec_tensor = stack_tensor.view(-1) # (2*n,)
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

