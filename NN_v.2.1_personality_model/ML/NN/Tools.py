import numpy as np

def orthogonal_initializer(inp):
    """
    initializer: orthogonal matrix
    """
    dim = len(inp.shape)
    if dim == 2:
        column_size = inp.shape[0]
        row_size = inp.shape[1]
        if row_size%column_size == 0:
            ortho_groups = int(row_size/column_size)
            end_in = column_size
        else:
            ortho_groups = 1+int(row_size/column_size)
            end_in = row_size%column_size
        output = np.zeros_like(inp)
        start_from = 0
        for _idx in range(ortho_groups):
            output[:, start_from:end_in], _, _ = np.linalg.svd(inp[:, start_from:end_in],
                                                               full_matrices=False)
            start_from = end_in
            end_in += column_size
    if dim == 3:
        kernel_size = inp.shape[0]
        column_size = inp.shape[1]
        row_size = inp.shape[2]
        if row_size%column_size == 0:
            ortho_groups = int(row_size/column_size)
            _end_in = column_size
        else:
            ortho_groups = 1+int(row_size/column_size)
            _end_in = row_size%column_size
        output = np.zeros_like(inp)
        start_from = 0
        for k_idx in range(kernel_size):
            start_from = 0
            end_in = _end_in
            for _idx in range(ortho_groups):
                output[k_idx, :, start_from:end_in], _, _ = np.linalg.svd(inp[k_idx, :, start_from:end_in],
                                                                          full_matrices=False)
                start_from = end_in
                end_in += column_size

    return output



def softmax(x):
    after_softmax = []
    for row in range(x.shape[0]):
        this_row = np.exp(x[row])/np.sum(np.exp(x[row]))
        after_softmax.append(this_row)
    return np.array(after_softmax)

def accuracy_test(pred, target):
    accuracy = 0
    for element in range(len(pred)):
        if pred[element] == target[element]:
            accuracy += 1./len(pred)
    return accuracy

