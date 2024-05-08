import numpy as np

a = np.array([
    [
        [0.1, 0.2, 0.3, 0.4],
        [0.11, 0.22, 0.33, 0]
    ],
])

np.save('mmi_imbalances.npy', a)

