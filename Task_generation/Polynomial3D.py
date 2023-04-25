import numpy as np

from torchmeta.utils.data import Task, MetaDataset
import torch
from numpy.polynomial.polynomial import polyval3d
from numpy.polynomial.chebyshev import chebval3d



class Polynomial3D(MetaDataset):

    def __init__(self, num_samples_per_task, degree=5, cube_range=[[0,1],[0,1],[0,1]], num_tasks=1000,chebyshev=True,
                 noise_std=None, transform=None, target_transform=None,
                 dataset_transform=None):
        super(Polynomial3D, self).__init__(meta_split='train',
            target_transform=target_transform, dataset_transform=dataset_transform)
        self.num_samples_per_task = num_samples_per_task
        self.num_tasks = num_tasks
        self.noise_std = noise_std
        self.transform = transform
        self.degree = degree
        self.chebyshev=chebyshev

        self.x_range = np.array(cube_range[0])
        self.y_range = np.array(cube_range[1])
        self.z_range = np.array(cube_range[2])
        self._coeff_range = np.array([-1, 1])

        self._coefficients = None

    @property
    def coefficients(self):
        if self._coefficients is None:
            self._coefficients = self.np_random.uniform(self._coeff_range[0],
                self._coeff_range[1], size=(self.num_tasks,self.degree, self.degree, self.degree))
        return self._coefficients


    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        coefficients = self.coefficients[index]
        task = Polynomial3DTask(index, coefficients, self.x_range, self.y_range, self.z_range, self.chebyshev,
            self.noise_std, self.num_samples_per_task, self.transform,
            self.target_transform, np_random=self.np_random)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task


class Polynomial3DTask(Task):
    def __init__(self, index, coefficients, x_range, y_range, z_range, chebyshev, noise_std,
                 num_samples, transform=None, target_transform=None,
                 np_random=None):
        super(Polynomial3DTask, self).__init__(index, None) # Regression task
        self.coefficients = coefficients
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.num_samples = num_samples
        self.noise_std = noise_std

        self.transform = transform
        self.target_transform = target_transform

        if np_random is None:
            np_random = np.random.RandomState(None)

        self.x = np_random.uniform(self.x_range[0], self.x_range[1],
            size=(num_samples if self.x_range[0] != self.x_range[1] else 1 ))
        self.y = np_random.uniform(self.y_range[0], self.y_range[1],
            size=(num_samples if self.y_range[0] != self.y_range[1] else 1 ))
        self.z = np_random.uniform(self.z_range[0], self.z_range[1],
            size=(num_samples if self.z_range[0] != self.z_range[1] else 1 ))
            
        self._inputs = torch.cartesian_prod(torch.from_numpy(self.x),torch.from_numpy(self.y),torch.from_numpy(self.z)).numpy()
        if not chebyshev:
            self._targets = polyval3d(self._inputs[:,0],self._inputs[:,1],self._inputs[:,2],self.coefficients)
        else:
            self._targets = chebval3d(self._inputs[:,0],self._inputs[:,1],self._inputs[:,2],self.coefficients)

        if (noise_std is not None) and (noise_std > 0.):
            self._targets += noise_std * np_random.randn(num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        input, target = self._inputs[index], self._targets[index]

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (input, target)