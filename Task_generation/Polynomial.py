import numpy as np
from torchmeta.utils.data import Task, MetaDataset



class Polynomial(MetaDataset):

    def __init__(self, num_samples_per_task, degree=10, num_tasks=1000000,
                 chebyshev=True, noise_std=None, transform=None, target_transform=None,
                 dataset_transform=None):
        super(Polynomial, self).__init__(meta_split='train',
            target_transform=target_transform, dataset_transform=dataset_transform)
        self.num_samples_per_task = num_samples_per_task
        self.num_tasks = num_tasks
        self.noise_std = noise_std
        self.transform = transform
        self.degree = degree
        self.chebyshev = chebyshev

        self._input_range = np.array([-1.0, 1.0])
        self._coeff_range = np.array([-1.0, 1.0])

        self._coefficients = None

    @property
    def coefficients(self):
        if self._coefficients is None:
            self._coefficients = self.np_random.uniform(self._coeff_range[0],
                self._coeff_range[1], size=(self.num_tasks,self.degree))
        return self._coefficients


    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        coefficients = self.coefficients[index]
        task = PolynomialTask(index, coefficients, self._input_range, self.chebyshev,
            self.noise_std, self.num_samples_per_task, self.transform,
            self.target_transform, np_random=self.np_random)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task


class PolynomialTask(Task):
    def __init__(self, index, coefficients, input_range, chebyshev, noise_std,
                 num_samples, transform=None, target_transform=None,
                 np_random=None):
        super(PolynomialTask, self).__init__(index, None) # Regression task
        self.coefficients = coefficients
        self.input_range = input_range
        self.num_samples = num_samples
        self.noise_std = noise_std

        self.transform = transform
        self.target_transform = target_transform

        if np_random is None:
            np_random = np.random.RandomState(None)

        self._inputs = np_random.uniform(input_range[0], input_range[1],
            size=(num_samples, 1))
        if chebyshev:
            self._targets = np.polynomial.chebyshev.Chebyshev(coefficients)(self._inputs)
        else:
            self._targets = np.polynomial.Polynomial(coefficients)(self._inputs)
            
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





