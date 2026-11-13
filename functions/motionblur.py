"""From https://github.com/LeviBorodenko/motionblur."""

from math import pi

import numpy as np
from numpy.random import beta, triangular, uniform
from PIL import Image, ImageDraw, ImageFilter

eps = 0.1


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def norm(lst: list) -> float:
    if not isinstance(lst, list):
        raise ValueError("Norm takes a list as its argument")
    if lst == []:
        return 0
    return sum((i**2 for i in lst)) ** 0.5


def polar2z(r: np.ndarray, 罐: np.ndarray) -> np.ndarray:
    return r * np.exp(1j * 罐)


class Kernel(object):
    """Random motion-kernel generation retained from LeviBorodenko/motionblur."""

    def __init__(self, size: tuple = (100, 100), intensity: float = 0):
        if not isinstance(size, tuple):
            raise ValueError("Size must be TUPLE of 2 positive integers")
        elif len(size) != 2 or type(size[0]) != type(size[1]) != int:
            raise ValueError("Size must be tuple of 2 positive INTEGERS")
        elif size[0] < 0 or size[1] < 0:
            raise ValueError("Size must be tuple of 2 POSITIVE integers")
        if type(intensity) not in [int, float, np.float32, np.float64]:
            raise ValueError("Intensity must be a number between 0 and 1")
        elif intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be a number between 0 and 1")
        self.SIZE = size
        self.INTENSITY = intensity
        self.SIZEx2 = tuple([2 * i for i in size])
        self.x, self.y = self.SIZEx2
        self.DIAGONAL = (self.x**2 + self.y**2) ** 0.5
        self.kernel_is_generated = False

    def _createPath(self):

        def getSteps():
            self.MAX_PATH_LEN = 0.75 * self.DIAGONAL * (uniform() + uniform(0, self.INTENSITY**2))
            steps = []
            while sum(steps) < self.MAX_PATH_LEN:
                step = beta(1, 30) * (1 - self.INTENSITY + eps) * self.DIAGONAL
                if step < self.MAX_PATH_LEN:
                    steps.append(step)
            self.NUM_STEPS = len(steps)
            self.STEPS = np.asarray(steps)

        def getAngles():
            self.MAX_ANGLE = uniform(0, self.INTENSITY * pi)
            self.JITTER = beta(2, 20)
            angles = [uniform(low=-self.MAX_ANGLE, high=self.MAX_ANGLE)]
            while len(angles) < self.NUM_STEPS:
                angle = triangular(0, self.INTENSITY * self.MAX_ANGLE, self.MAX_ANGLE + eps)
                if uniform() < self.JITTER:
                    angle *= -np.sign(angles[-1])
                else:
                    angle *= np.sign(angles[-1])
                angles.append(angle)
            self.ANGLES = np.asarray(angles)

        getSteps()
        getAngles()
        complex_increments = polar2z(self.STEPS, self.ANGLES)
        self.path_complex = np.cumsum(complex_increments)
        self.com_complex = sum(self.path_complex) / self.NUM_STEPS
        center_of_kernel = (self.x + 1j * self.y) / 2
        self.path_complex -= self.com_complex
        self.path_complex *= np.exp(1j * uniform(0, pi))
        self.path_complex += center_of_kernel
        self.path = [(i.real, i.imag) for i in self.path_complex]

    def _createKernel(self):
        if self.kernel_is_generated:
            return None
        self._createPath()
        self.kernel_image = Image.new("RGB", self.SIZEx2)
        self.painter = ImageDraw.Draw(self.kernel_image)
        self.painter.line(xy=self.path, width=int(self.DIAGONAL / 150))
        self.kernel_image = self.kernel_image.filter(
            ImageFilter.GaussianBlur(radius=int(self.DIAGONAL * 0.01))
        )
        self.kernel_image = self.kernel_image.resize(self.SIZE, resample=Image.LANCZOS)
        self.kernel_image = self.kernel_image.convert("L")
        self.kernel_is_generated = True

    @property
    def kernelMatrix(self) -> np.ndarray:
        self._createKernel()
        kernel = np.asarray(self.kernel_image, dtype=np.float32)
        kernel /= np.sum(kernel)
        return kernel
