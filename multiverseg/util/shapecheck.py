""" 
Copied from https://github.com/JJGO/pylot/blob/master/pylot/util/shapecheck.py to reduce dependencies
"""
import string
from typing import Union, Tuple
import torch
import numpy as np
from tabulate import tabulate

NormalizedSpec = Tuple[Union[str, int], ...]
Spec = Union[str, NormalizedSpec]
Tensor = Union[np.ndarray, torch.Tensor]


class ShapeMismatchError(Exception):
    pass


class ShapeChecker:

    VALID_CHARS = set(
        string.ascii_lowercase + string.ascii_uppercase + string.digits + "+-/*" + "_"
    )

    def __init__(self):
        self._dims = {}

    @classmethod
    def check_sanitized(cls, spec):
        for value in spec:
            chars = set(map(str, value))
            if not chars.issubset(cls.VALID_CHARS):
                raise ValueError(
                    "Invalid input, shape ids must be [A-Z][a-z][0-9][+-/*]\n"
                    f"Received: {value}"
                )

    def _generate_error_msg(self, spec):
        pass

    @staticmethod
    def _is_composite(dim):
        return any(op in dim for op in "+-*/")

    def check(self, tensor: Tensor, spec: Spec, **known_dims):

        for dim, value in known_dims.items():
            if dim in self._dims and value != (current := self._dims[dim]):
                msg = f"Provided Dimension {dim}={value} is already known and has value {current}"
                raise ShapeMismatchError(msg)
            self._dims[dim] = value

        if isinstance(spec, str):
            spec = tuple(spec.split())

        self.check_sanitized(spec)

        shape = tuple(tensor.shape)

        if len(spec) != len(shape):
            msg = f"Mismatched number of dimensions, spec has {len(spec)} and tensor has {len(shape)}\n"
            msg += f"  Spec  {spec}\n"
            msg += f"  Shape {shape}\n"
            msg += f"  Known dims: {self._dims}"
            raise ShapeMismatchError(msg)

        eval_spec = list(spec)
        mismatches = []

        # We process simple dims first, then move onto composite
        for i, dim in sorted(enumerate(spec), key=lambda x: self._is_composite(x[1])):
            if isinstance(dim, int) or (isinstance(dim, str) and dim.isnumeric()):
                dim = int(dim)
            elif isinstance(dim, str):
                if dim == "_":  # Wildcard
                    dim = shape[i]
                elif dim in self._dims:
                    dim = self._dims[dim]
                elif self._is_composite(dim):
                    try:
                        dim = eval(dim, {}, self._dims)
                    except NameError:
                        pass
                else:
                    self._dims[dim] = shape[i]
                    dim = shape[i]
            else:
                raise TypeError(f"Found type {type(dim)}, dims must be str or int")

            if shape[i] != dim:
                mismatches.append(i)
            eval_spec[i] = dim

        if len(mismatches) > 0:

            idx = [""] + list(range(len(shape)))
            data = [
                ["Spec"] + list(spec),
                ["Eval"] + list(eval_spec),
                ["Tensor"] + list(shape),
            ]

            msg = (
                f"Tensor does not satisfy spec at dimensions: {sorted(mismatches)}:\n\n"
            )
            msg += tabulate(data, headers=idx, colalign=("right",) * len(idx))
            msg += "\n\n" + f"Known dims: {self._dims}"

            raise ShapeMismatchError(msg)