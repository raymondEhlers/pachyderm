""" Interface for fitting with Minuit.

Some useful references on understanding binned vs unbinned fitting, likelihoods, etc:

- https://www.nbi.dk/~petersen/Teaching/Stat2015/Week3/week3.html
- https://www.nbi.dk/~petersen/Teaching/Stat2015/Week3/LikelihoodFit.py
- https://www.nbi.dk/~petersen/Teaching/Stat2017/Week3/AS2017_1205_Likelihood.pdf

See also the appendix of my thesis.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""
from __future__ import annotations

__all__ = [
    "base",
    "cost_function",
    "function",
    "integration",
]

from .base import (  # noqa: F401
    BaseFitResult,
    FitFailed,
    FitResult,
    FuncCode,
    calculate_function_errors,
    chi_squared_probability,
    evaluate_gradient,
    extract_function_values,
)
from .cost_function import (  # noqa: F401
    BinnedChiSquared,
    BinnedLogLikelihood,
    ChiSquared,
    CostFunctionBase,
    LogLikelihood,
    SimultaneousFit,
    binned_chi_squared_safe_for_zeros,
)
from .function import AddPDF, DividePDF, MultiplyPDF, SubtractPDF, extended_gaussian, gaussian  # noqa: F401
from .integration import Fit as Fit
from .integration import T_FitArguments as T_FitArguments
from .integration import fit_with_minuit as fit_with_minuit
