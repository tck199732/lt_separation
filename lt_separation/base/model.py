import inspect
import numpy as np
from abc import ABC, abstractmethod
from typing import Literal


def rosenbluth(
    epsilon: np.ndarray,
    phi: np.ndarray,
    sigma_T: np.ndarray,
    sigma_L: np.ndarray,
    sigma_LT: np.ndarray,
    sigma_TT: np.ndarray,
) -> np.ndarray:
    """
    Parameters
    ----------
    epsilon : np.ndarray
        The virtual photon polarization parameter.
    phi : np.ndarray
        The azimuthal angle in radians.
    sigma_T : np.ndarray
        The transverse cross-section component.
    sigma_L : np.ndarray
        The longitudinal cross-section component.
    sigma_LT : np.ndarray
        The longitudinal-transverse interference cross-section component.
    sigma_TT : np.ndarray
        The transverse-transverse interference cross-section component.
    Returns
    -------
    np.ndarray
        The differential cross-section evaluated at the given kinematics and components, up to a factor of dphi. Note that the outcome is in the same unit as in the input cross-sections.
    """
    return (1 / (2 * np.pi)) * (
        sigma_T
        + epsilon * sigma_L
        + np.sqrt(2 * epsilon * (1 + epsilon)) * sigma_LT * np.cos(phi)
        + epsilon * sigma_TT * np.cos(2 * phi)
    )


class XsectModel(ABC):
    def __init__(self):
        pass

    def __call__(self, epsilon, phi, *args, **kwargs):
        return self.evaluate(epsilon, phi, *args, **kwargs)

    @classmethod
    @abstractmethod
    def num_params(cls) -> int:
        """
        Return the number of parameters in this model. Each concrete model defines this.
        """

    @abstractmethod
    def sigma_L(self, *p, **kwargs):
        """
        Longitudinal cross-section component. The number of positional arguments *p should match num_params().
        """

    @abstractmethod
    def sigma_T(self, *p, **kwargs):
        """
        Transverse cross-section component. The number of positional arguments *p should match num_params().
        """

    @abstractmethod
    def sigma_LT(self, *p, **kwargs):
        """
        Interference cross-section component between longitudinal and transverse photons. The number of positional arguments *p should match num_params().
        """

    @abstractmethod
    def sigma_TT(self, *p, **kwargs):
        """
        Interference cross-section component between transverse photons. The number of positional arguments *p should match num_params().
        """

    @abstractmethod
    def dsigma_L(self, *p, **kwargs) -> list[np.ndarray]:
        """
        Derivative of longitudinal cross-section component with respect to parameters. The number of positional arguments *p should match num_params().
        Returns a list of derivatives with respect to each parameter.
        """

    @abstractmethod
    def dsigma_T(self, *p, **kwargs) -> list[np.ndarray]:
        """
        Derivative of transverse cross-section component with respect to parameters. The number of positional arguments *p should match num_params().
        Returns a list of derivatives with respect to each parameter.
        """

    @abstractmethod
    def dsigma_LT(self, *p, **kwargs) -> list[np.ndarray]:
        """
        Derivative of interference cross-section component between longitudinal and transverse photons with respect to parameters. The number of positional arguments *p should match num_params().
        Returns a list of derivatives with respect to each parameter.
        """

    @abstractmethod
    def dsigma_TT(self, *p, **kwargs) -> list[np.ndarray]:
        """
        Derivative of interference cross-section component between transverse photons with respect to parameters. The number of positional arguments *p should match num_params().
        Returns a list of derivatives with respect to each parameter.
        """

    def sigma(
        self,
        name: Literal["L", "T", "LT", "TT"],
        *p,
        **kwargs,
    ):
        """
        Evaluate a specific cross-section component by name. The number of positional arguments *p should match num_params().
        Parameters
        ----------
        name : Literal["L", "T", "LT", "TT"]
            The name of the cross-section component to evaluate. Should be one of "L", "T", "LT", or "TT".
        **kwargs
            Additional kinematic arguments to be passed to the cross-section function.
        Returns
        -------
        float or np.ndarray
            The evaluated cross-section component.
        """

        acceptable_names = ["L", "T", "LT", "TT"]
        if name not in acceptable_names:
            raise ValueError(
                f"Invalid sigma name: {name}. Must be one of {acceptable_names}."
            )
        method = getattr(self, f"sigma_{name}")

        if not callable(method):
            raise ValueError(f"Method {name} is not callable.")

        return method(*p, **kwargs)

    @property
    @abstractmethod
    def sigma_L_param_indices(self) -> list[int]:
        """
        Return a list of parameter indices used in sigma_L. Each concrete model defines this.
        """

    @property
    @abstractmethod
    def sigma_T_param_indices(self) -> list[int]:
        """
        Return a list of parameter indices used in sigma_T. Each concrete model defines this.
        """

    @property
    @abstractmethod
    def sigma_LT_param_indices(self) -> list[int]:
        """
        Return a list of parameter indices used in sigma_LT. Each concrete model defines this.
        """

    @property
    @abstractmethod
    def sigma_TT_param_indices(self) -> list[int]:
        """
        Return a list of parameter indices used in sigma_TT. Each concrete model defines this.
        """

    def param_indices(self, name: Literal["L", "T", "LT", "TT"]) -> list[int]:
        """
        Return a list of parameter indices used in the specified cross-section component.
        Parameters
        ----------
        name : Literal["L", "T", "LT", "TT"]
            The name of the cross-section component. Should be one of "L", "T", "LT", or "TT".
        Returns
        -------
        list[int]
            A list of parameter indices used in the specified cross-section component.
        """
        acceptable_names = ["L", "T", "LT", "TT"]
        if name not in acceptable_names:
            raise ValueError(
                f"Invalid sigma name: {name}. Must be one of {acceptable_names}."
            )
        return getattr(self, f"sigma_{name}_param_indices")

    @property
    @abstractmethod
    def param_constraints(self) -> list[tuple[float, float]]:
        """
        Return a list of (min, max) tuples for each parameter, or None if no constraints.
        The length of the list should match num_params().
        Each concrete model defines this.
        """

    def evaluate(self, epsilon, phi, *args, **kwargs):
        """
        Evaluate Rosenbluth formula in Python. Here, *args refer to parameters for composing the differential cross-section
        model, and **kwargs contain kinematic dependencies (e.g. t, Q2, W, theta).
        """

        sigma_L = self.sigma_L(*args, **kwargs)
        sigma_T = self.sigma_T(*args, **kwargs)
        sigma_LT = self.sigma_LT(*args, **kwargs)
        sigma_TT = self.sigma_TT(*args, **kwargs)

        return rosenbluth(epsilon, phi, sigma_T, sigma_L, sigma_LT, sigma_TT)

    def evaluate_from_params(self, epsilon, phi, params, **kwargs):
        """Convenience wrapper that accepts params as a sequence (instead of *args)."""
        return self.evaluate(epsilon, phi, *params, **kwargs)

    @classmethod
    @abstractmethod
    def cpp_variables(cls) -> list[str]:
        """
        Return a list of variable names (besides phi, epsilon)
        that this model depends on. Each concrete model defines this.
        """

    @abstractmethod
    def cpp_terms(self, *params, **kwargs) -> dict[str, str]:
        """Return dict of term-name -> C++ expression strings."""

    def cpp_script(self, name: str, *params) -> str:
        """
        Generate C++ function for the full Rosenbluth formula with given parameters.
        Parameters
        ----------
        name : str
            The name of the C++ function to generate.
        *params
            The model parameters to be used in the C++ function.
        Returns
        -------
        str
            A string containing the C++ function definition for the Rosenbluth formula.
        """

        terms = self.cpp_terms(*params)
        assert set(["L", "T", "LT", "TT"]).issubset(
            terms.keys()
        ), "cpp_terms must return all four terms: L, T, LT, TT"

        var_names = [v for v in self.cpp_variables() if not v in ["phi", "epsilon"]]
        func_args = ", ".join(
            ["float phi", "float epsilon"] + [f"float {v}" for v in var_names]
        )

        return f"""
        #include <cmath>
        float {name}({func_args}) {{
            float sigma_L  = {terms['L']};
            float sigma_T  = {terms['T']};
            float sigma_LT = {terms['LT']};
            float sigma_TT = {terms['TT']};
            return (1.0 / (2.0 * {np.pi})) *
                   (sigma_T + epsilon * sigma_L +
                    std::sqrt(2.0 * epsilon * (1.0 + epsilon)) * sigma_LT * std::cos(phi) +
                    epsilon * sigma_TT * std::cos(2.0 * phi));
        }}
        """

    def separated_sigma_cpp_script(
        self, name: Literal["L", "T", "LT", "TT"], *params
    ) -> str:
        """
        Generate C++ function for sigma_{name} with given parameters.
        Parameters
        ----------
        name : Literal["L", "T", "LT", "TT"]
            The name of the cross-section component to generate the C++ function for. Should be one of "L", "T", "LT", or "TT".
        *params
            The model parameters to be used in the C++ function.
        Returns
        -------
        str
            A string containing the C++ function definition for sigma_{name}.
        """
        var_names = [v for v in self.cpp_variables() if not v in ["phi", "epsilon"]]
        func_args = ", ".join([f"float {v}" for v in var_names])

        return f"""
        #include <cmath>
        float sigma_{name}({func_args}) {{
            return {self.cpp_terms(*params)[name]};
        }}
        """

    def build_wrapper(self, name: Literal["L", "T", "LT", "TT"], **kwargs):
        """
        Build a Python wrapper function for a specific cross-section component (e.g., sigma_L), with fixed kinematic arguments in **kwargs (e.g., t=0.3, W=3.14). The returned function will only take the relevant model parameters as arguments.
        Parameters
        ----------
        name : Literal["L", "T", "LT", "TT"]
            The name of the cross-section component to build the wrapper for. Should be one of "L", "T", "LT", or "TT".
        **kwargs
            Fixed kinematic arguments to be passed to the cross-section function.
        Returns
        -------
        function
            A Python function that takes only the relevant model parameters as arguments and returns the evaluated cross-section component.

        Example
        -------
        >>> model = YourXsectModel()
        >>> params = np.loadtxt("params.txt")
        >>> f = model.build_wrapper("L", t=0.3, W=3.14)
        >>> # suppose sigma_L uses parameters at indices [0, 2, 5]
        >>> val = f(params[0], params[2], params[5])
        >>> val_direct = model.sigma_L(*params, t=0.3, W=3.14)
        >>> assert np.allclose(val, val_direct)
        """

        acceptable_names = [
            "L",
            "T",
            "LT",
            "TT",
        ]
        if name not in acceptable_names:
            raise ValueError(
                f"Invalid sigma name: {name}. Must be one of {acceptable_names}."
            )

        method = getattr(self, f"sigma_{name}")
        used_indices = getattr(self, f"sigma_{name}_param_indices")
        arg_names = [f"p{i}" for i in used_indices]

        def f(*args):
            p = [0.0] * self.num_params()
            for val, idx in zip(args, used_indices):
                p[idx] = val
            return method(*p, **kwargs)

        sig = inspect.Signature(
            [
                inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for n in arg_names
            ]
        )
        f.__signature__ = sig
        f.__name__ = name + "_fit"
        return f

    def _get_variable(self, **kwargs):
        """
        Instance method wrapper that calls the classmethod version.
        """
        return self.__class__._get_variable_cls(**kwargs)

    @classmethod
    def _get_variable_cls(cls, **kwargs):
        """
        Class method to extract variables and their errors from kwargs based on cpp_variables().
        Works without an instance. Raises ValueError if any variable is missing.
        Returns a dict: {var: value, var_err: error array}.
        """
        if not hasattr(cls, "cpp_variables"):
            raise TypeError(f"{cls.__name__} must implement 'cpp_variables()' method.")

        missing_vars = [v for v in cls.cpp_variables() if v not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        out = {}
        for var in cls.cpp_variables():
            value = np.array(kwargs[var])

            # Extract error; default to zeros if not provided
            err = kwargs.get(f"{var}_err")
            err = np.array(err) if err is not None else np.zeros_like(value)

            out[var] = value
            out[f"{var}_err"] = err

        return out
