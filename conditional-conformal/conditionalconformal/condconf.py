from __future__ import annotations

from functools import lru_cache, partial
from typing import Callable

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog
from sklearn.metrics.pairwise import pairwise_kernels

FUNCTION_DEFAULTS = {"kernel": None, "gamma": 1, "lambda": 1}

class CondConf:
    def __init__(
            self,
            score_fn: Callable,
            Phi_fn: Callable,
            quantile_fn: Callable = None,
            infinite_params: dict = None,
            seed: int = 0
        ):
        """
        Constructs the CondConf object that caches relevant information for
        generating conditionally valid prediction sets.

        We define the score function and set of conditional guarantees
        that we care about in this function.

        Parameters
        ---------
        score_fn : Callable[np.ndarray, np.ndarray] -> np.ndarray
            Fixed (vectorized) conformity score function that takes in
            X and Y as inputs and returns S as output

        Phi_fn : Callable[np.ndarray] -> np.ndarray
            Function that defines finite basis set that we provide
            exact conditional guarantees over
        
        infinite_params : dict = {}
            Dictionary containing parameters for the RKHS component of the fit
            Valid keys are ('kernel', 'gamma', 'lambda')
                'kernel' should be a valid kernel name for sklearn.metrics.pairwise_kernels
                'gamma' is a hyperparameter for certain kernels
                'lambda' is the regularization penalty applied to the RKHS component
        """
        # Backwards compatibility for the existing experiments, which pass
        # `infinite_params` as the third positional argument.
        if isinstance(quantile_fn, dict) and infinite_params is None:
            infinite_params = quantile_fn
            quantile_fn = None

        self.score_fn = score_fn
        self.Phi_fn = Phi_fn
        self.quantile_fn = quantile_fn
        self.infinite_params = {**FUNCTION_DEFAULTS, **(infinite_params or {})}
        self.rng = np.random.default_rng(seed=seed)
        self._is_setup = False
        self._phi_transform = None
        self.quantile_calib = None
        self._kernel_matrix_calib = None
        self._kernel_chol_calib = None

    def _check_is_setup(self):
        if not self._is_setup:
            raise RuntimeError("Call setup_problem before requesting predictions or coverage.")

    @staticmethod
    def _coerce_features(x: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"{name} must have shape (n_samples, n_features); got {arr.shape}.")
        return arr

    @staticmethod
    def _coerce_vector(values: np.ndarray, expected_len: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape[0] != expected_len:
            raise ValueError(
                f"{name} must return one value per sample; expected {expected_len}, got {arr.shape[0]}."
            )
        return arr

    def _evaluate_phi(self, x: np.ndarray) -> np.ndarray:
        x_arr = self._coerce_features(x, "x")
        phi = np.asarray(self.Phi_fn(x_arr), dtype=float)
        if phi.ndim == 1:
            phi = phi.reshape(-1, 1)
        if phi.ndim != 2 or phi.shape[0] != len(x_arr):
            raise ValueError(
                "Phi_fn must return a 2D array with one row per sample."
            )
        if self._phi_transform is not None:
            phi = phi @ self._phi_transform
        return phi

    def _get_quantiles(self, quantile: float, x_test: np.ndarray):
        self._check_is_setup()
        x_test = self._coerce_features(x_test, "x_test")
        if len(x_test) != 1:
            raise ValueError("This method expects a single test point.")

        if quantile is None:
            if self.quantile_fn is None:
                raise ValueError("quantile must be specified when quantile_fn is not provided.")
            quantile_test = self._coerce_vector(
                self.quantile_fn(x_test), 1, "quantile_fn"
            )[0]
            quantiles = np.concatenate(
                (self.quantile_calib, [[quantile_test]]),
                axis=0,
            )
        else:
            quantile_test = float(quantile)
            quantiles = np.full((len(self.scores_calib) + 1, 1), quantile_test)
        return quantile_test, quantiles

    def setup_problem(
            self,
            x_calib : np.ndarray,
            y_calib : np.ndarray
    ):
        """
        setup_problem sets up the final fitting problem for a 
        particular calibration set

        The resulting cvxpy Problem object is stored inside the CondConf parent.

        Arguments
        ---------
        x_calib : np.ndarray
            Covariate data for the calibration set

        y_calib : np.ndarray
            Labels for the calibration set
        """
        x_calib = self._coerce_features(x_calib, "x_calib")
        y_calib = np.asarray(y_calib)
        if len(y_calib) != len(x_calib):
            raise ValueError(
                f"x_calib and y_calib must contain the same number of samples; got {len(x_calib)} and {len(y_calib)}."
            )

        self.x_calib = x_calib
        self.y_calib = y_calib
        self._phi_transform = None
        self._kernel_matrix_calib = None
        self._kernel_chol_calib = None
        self._get_calibration_solution.cache_clear()

        if self.quantile_fn is not None and self.infinite_params.get("kernel") is not None:
            raise ValueError(
                "The RKHS solver currently supports only a scalar quantile; quantile_fn is unsupported."
            )

        phi_calib = self._evaluate_phi(x_calib)

        _, s, Vt = np.linalg.svd(phi_calib, full_matrices=False)
        
        # Set a tolerance to decide which singular values are nonzero
        tol = 1e-10
        r = np.sum(s > tol)

        if r == 0:
            self._phi_transform = np.zeros((phi_calib.shape[1], 1))
            phi_calib = np.zeros((len(x_calib), 1))
        elif r < phi_calib.shape[1]:
            self._phi_transform = Vt.T[:, :r]
            phi_calib = phi_calib @ self._phi_transform
        
        self.phi_calib = phi_calib
        self.scores_calib = self._coerce_vector(
            self.score_fn(x_calib, y_calib),
            len(x_calib),
            "score_fn",
        )
        if self.infinite_params.get("kernel") is not None:
            self._kernel_matrix_calib, self._kernel_chol_calib = _get_kernel_matrix(
                self.x_calib,
                self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]),
                self.infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"]),
            )

        if self.quantile_fn is not None:
            self.quantile_calib = self._coerce_vector(
                self.quantile_fn(x_calib),
                len(x_calib),
                "quantile_fn",
            ).reshape(-1, 1)
        else:
            self.quantile_calib = None

        self.cvx_problem = setup_cvx_problem(
            self.x_calib,
            self.scores_calib,
            self.phi_calib,
            self.infinite_params,
            kernel_chol=self._kernel_chol_calib,
        )
        self._is_setup = True


    @lru_cache()
    def _get_calibration_solution(
            self,
            quantile : float
    ):
        self._check_is_setup()
        S = self.scores_calib.reshape(-1, 1)
        Phi = self.phi_calib.astype(float)

        if quantile is None:
            bounds = np.concatenate((self.quantile_calib - 1, self.quantile_calib), axis=1)
        else:
            bounds = np.asarray([quantile - 1, quantile])
            bounds = np.tile(bounds.reshape(1,-1), (len(S), 1))

        res = _solve_quantile_lp(-1 * S, Phi, bounds, method="highs")
        primal_vars = -1 * res.eqlin.marginals.reshape(-1,1)
        dual_vars = res.x.reshape(-1,1)

        residuals = S - (Phi @ primal_vars)
        interpolated_pts = np.isclose(residuals, 0)

        # if I didn't converge to a solution that interpolates at least Phi.shape[1] pts, 
        # I need to manually find one via a modified simplex iteration
        if interpolated_pts.sum() < Phi.shape[1]:
            num_to_add = Phi.shape[1] - interpolated_pts.sum()
            for _ in range(num_to_add):
                candidate_pts = interpolated_pts.copy().flatten()

                # find candidate idx for interpolation, e.g., new covariate that is
                # linearly independent of the previously interpolated points
                Q, _ = np.linalg.qr(Phi[candidate_pts].T)
                projections = Phi @ Q @ Q.T
                norms = np.linalg.norm(Phi - projections, axis=1)
                candidate_idx = np.where(norms > 1e-5)[0][0]
                candidate_pts[candidate_idx] = True

                # find direction to solution that would interpolate the new point
                gamma, _, _, _ = np.linalg.lstsq(Phi[candidate_pts], S[candidate_pts], rcond=None)
                direction = gamma.reshape(-1,1) - primal_vars
                step_sizes = residuals / (Phi @ direction)

                # check the non-basic indices for which a step in this direction could have led to interpolation
                # e.g., those for which the step size is positive and the point is not already interpolated
                positive_indices = np.where((step_sizes > 0) & ~interpolated_pts)[0]

                # take smallest possible step that would lead to interpolation
                primal_vars += np.min(step_sizes[positive_indices]) * direction

                residuals = S - (Phi @ primal_vars)
                interpolated_pts = np.isclose(residuals, 0)

        return dual_vars, primal_vars
    
    def _compute_exact_cutoff(
            self,
            quantiles,
            primals,
            duals,
            phi_test,
            dual_threshold
    ):
        def get_current_basis(primals, duals, Phi, S, quantiles):
            interp_bools = np.logical_and(~np.isclose(duals, quantiles - 1), ~np.isclose(duals, quantiles))
            if np.sum(interp_bools) == Phi.shape[1]:
                return interp_bools
            preds = (Phi @ primals).flatten()
            active_indices = np.where(interp_bools)[0]
            interp_indices = np.where(np.isclose(np.abs(S - preds), 0))[0]
            diff_indices = np.setdiff1d(interp_indices, active_indices)
            num_missing = Phi.shape[1] - np.sum(interp_bools)
            if num_missing < len(diff_indices):
                from itertools import combinations
                for cand_indices in combinations(diff_indices, num_missing):
                    cand_phi = Phi[np.concatenate((active_indices, cand_indices))]
                    if np.isfinite(np.linalg.cond(cand_phi)):
                        interp_bools[np.asarray(cand_indices)] = True
                        break
            else:
                interp_bools[diff_indices] = True
            if np.sum(interp_bools) != Phi.shape[1]:
                raise ValueError("Initial basis could not be found - retry with exact=False.")
            return interp_bools
        
        if np.allclose(phi_test, 0):
            return np.inf if quantiles[-1] >= 0.5 else -np.inf
                
        basis = get_current_basis(primals, duals, self.phi_calib, self.scores_calib, quantiles[:-1])
        S_test = phi_test @ primals

        duals = np.concatenate((duals.flatten(), [0]))
        basis = np.concatenate((basis.flatten(), [False]))
        phi = np.concatenate((self.phi_calib, phi_test.reshape(1,-1)), axis=0)
        S = np.concatenate((self.scores_calib.reshape(-1,1), S_test.reshape(-1,1)), axis=0)

        candidate_idx = phi.shape[0] - 1
        num_iters = 0
        while True:
            # get direction vector for dual variable step
            direction = -1 * np.linalg.solve(phi[basis].T, phi[candidate_idx].reshape(-1,1)).flatten()

            # only consider non-zero entries of the direction vector
            active_indices = ~np.isclose(direction, 0)
            active_direction = direction[active_indices]
            active_basis = basis.copy()
            active_basis[np.where(basis)[0][~active_indices]] = False

            positive_step = True if duals[candidate_idx] <= 0 else False
            if candidate_idx == phi.shape[0] - 1:
                positive_step = True if dual_threshold >= 0 else False

            if positive_step:
                gap_to_bounds = np.maximum(
                    (quantiles[active_basis].flatten() - duals[active_basis]) / active_direction,
                    ((quantiles[active_basis].flatten() - 1) - duals[active_basis]) / active_direction
                )
                step_size = np.min(gap_to_bounds)
                departing_idx = np.where(active_basis)[0][np.argmin(gap_to_bounds)]
            else:
                gap_to_bounds = np.minimum(
                    (quantiles[active_basis].flatten() - duals[active_basis]) / active_direction,
                    ((quantiles[active_basis].flatten() - 1) - duals[active_basis]) / active_direction
                )
                step_size = np.max(gap_to_bounds)
                departing_idx = np.where(active_basis)[0][np.argmax(gap_to_bounds)]
            candidate_quantile = float(np.asarray(quantiles[candidate_idx]).reshape(-1)[0])
            candidate_dual = float(np.asarray(duals[candidate_idx]).reshape(-1)[0])
            step_size_clip = float(np.clip(
                step_size,
                a_max=candidate_quantile - candidate_dual,
                a_min=(candidate_quantile - 1) - candidate_dual,
            ))

            duals[basis] += step_size_clip * direction
            duals[candidate_idx] += step_size_clip
            # print("Current value of final dual", duals[-1], "target threshold", dual_threshold)

            if dual_threshold > 0 and duals[-1] > dual_threshold:
                break

            if dual_threshold < 0 and duals[-1] < dual_threshold:
                break

            if step_size_clip == step_size:
                basis[departing_idx] = False
                basis[candidate_idx] = True
            
            if np.isclose(duals[-1], dual_threshold):
                break

            # TODO: make this a SMW update and reuse in the direction vector calc...
            reduced_A = np.linalg.solve(phi[basis].T, phi[~basis].T)
            reduced_costs = (S[~basis].T - S[basis].T @ reduced_A).flatten()
            bottom = reduced_A[-1]
            bottom[np.isclose(bottom, 0)] = np.inf
            req_change = reduced_costs / bottom
            if dual_threshold >= 0:
                ignore_entries = (np.isclose(bottom, 0) | np.asarray(req_change <= 1e-5))  
            else:
                ignore_entries = (np.isclose(bottom, 0) | np.asarray(req_change >= -1e-5))  
            if np.sum(~ignore_entries) == 0:
                S[-1] = np.inf if quantiles[-1] >= 0.5 else -np.inf
                break
            if dual_threshold >= 0:
                candidate_idx = np.where(~basis)[0][np.where(~ignore_entries, req_change, np.inf).argmin()]
                S[-1] += np.min(req_change[~ignore_entries])
            else:
                candidate_idx = np.where(~basis)[0][np.where(~ignore_entries, req_change, -np.inf).argmax()]
                S[-1] += np.max(req_change[~ignore_entries])
            num_iters += 1
            if num_iters > 10000:
                S[-1] = np.inf if dual_threshold > 0 else -1 * np.inf
        return S[-1]

    def predict(
            self,
            quantile : float,
            x_test : np.ndarray,
            score_inv_fn : Callable,
            S_min : float = None,
            S_max : float = None,
            randomize : bool = False,
            exact : bool = True,
            threshold : float = None
    ):
        """
        Returns the (conditionally valid) prediction set for a given 
        test point

        Arguments
        ---------
        quantile : float
            Nominal quantile level
        x_test : np.ndarray
            Single test point
        score_inv_fn : Callable[float, np.ndarray] -> .
            Function that takes in a score threshold S^* and test point x and 
            outputs all values of y such that S(x, y) <= S^*
        S_min : float = None
            Lower bound (if available) on the conformity scores
        S_max : float = None
            Upper bound (if available) on the conformity scores
        randomize : bool = False
            Randomize prediction set for exact coverage
        exact : bool = True
            Avoid binary search and compute threshold exactly

        Returns
        -------
        prediction_set
        """
        self._check_is_setup()
        x_test = self._coerce_features(x_test, "x_test")
        quantile_test, quantiles = self._get_quantiles(quantile, x_test)
        if threshold is None:
            if randomize:
                threshold = float(self.rng.uniform(low=quantile_test - 1, high=quantile_test))
            else:
                if quantile_test < 0.5:
                    threshold = quantile_test - 1
                else:
                    threshold = quantile_test
        
        if exact:
            if self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel']):
                raise ValueError("Exact computation doesn't support RKHS quantile regression for now.")
            try:
                if np.allclose(quantiles[0], quantiles):
                    naive_duals, naive_primals = self._get_calibration_solution(
                        quantiles.flatten()[0]
                    )
                else:
                    naive_duals, naive_primals = self._get_calibration_solution(
                        None
                    )
                score_cutoff = self._compute_exact_cutoff(
                    quantiles,
                    naive_primals,
                    naive_duals,
                    self._evaluate_phi(x_test),
                    threshold
                )
            except (np.linalg.LinAlgError, RuntimeError, ValueError):
                _solve = partial(_solve_dual, gcc=self, x_test=x_test, quantiles=quantiles, threshold=threshold)
                if S_min is None:
                    S_min = float(np.min(self.scores_calib))
                if S_max is None:
                    S_max = float(np.max(self.scores_calib))
                scale = max(abs(S_min), abs(S_max), 1.0)
                lower, upper = binary_search(_solve, min(S_min, -scale), max(S_max, scale) * 2)
                pivot = lower if quantile_test < 0.5 else upper
                score_cutoff = self._get_threshold(pivot, x_test, quantiles)
        else:
            _solve = partial(_solve_dual, gcc=self, x_test=x_test, quantiles=quantiles, threshold=threshold)

            if S_min is None:
                S_min = float(np.min(self.scores_calib))
            if S_max is None:
                S_max = float(np.max(self.scores_calib))
            scale = max(abs(S_min), abs(S_max), 1.0)
            lower, upper = binary_search(_solve, min(S_min, -scale), max(S_max, scale) * 2)

            if quantile_test < 0.5:
                score_cutoff = self._get_threshold(lower, x_test, quantiles)
            else:
                score_cutoff = self._get_threshold(upper, x_test, quantiles)

        return score_inv_fn(score_cutoff, x_test)

    def estimate_coverage(
            self,
            quantile : float,
            weights : np.ndarray,
            x : np.ndarray = None
    ):
        """
        estimate_coverage estimates the true percentile of the issued estimate of the
        conditional quantile under the covariate shift induced by 'weights'

        If we are ostensibly estimating the 0.95-quantile using an RKHS fit, we may 
        determine using our theory that the true percentile of this estimate is only 0.93

        Arguments
        ---------
        quantile : float
            Nominal quantile level
        weights : np.ndarray
            RKHS weights for tilt under which the coverage is estimated
        x : np.ndarray = None
            Points for which the RKHS weights are defined. If None, we assume
            that weights corresponds to x_calib

        Returns
        -------
        estimated_alpha : float
            Our estimate for the realized quantile level
        """
        self._check_is_setup()
        if self.infinite_params.get("kernel") is None:
            raise ValueError("estimate_coverage is only defined for the RKHS solver path.")
        weights = weights.reshape(-1,1)
        prob = setup_cvx_problem_calib(
            quantile,
            self.x_calib,
            self.scores_calib,
            self.phi_calib,
            self.infinite_params,
            kernel_chol=self._kernel_chol_calib,
        )
        _solve_cvx_problem(prob)

        fitted_weights = prob.var_dict['weights'].value
        if x is not None:
            K = pairwise_kernels(
                X=x,
                Y=self.x_calib,
                metric=self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]),
                gamma=self.infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])
            )
        else:
            K = self._kernel_matrix_calib
        inner_prod = weights.T @ K @ fitted_weights
        expectation = np.mean(weights.T @ K)
        #penalty = self.infinite_params['lambda'] * (inner_prod / expectation)
        penalty = (1 / len(self.x_calib)) * (inner_prod / expectation)
        return quantile - penalty
    
    def predict_naive(
            self,
            quantile : float,
            x : np.ndarray,
            score_inv_fn : Callable
    ):
        """
        If we do not wish to include the imputed data point, we can sanity check that
        the regression is appropriately adaptive to the conditional variability in the data
        by running a quantile regression on the calibration set without any imputation. 
        When n_calib is large and the fit is stable, we expect these two sets to nearly coincide.

        Arguments
        ---------
        quantile : float
            Nominal quantile level
        x : np.ndarray
            Set of points for which we are issuing prediction sets
        score_inv_fn : Callable[np.ndarray, np.ndarray] -> np.ndarray
            Vectorized function that takes in a score threshold S^* and test point x and 
            outputs all values of y such that S(x, y) <= S^*
        
        Returns
        -------
        prediction_sets
        
        """
        self._check_is_setup()
        x = self._coerce_features(x, "x")
        if quantile is None and self.quantile_calib is None:
            raise ValueError("quantile must be provided when quantile_fn is not set.")
        
        if self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel']):
            prob = setup_cvx_problem_calib(
                quantile,
                self.x_calib,
                self.scores_calib,
                self.phi_calib,
                self.infinite_params,
                kernel_chol=self._kernel_chol_calib,
            )
            _solve_cvx_problem(prob)

            eta = prob.var_dict['weights'].value
            beta = prob.constraints[-1].dual_value
            K = pairwise_kernels(
                X=x,
                Y=self.x_calib,
                metric=self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]),
                gamma=self.infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])
            )
            threshold = (
                _kernel_dual_scale(self.infinite_params, len(self.x_calib))
                * (K @ eta)
                + self._evaluate_phi(x) @ beta
            )
        else:
            if quantile is None:
                _, beta = self._get_calibration_solution(None)
            else:
                _, beta = self._get_calibration_solution(float(quantile))
            threshold = self._evaluate_phi(x) @ beta

        return score_inv_fn(threshold, x)
    
    def verify_coverage(
            self,
            x : np.ndarray,
            y : np.ndarray,
            quantile : float,
            randomize : bool = False,
            resolve : bool = False,
            return_dual : bool = False,
            eps : float = 0.001
    ):
        """
        In some experiments, we may simply be interested in verifying the coverage of our method.
        In this case, we do not need to binary search for the threshold S^*, but only need to verify that
        S <= f_S(x) for the true value of S. This function implements this check for test points
        denoted by x and y

        Arguments
        ---------
        x : np.ndarray
            A vector of test covariates
        y : np.ndarray
            A vector of test labels
        quantile : float
            Nominal quantile level
        resolve : bool
            Resolve LP/QP with posited value to determine coverage

        Returns
        -------
        coverage_booleans : np.ndarray
        """
        self._check_is_setup()
        x = self._coerce_features(x, "x")
        y = np.asarray(y)
        if len(y) != len(x):
            raise ValueError(
                f"x and y must contain the same number of samples; got {len(x)} and {len(y)}."
            )

        covers = []
        duals = []

        if self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel']):        
            for x_val, y_val in zip(x, y):
                x_row = x_val.reshape(1, -1)
                quantile_test, quantiles = self._get_quantiles(quantile, x_row)
                S_true = self._coerce_vector(self.score_fn(x_row, np.asarray([y_val])), 1, "score_fn")
                eta = self._get_dual_solution(S_true[0], x_row, quantiles)
                dual_value = float(np.asarray(eta).reshape(-1)[-1])
                if randomize:
                    threshold = float(self.rng.uniform(low=quantile_test - 1, high=quantile_test))
                elif quantile_test > 0.5:
                    threshold = quantile_test - eps
                else:
                    threshold = quantile_test - 1 + eps
                if quantile_test > 0.5:
                    covers.append(dual_value < threshold)
                else:
                    covers.append(dual_value > threshold)
                duals.append(dual_value)

        else:
            for x_val, y_val in zip(x, y):
                x_row = x_val.reshape(1, -1)
                quantile_test, quantiles = self._get_quantiles(quantile, x_row)
                if randomize:
                    threshold = float(self.rng.uniform(low=quantile_test - 1, high=quantile_test))
                elif quantile_test > 0.5:
                    threshold = quantile_test
                else:
                    threshold = quantile_test - 1

                S_true = self._coerce_vector(self.score_fn(x_row, np.asarray([y_val])), 1, "score_fn")
                if resolve:
                    eta = self._get_dual_solution(S_true[0], x_row, quantiles)
                    dual_value = float(np.asarray(eta).reshape(-1)[-1])
                    if quantile_test > 0.5:
                        covers.append(dual_value < threshold)
                    else:
                        covers.append(dual_value > threshold)
                    duals.append(dual_value)
                else:
                    naive_duals, naive_primals = self._get_calibration_solution(
                        None if quantile is None else float(quantile)
                    )
                    score_cutoff = self._compute_exact_cutoff(
                        quantiles,
                        naive_primals,
                        naive_duals,
                        self._evaluate_phi(x_row),
                        threshold
                    )
                    score_value = float(S_true[0])
                    score_cutoff = float(np.asarray(score_cutoff).reshape(-1)[0])
                    if quantile_test > 0.5:
                        covers.append(score_value < score_cutoff)
                    else:
                        covers.append(score_value > score_cutoff)
                    duals.append(np.nan)
        if return_dual:
            return np.asarray(covers), np.asarray(duals)
        return np.asarray(covers)
  
    def _get_dual_solution(
        self,
        S : float,
        x : np.ndarray,
        quantiles : np.ndarray
    ):
        self._check_is_setup()
        x = self._coerce_features(x, "x")
        if len(x) != 1:
            raise ValueError("This method expects a single test point.")
        quantiles = _normalize_quantiles(quantiles, len(self.scores_calib) + 1)

        if self.infinite_params.get("kernel", FUNCTION_DEFAULTS['kernel']):
            prob = finish_dual_setup(
                self.cvx_problem,
                S,
                x,
                quantiles[-1][0],
                self._evaluate_phi(x),
                self.x_calib,
                self.infinite_params,
                kernel_chol=self._kernel_chol_calib,
            )
            _solve_cvx_problem(prob)
            return prob.var_dict['weights'].value
        else:
            S = np.concatenate([self.scores_calib, [S]])
            Phi = np.concatenate([self.phi_calib, self._evaluate_phi(x)], axis=0)
            bounds = np.concatenate((quantiles - 1, quantiles), axis=1)
            res = _solve_quantile_lp(
                -1 * S,
                Phi,
                bounds,
                method="highs-ds",
                options={"presolve": False},
            )
            eta = res.x
        return eta
    
    
    def _get_primal_solution(
        self,
        S : float,
        x : np.ndarray,
        quantiles : np.ndarray
    ):
        self._check_is_setup()
        x = self._coerce_features(x, "x")
        if len(x) != 1:
            raise ValueError("This method expects a single test point.")
        quantiles = _normalize_quantiles(quantiles, len(self.scores_calib) + 1)

        if self.infinite_params.get("kernel", FUNCTION_DEFAULTS['kernel']):
            prob = finish_dual_setup(
                self.cvx_problem,
                S,
                x,
                quantiles[-1][0],
                self._evaluate_phi(x),
                self.x_calib,
                self.infinite_params,
                kernel_chol=self._kernel_chol_calib,
            )
            _solve_cvx_problem(prob)

            weights = prob.var_dict['weights'].value
            beta = prob.constraints[-1].dual_value
        else:
            S = np.concatenate([self.scores_calib, [S]])
            Phi = np.concatenate([self.phi_calib, self._evaluate_phi(x)], axis=0)
            bounds = np.concatenate((quantiles - 1, quantiles), axis=1)
            res = _solve_quantile_lp(
                -1 * S,
                Phi,
                bounds,
                method="highs-ds",
                options={"presolve": False},
            )
            beta = -1 * res.eqlin.marginals
            weights = None
        return beta, weights
    
    def _get_threshold(
        self,
        S : float,
        x : np.ndarray,
        quantiles : np.ndarray
    ):
        beta, weights = self._get_primal_solution(S, x, quantiles)

        x = self._coerce_features(x, "x")
        threshold = self._evaluate_phi(x) @ beta
        if self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel']):
            K = pairwise_kernels(
                X=np.concatenate([self.x_calib, x.reshape(1,-1)], axis=0),
                Y=np.concatenate([self.x_calib, x.reshape(1,-1)], axis=0),
                metric=self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]),
                gamma=self.infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])
            )
            threshold = (
                _kernel_dual_scale(self.infinite_params, len(self.x_calib) + 1)
                * (K @ weights)[-1]
                + threshold
            )
        return threshold


def _normalize_quantiles(quantiles: np.ndarray, expected_len: int) -> np.ndarray:
    arr = np.asarray(quantiles, dtype=float)
    if arr.ndim == 0:
        arr = np.full((expected_len, 1), float(arr))
    else:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] != expected_len:
        raise ValueError(
            f"Expected {expected_len} quantile values, got {arr.shape[0]}."
        )
    return arr


def _kernel_dual_scale(infinite_params: dict, n_calib: int) -> float:
    return 1.0 / (2 * n_calib * infinite_params.get("lambda", FUNCTION_DEFAULTS["lambda"]))


def _solve_cvx_problem(prob: cp.Problem) -> None:
    if "MOSEK" in cp.installed_solvers():
        prob.solve(solver="MOSEK", verbose=False, warm_start=True)
    else:
        prob.solve(solver="OSQP", verbose=False, warm_start=True)


def _solve_quantile_lp(c, phi, bounds, method="highs", options=None):
    phi = np.asarray(phi, dtype=float)
    if phi.ndim == 1:
        phi = phi.reshape(-1, 1)

    result = linprog(
        c=np.asarray(c, dtype=float).reshape(-1),
        A_eq=phi.T,
        b_eq=np.zeros((phi.shape[1],), dtype=float),
        bounds=np.asarray(bounds, dtype=float),
        method=method,
        options=options,
    )
    if not result.success:
        raise RuntimeError(f"Linear program failed: {result.message}")
    return result


def binary_search(func, lower, upper, tol=1e-3):
    lower, upper = float(lower), float(upper)
    assert (upper + tol) > upper
    while (upper - lower) > tol:
        mid = (lower + upper) / 2
        if func(mid) > 0:
            upper = mid
        else:
            lower = mid
    return lower, upper


def _solve_dual(S, gcc, x_test, quantiles, threshold=None):
    quantiles = _normalize_quantiles(quantiles, len(gcc.scores_calib) + 1)
    if gcc.infinite_params.get('kernel', None):
        prob = finish_dual_setup(
            gcc.cvx_problem,
            S,
            x_test,
            quantiles[-1][0],
            gcc._evaluate_phi(x_test),
            gcc.x_calib,
            gcc.infinite_params,
            kernel_chol=gcc._kernel_chol_calib,
        )
        _solve_cvx_problem(prob)
        eta = prob.var_dict['weights'].value
    else:
        S = np.concatenate([gcc.scores_calib, [S]], dtype=float)
        Phi = np.concatenate([gcc.phi_calib, gcc._evaluate_phi(x_test)], axis=0, dtype=float)
        bounds = np.concatenate((quantiles - 1, quantiles), axis=1)
        res = _solve_quantile_lp(
            -1 * S,
            Phi,
            bounds,
            method="highs",
            options={"presolve": False},
        )
        eta = res.x

    if threshold is None:
        if quantiles[-1][0] < 0.5:
            threshold = quantiles[-1][0] - 1
        else:
            threshold = quantiles[-1][0]
    return eta[-1] - threshold


def setup_cvx_problem(
    x_calib, 
    scores_calib, 
    phi_calib,
    infinite_params = None,
    kernel_chol = None,
):
    infinite_params = {**FUNCTION_DEFAULTS, **(infinite_params or {})}
    n_calib = len(scores_calib)
    if phi_calib is None:
        phi_calib = np.ones((n_calib,1))
        
    eta = cp.Variable(name="weights", shape=n_calib + 1)

    quantile = cp.Parameter(name="quantile")
        
    scores_const = cp.Constant(scores_calib.reshape(-1,1))
    scores_param = cp.Parameter(name="S_test", shape=(1,1))
    scores = cp.vstack([scores_const, scores_param])
    
    Phi_calibration = cp.Constant(phi_calib)
    Phi_test = cp.Parameter(name="Phi_test", shape=(1, phi_calib.shape[1]))
    Phi = cp.vstack([Phi_calibration, Phi_test])

    kernel = infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"])
    gamma = infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])

    if kernel is None: # no RKHS fitting
        constraints = [
            (quantile - 1) <= eta,
            quantile >= eta,
            eta.T @ Phi == 0
        ]
        prob = cp.Problem(
            cp.Minimize(-1 * cp.sum(cp.multiply(eta, cp.vec(scores, order="F")))),
            constraints
        )
    else: # RKHS fitting
        radius = cp.Parameter(name="radius", nonneg=True)        
        L_11 = kernel_chol
        if L_11 is None:
            _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)
    
        L_11_const = cp.Constant(
            np.hstack([L_11, np.zeros((L_11.shape[0], 1))])
            )
        L_21_22_param = cp.Parameter(name="L_21_22", shape=(1, n_calib + 1))
        L = cp.vstack([L_11_const, L_21_22_param])
    
        C = radius / (2 * (n_calib + 1))

        # this is really C * (quantile - 1) and C * quantile
        constraints = [
            (quantile - 1) <= eta,
            quantile >= eta,
            eta.T @ Phi == 0]
        prob = cp.Problem(
                    cp.Minimize(0.5 * C * cp.sum_squares(L.T @ eta) - cp.sum(cp.multiply(eta, cp.vec(scores, order="F")))),
                    constraints
                )
    return prob


def _get_kernel_matrix(x_calib, kernel, gamma):
    K = pairwise_kernels(
        X=x_calib,
        metric=kernel,
        gamma=gamma
    ) + 1e-5 * np.eye(len(x_calib))

    K_chol = np.linalg.cholesky(K)
    return K, K_chol


def finish_dual_setup(
    prob : cp.Problem,
    S : np.ndarray, 
    X : np.ndarray,
    quantile : float,
    Phi : np.ndarray,
    x_calib : np.ndarray,
    infinite_params = None,
    kernel_chol = None,
):
    infinite_params = {**FUNCTION_DEFAULTS, **(infinite_params or {})}
    prob.param_dict['S_test'].value = np.asarray([[S]])
    prob.param_dict['Phi_test'].value = Phi.reshape(1,-1)
    prob.param_dict['quantile'].value = quantile

    kernel = infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel'])
    gamma = infinite_params.get('gamma', FUNCTION_DEFAULTS['gamma'])
    radius = 1 / infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda'])

    if kernel is not None:
        K_12 = pairwise_kernels(
            X=np.concatenate([x_calib, X.reshape(1,-1)], axis=0),
            Y=X.reshape(1,-1),
            metric=kernel,
            gamma=gamma
            )

        if 'K_12' in prob.param_dict:
            prob.param_dict['K_12'].value = K_12[:-1]
            prob.param_dict['K_21'].value = K_12.T

        L_11 = kernel_chol
        if L_11 is None:
            _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)
        K_22 = pairwise_kernels(
            X=X.reshape(1,-1),
            metric=kernel,
            gamma=gamma
            )
        L_21 = np.linalg.solve(L_11, K_12[:-1]).T
        L_22 = K_22 - L_21 @ L_21.T
        L_22[L_22 < 0] = 0
        L_22 = np.sqrt(L_22)    
        prob.param_dict['L_21_22'].value = np.hstack([L_21, L_22])
    
        prob.param_dict['radius'].value = radius

        # update quantile definition for silly cvxpy reasons
        prob.param_dict['quantile'].value = quantile
        #prob.param_dict['quantile'].value *= radius / (len(x_calib) + 1)
    
    return prob

def setup_cvx_problem_calib(
    quantile,
    x_calib, 
    scores_calib, 
    phi_calib,
    infinite_params = None,
    kernel_chol = None,
):
    infinite_params = {**FUNCTION_DEFAULTS, **(infinite_params or {})}
    n_calib = len(scores_calib)
    if phi_calib is None:
        phi_calib = np.ones((n_calib,1))
        
    eta = cp.Variable(name="weights", shape=n_calib)
        
    scores = cp.Constant(scores_calib.reshape(-1,1))
    
    Phi = cp.Constant(phi_calib)

    kernel = infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"])
    gamma = infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])

    if kernel is None: # no RKHS fitting
        constraints = [
            (quantile - 1) <= eta,
            quantile >= eta,
            eta.T @ Phi == 0
        ]
        prob = cp.Problem(
            cp.Minimize(-1 * cp.sum(cp.multiply(eta, cp.vec(scores, order="F")))),
            constraints
        )
    else: # RKHS fitting
        radius = 1 / infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda'])
        L = kernel_chol
        if L is None:
            _, L = _get_kernel_matrix(x_calib, kernel, gamma)
    
        C = radius / (2 * n_calib)

        constraints = [
             (quantile - 1) <= eta,
             quantile >= eta,
            eta.T @ Phi == 0]
        prob = cp.Problem(
                    cp.Minimize(0.5 * C * cp.sum_squares(L.T @ eta) - cp.sum(cp.multiply(eta, cp.vec(scores, order="F")))),
                    constraints
                )
    return prob
