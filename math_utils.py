import numpy as np
from scipy.special import ndtr
from scipy.optimize import least_squares


def _normal_price(dc: np.array,
                  fut: np.array,
                  strikes: np.array,
                  t2e: np.array,
                  vol: np.array,
                  pcs: np.array
                  ) -> np.array:
    """
    Black normal model for option pricing of calls, puts, and straddles.
    :param dc: Discount factors to expiry
    :param fut: Futures prices
    :param strikes: Strike prices
    :param t2e: Time to expiry
    :param vol: Normal volatilities
    :param pcs: Option type indicator (-1 for put, 1 for call, 2 for straddle)
    :return: Option premiums
    """
    moneyness = fut - strikes
    vt = vol * np.sqrt(t2e)
    d = moneyness / vt

    # Calculate standard normal CDF and PDF
    cdf = ndtr(d)
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d ** 2)

    # Calculate phi_d_pcs based on pcs values
    phi_d_pcs = np.where(pcs == 1, cdf,  # Call option
                         np.where(pcs == -1, cdf - 1,  # Put option
                                  np.where(pcs == 2, 2 * cdf - 1,  # Straddle
                                           0)))  # Default case (pcs not -1, 1, or 2)

    # Define k(pcs) for scaling
    k = np.where(pcs == 2, 2, 1)

    # Calculate forward price
    fwd_price = moneyness * phi_d_pcs + k * vt * pdf

    # Calculate option premium
    premium = dc * fwd_price

    return premium


def _normal_vega(dc: np.array,
                 fut: np.array,
                 strikes: np.array,
                 t2e: np.array,
                 vol: np.array,
                 pcs: np.array
                 ) -> np.array:
    """
    Black normal model vega calculation for calls, puts, and straddles.
    Vega is adjusted based on the option type.
    :param dc: Discount factors to expiry
    :param fut: Futures prices
    :param strikes: Strike prices
    :param t2e: Time to expiry
    :param vol: Normal volatilities
    :param pcs: Option type indicator (-1 for put, 1 for call, 2 for straddle)
    :return: Option vegas
    """
    vt = vol * np.sqrt(t2e)
    moneyness = fut - strikes
    d = moneyness / vt

    # Calculate standard normal PDF
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d ** 2)

    # Define k(pcs) for scaling vega
    k = np.where(pcs == 2, 2, 1)

    # Calculate vega
    vega = dc * np.sqrt(t2e) * pdf * k

    return vega

def _implied_normal_vol(dc: np.array,
                        fut: np.array,
                        strikes: np.array,
                        t2e: np.array,
                        cp: np.array,
                        premium_market: np.array,
                        initial_vol: np.array = None
                        ) -> np.array:
    """
    Solve for implied normal volatility given market premiums.
    """
    # Ensure inputs are numpy arrays
    dc = np.asarray(dc, dtype=np.float64)
    fut = np.asarray(fut, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)
    t2e = np.asarray(t2e, dtype=np.float64)
    cp = np.asarray(cp, dtype=np.float64)
    premium_market = np.asarray(premium_market, dtype=np.float64)

    # Initial guess for volatility if not provided
    if initial_vol is None:
        initial_vol = np.full_like(premium_market, 0.01 * fut)

    # Define the objective function that returns residuals
    def objective_function(vol):
        # Compute model premiums
        premium_model = _normal_price(dc, fut, strikes, t2e, vol, cp)
        residuals = premium_model - premium_market
        return residuals

    n = len(fut)
    jac = np.zeros((n, n))
    # Define the Jacobian function
    def jacobian(vol):
        # Compute vegas
        vega = _normal_vega(dc, fut, strikes, t2e, vol, cp)
        # Since each residual depends only on its own volatility, the Jacobian is diagonal
        # We create a full Jacobian matrix with zeros and fill the diagonal with vega
        np.fill_diagonal(jac, vega)
        return jac

    # Bounds for the volatility (non-negative)
    vol_bounds = (1e-8, np.inf)

    # Solve using method 'trf' with bounds and Jacobian
    result = least_squares(objective_function, initial_vol, jac=jacobian, bounds=vol_bounds, method='trf')

    # Extract the implied volatilities from the result
    implied_vol = result.x

    return implied_vol

def _normal_sabr_implied_vol(
    F: float | np.ndarray,
    K: np.ndarray,
    T: float | np.ndarray,
    alpha: float | np.ndarray,
    rho: float | np.ndarray,
    nu: float | np.ndarray
) -> np.ndarray:
    """
    Computes the normal SABR model-implied volatilities using an accurate approximation,
    suitable for vectorization and JIT compilation.

    :param F: Forward prices (scalar or array)
    :param K: Strike prices (array)
    :param T: Times to expiry (scalar or array)
    :param alpha: SABR alpha parameters (scalar or array)
    :param rho: SABR rho parameters (scalar or array)
    :param nu: SABR nu parameters (scalar or array)
    :return: Normal SABR implied volatilities (array)
    """
    # Compute moneyness
    FK = F - K

    # Compute z parameter
    z = (nu / alpha) * FK

    # Compute numerator and denominator for chi(z)
    sqrt_term = np.sqrt(1 - 2 * rho * z + z ** 2)
    numerator = sqrt_term + z - rho
    denominator = 1 - rho

    # Ensure denominator is not zero
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)

    # Compute chi(z)
    chi = numerator / denominator

    # Compute ln(chi)
    ln_chi = np.log(chi)

    # Handle cases where ln_chi is zero (to avoid division by zero)
    ln_chi = np.where(np.abs(ln_chi) < 1e-12, 1e-12, ln_chi)

    # Compute z_over_ln_chi, handling the case when z and ln_chi are zero
    z_over_ln_chi = np.where(np.abs(ln_chi) > 1e-12, z / ln_chi, 1.0)

    # Compute the base implied volatility
    sigma_n = alpha * z_over_ln_chi

    # Compute correction term
    correction = 1 + ((nu ** 2) * T / 24) * (1 - 3 * rho ** 2)

    # Apply the correction term
    sigma_n = sigma_n * correction

    return sigma_n

def _sabr_fitter(
        F: float,
        T: float,
        K: np.ndarray,
        sigma_n_market: np.ndarray
) -> (float, float, float):
    """
    Calibrates the normal SABR parameters (alpha, rho, nu) to fit the observed
    market normal volatilities.

    :param F: Forward price (scalar)
    :param T: Time to expiry (scalar)
    :param K: Strike prices (array)
    :param sigma_n_market: Observed market normal volatilities (array)
    :return: Tuple of calibrated parameters (alpha, rho, nu)
    """
    # Ensure inputs are numpy arrays
    K = np.asarray(K, dtype=np.float64)
    sigma_n_market = np.asarray(sigma_n_market, dtype=np.float64)

    # Initial guesses
    alpha0 = np.mean(sigma_n_market)
    rho0 = 0.0
    nu0 = 0.1
    x0 = np.array([alpha0, rho0, nu0])

    # Bounds for parameters
    # alpha > 0, rho in (-1, 1), nu > 0
    bounds_lower = [1e-8, -0.9999, 1e-8]
    bounds_upper = [np.inf, 0.9999, np.inf]

    # Objective function for least squares
    def objective(params):
        alpha, rho, nu = params
        # Ensure parameters are within bounds
        if alpha <= 0 or nu <= 0 or not (-1 < rho < 1):
            return np.full_like(sigma_n_market, np.inf)
        # Compute model-implied volatilities
        sigma_n_model = _normal_sabr_implied_vol(F, K, T, alpha, rho, nu)
        # Residuals
        residuals = sigma_n_model - sigma_n_market
        return residuals

    # Use least squares optimization
    result = least_squares(
        objective,
        x0,
        bounds=(bounds_lower, bounds_upper),
        method='trf'
    )

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    # Extract calibrated parameters
    alpha_calibrated, rho_calibrated, nu_calibrated = result.x
    return alpha_calibrated, rho_calibrated, nu_calibrated


def debug_pricer():
    k = 95.000 + 0.125 * np.arange(7)
    f  = 95.6375 * np.ones(7)
    cp = np.ones(7)
    dc = 1 / (1 + 0.05 / 360) ** 56.1875 * np.ones(7)
    vol = np.array([1.8323, 0.7268, 0.5831, 0.5765, 0.5813, 0.5999, 0.6126])
    t = 56.1875 / 360 * np.ones(7)
    p = _normal_price(dc, f, k, t, vol, cp)
    vol2 = _implied_normal_vol(dc, f, k, t, cp, p)
    return p, vol2

def debug_sabr():
    pass

if __name__ == '__main__':
    debug_sabr()
    exit(0)