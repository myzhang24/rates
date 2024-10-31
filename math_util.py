import numpy as np
from scipy.special import ndtr
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from date_util import parse_date
from fixing import FixingManager, past_discount
from swap import SOFRSwap


# Pricing functions
def _create_overlap_matrix(start_end_dates: np.ndarray,
                           knot_dates: np.ndarray) -> np.ndarray:
    """
    This function creates overlap matrix, how many calendar days is each future exposed to each meeting-to-meeting period
    :param start_end_dates:
    :param knot_dates:
    :return:
    """
    knot_ends = np.roll(knot_dates, -1) - 1
    knot_ends[-1] = 1_000_000
    starts = np.maximum(start_end_dates[:, 0].reshape(-1, 1), knot_dates.reshape(1, -1))
    ends = np.minimum(start_end_dates[:, 1].reshape(-1, 1), knot_ends.reshape(1, -1))
    return np.maximum(0, ends - starts + 1)

def _calculate_stub_fixing(ref_date: float,
                           start_end_dates: np.ndarray,
                           fixing: FixingManager,
                           multiplicative=False,
                           ) -> (float, np.ndarray):
    """
    This function calculates the stub fixings. Returns overnight rate and the stub fixings sum or accrual
    :param multiplicative:
    :param ref_date:
    :param start_end_dates:
    :return:
    """
    res = np.ones((start_end_dates.shape[0]), ) if multiplicative else np.zeros((start_end_dates.shape[0]), )
    for i in range(start_end_dates.shape[0]):
        start, end = start_end_dates[i, :]
        if start >= ref_date:
            pass
        fixings = 1e-2 * fixing.get_fixings_asof(parse_date(start), parse_date(ref_date-1))
        if multiplicative:
            res[i] = (1 + fixings / 360.0).prod()
        else:
            res[i] = fixings.sum()
    on = 1e-2 * fixing.get_fixings_asof(parse_date(ref_date-4), parse_date(ref_date-1)).iloc[-1]
    return on, res

def _price_1m_futures(future_knot_values: np.ndarray,
                      overlap_matrix: np.ndarray,
                      stub_fixings: np.ndarray,
                      n_days: np.ndarray) -> np.ndarray:
    """
    This function calculates 1m future prices. Low level numpy function.
    :param n_days:
    :param overlap_matrix:
    :param stub_fixings:
    :param future_knot_values:
    :return:
    """
    rate_sum = np.matmul(overlap_matrix, future_knot_values.reshape(-1, 1)).squeeze()
    rate_sum += stub_fixings
    return 1e2 * (1 - rate_sum / n_days)

def _price_3m_futures(future_knot_values: np.ndarray,
                      overlap_matrix: np.ndarray,
                      stub_fixings: np.ndarray,
                      n_days: np.ndarray) -> np.ndarray:
    """
    This function calculates 3m future prices. Low-level numpy function
    :param n_days:
    :param overlap_matrix:
    :param stub_fixings:
    :param future_knot_values:
    :return:
    """

    rate_prod = np.exp(np.matmul(overlap_matrix, np.log(1 + future_knot_values.reshape(-1, 1) / 360.0)).squeeze())
    rate_prod *= stub_fixings
    rate_avg = 360.0 * (rate_prod - 1) / n_days
    return 1e2 * (1 - rate_avg)

def _prepare_swap_batch_price(swaps: list[SOFRSwap]) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    This function returns the schedule block, a dcf block and a partition array
    :param swaps:
    :return:
    """
    schedules = [swap.get_float_leg_schedule(True).values for swap in swaps]
    partition = np.array([len(x) for x in schedules])
    schedule_block = np.concatenate(schedules, axis=0)
    schedules = schedule_block[:, :-1]
    dcfs = schedule_block[:, -1].squeeze()
    return schedules, dcfs, partition

def _sum_partitions(arr: np.ndarray, partitions: np.ndarray):
    """
    This function sums a lists of entries in arr according to partition given by part
    :param arr:
    :param partitions:
    :return:
    """
    indices = np.r_[0, np.cumsum(partitions)[:-1]]
    result = np.add.reduceat(arr, indices)
    return result

def _df(ref_date: float, dates: np.ndarray, knot_dates: np.ndarray, knot_values: np.ndarray):
    """
    Compute df using cubic spline interpolation on zero coupon rate knots. Low level numpy function.
    :param dates:
    :param knot_dates:
    :param knot_values:
    :param ref_date:
    :return:
    """
    cs = CubicSpline(knot_dates, knot_values, extrapolate=False)
    zero_rates = cs(dates)
    zero_rates[dates < knot_dates[0]] = knot_values[0]
    zero_rates[dates > knot_dates[-1]] = knot_values[-1]
    t_vect = (dates - ref_date) / 360.0
    return np.exp(-zero_rates * t_vect)

def _price_swap_rates(swap_knot_values: np.ndarray,
                      ref_date: float,
                      swap_knot_dates: np.ndarray,
                      schedules: np.ndarray,
                      dcfs: np.ndarray,
                      partitions: np.ndarray
                      ) -> np.ndarray:
    """
    This function evaluates par rates for swaps. Low level numpy function.
    :param dcfs:
    :param ref_date:
    :param partitions:
    :param swap_knot_values:
    :param swap_knot_dates:
    :param schedules:
    :return:
    """
    dfs = _df(ref_date, schedules, swap_knot_dates, swap_knot_values)
    if schedules[0, 0] < ref_date:
        dfs[0, 0] = past_discount(schedules[0, 0], ref_date)    # If first swap is stub
    numerators = (dfs[:, 0] / dfs[:, 1] - 1) * dfs[:, 2]  # fwd_i * df_i
    numerators = _sum_partitions(numerators, partitions)
    denominators = dcfs * dfs[:, 2]  # dcf_i * df_i
    denominators = _sum_partitions(denominators, partitions)
    rates = 1e2 * numerators / denominators
    return rates

# For discrete OIS compounding df
def _last_published_value(reference_dates: np.ndarray,
                          knot_dates: np.ndarray,
                          knot_values: np.ndarray) -> np.ndarray:
    """
    This function looks up reference_values for reference_dates according to knot_dates, knot_values
    :param reference_dates:
    :param knot_dates:
    :param knot_values:
    :return:
    """
    indices = np.searchsorted(knot_dates, reference_dates, side='right') - 1
    indices = np.clip(indices, 0, len(knot_values) - 1)
    return knot_values[indices]

def _ois_compound(reference_dates: np.ndarray,
                  reference_rates: np.ndarray):
    """
    This function computes the compounded OIS rate given the fixings
    :param reference_dates:
    :param reference_rates:
    :return:
    """
    num_days = np.diff(reference_dates)
    rates = reference_rates[:-1]
    annualized_rate = np.prod(1 + rates * num_days / 360) - 1
    return 360 * annualized_rate / num_days.sum()

def _normal_price(dc: float | np.ndarray,
                  t2e: float | np.ndarray,
                  fut: float | np.ndarray,
                  strikes: float | np.ndarray,
                  vol: float | np.ndarray,
                  pcs: float | np.ndarray
                  ) -> float | np.ndarray:
    """
    Black normal model for option pricing of calls, puts, and straddles.
    :param dc: Discount factors to expiry
    :param fut: Futures prices
    :param strikes: Strike prices
    :param t2e: Time to expiry
    :param vol: Normal volatility
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

    # Calculate forward price
    fwd_price = moneyness * phi_d_pcs + np.abs(pcs) * vt * pdf

    # Calculate option premium
    premium = dc * fwd_price

    return premium.squeeze()


def _normal_greek(dc: float | np.ndarray,
                  t2e: float | np.ndarray,
                  fut: float | np.ndarray,
                  strikes: np.ndarray,
                  vol: np.ndarray,
                  pcs: np.ndarray
                  ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Computes delta, gamma, vega, and theta for options under the Bachelier (normal) model.

    :param dc: Discount factors to expiry
    :param fut: Futures prices
    :param strikes: Strike prices
    :param t2e: Times to expiry
    :param vol: Normal volatility
    :param pcs: Option type indicator (-1 for put, 1 for call, 2 for straddle)
    :return: Tuple of arrays (delta, gamma, vega, theta)
    """
    moneyness = fut - strikes
    vt = vol * np.sqrt(t2e)
    d = moneyness / vt

    # Standard normal PDF and CDF
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d ** 2)
    cdf = ndtr(d)

    # phi_d_pcs
    phi_d_pcs = np.where(pcs == 1, cdf,              # Call option
                  np.where(pcs == -1, cdf - 1,       # Put option
                    np.where(pcs == 2, 2 * cdf - 1,  # Straddle
                             0)))  # Default case

    # Delta
    delta = dc * phi_d_pcs

    # Gamma
    gamma = dc * pdf / (vol * np.sqrt(t2e)) * np.abs(pcs)

    # Vega
    vega = dc * np.sqrt(t2e) * pdf * np.abs(pcs)

    # Theta
    theta = -dc * pdf * vol / (2 * np.sqrt(t2e)) * np.abs(pcs)

    return delta, gamma, vega, theta

def _implied_normal_vol(dc: np.array,
                        t2e: np.array,
                        fut: np.array,
                        strikes: np.array,
                        premium_market: np.array,
                        cp: np.array,
                        initial_vol: np.array = None
                        ) -> np.array:
    """
    Solve for implied normal volatility given market premiums.
    """
    # Initial guess for volatility if not provided
    if initial_vol is None:
        initial_vol = np.full_like(premium_market, 0.01 * fut)

    # Define the objective function that returns residuals
    def objective_function(vol):
        # Compute model premiums
        premium_model = _normal_price(dc, t2e, fut, strikes, vol, cp)
        residuals = premium_model - premium_market
        return residuals

    n = 1 if isinstance(strikes, float) else len(strikes)
    jac = np.zeros((n, n))
    # Define the Jacobian function
    def jacobian(vol):
        # Compute vegas
        _, _, vega, _ = _normal_greek(dc, t2e, fut, strikes, vol, cp)
        # Since each residual depends only on its own volatility, the Jacobian is diagonal
        # We create a full Jacobian matrix with zeros and fill the diagonal with vega
        np.fill_diagonal(jac, vega)
        return jac

    # Bounds for the volatility (non-negative)
    vol_bounds = (1e-8, np.inf)

    # Solve using method 'trf' with bounds and Jacobian
    result = least_squares(objective_function, initial_vol, jac=jacobian, bounds=vol_bounds, method='trf')

    # Extract the implied volatility from the result
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
    Computes the normal SABR model-implied volatility using an accurate approximation,
    suitable for vectorization and JIT compilation.

    :param F: Forward prices (scalar or array)
    :param K: Strike prices (array)
    :param T: Times to expiry (scalar or array)
    :param alpha: SABR alpha parameters (scalar or array)
    :param rho: SABR rho parameters (scalar or array)
    :param nu: SABR nu parameters (scalar or array)
    :return: Normal SABR implied volatility (array)
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
    market normal volatility.

    :param F: Forward price (scalar)
    :param T: Time to expiry (scalar)
    :param K: Strike prices (array)
    :param sigma_n_market: Observed market normal volatility (array)
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
        # Compute model-implied volatility
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
    k = 95.25
    f  = 95.635
    cp = 1
    dc = 0.9927
    vol = 0.5977
    t = 39.1/252
    p = _normal_price(dc, t, f, k, vol, cp)
    vol2 = _implied_normal_vol(dc, t, f, k, p, cp)
    return p, vol2

def debug_sabr():
    pass

if __name__ == '__main__':
    debug_pricer()
    exit(0)