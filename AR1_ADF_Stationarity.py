import numpy as np
from statsmodels.tsa.stattools import adfuller


class RandomWalk:
    """A position evolving step-by-step randomly.

    Public Attributes:
        - position: The net change of all steps from origin 0.0
        - theta: The probability of stepping up
        - step_up: How much to increment up by when stepping up
        - step_down: How much to increment down by when stepping down
        - path: A history of all positions, right-most being current position
    """

    constant: float
    path: list[tuple[float, float]]
    name: str

    def __init__(self, constant: float, name: str) -> None:
        """Create a new random walk."""
        self.constant = constant
        self.path = [(0, 0)]
        self.name = name

    def step(self) -> None:
        """Move *one* step forward."""
        step = self.constant + np.random.normal(0, 1)
        new_pos = self.path[-1][0] + 1.0, self.path[-1][1] + step
        self.path.append(new_pos)

    def run(self, n: int) -> None:
        """Move <n> consecutive steps forward."""
        for i in range(n):
            self.step()


def get_squared_diff(rw1: RandomWalk, rw2: RandomWalk, beta: float) -> float:
    """Return the sum of squared differences between points in time
    between <rw1> and <rw2> with a linear coefficient <beta>.

    Precondition: len(rw1.path) == len(rw2.path)
    """
    total = 0

    for i in range(len(rw1.path)):
        residual = rw1.path[i][1] - beta * rw2.path[i][1]
        total += residual**2

    return total


def get_hedge_ratio(rw1: RandomWalk, rw2: RandomWalk) -> float:
    """Return the coefficient beta minimizing squared
    residues between the two walks over time.
    """
    coefficients = np.linspace(0, 2, 1000)

    differences = []

    for coef in coefficients:
        differences.append(get_squared_diff(rw1, rw2, coef))

    min_index = differences.index(min(differences))

    return coefficients[min_index]


def get_adf(rw1: RandomWalk, rw2: RandomWalk, phi: float) -> tuple[bool, float]:
    """Return whether the spread between <rw1> and <rw2> is stationary
    given <phi> witnesses <rw1> and <rw2> are cointegrated.
    As well as the resulting adf p-value.

    Precondition: <rw1> and <rw2> are cointegrated.
    """

    spread = np.array(rw1.path).flatten() - (phi * np.array(rw2.path)).flatten()

    adf = adfuller(spread)[1]

    if adf <= 0.05:
        return True, adf
    return False, adf


def get_spread_squared_diff(spread: np.ndarray, beta: float) -> float:
    """Return the sum of squared differences of adjacent terms in <spread>
    given a residue minimizing coefficient <beta>.
    """
    total = 0

    for i in range(1, len(spread)):
        residual = spread[i] - beta * spread[i - 1]
        total += residual**2

    return total


def estimate_ar1_phi(rw1: RandomWalk, rw2: RandomWalk, beta: float) -> float:
    """Return the coefficient which minimizes squared distances
    between adjacent series points in the spread with noise."""
    spread = np.array(rw1.path).flatten() - (beta * np.array(rw2.path)).flatten()
    coefficients = np.linspace(0, 2, 5000)
    differences = []

    for coef in coefficients:
        differences.append(get_spread_squared_diff(spread, coef))

    min_index = differences.index(min(differences))

    return coefficients[min_index]


"""
Cointegration Gameplan

1. Start with two asset price series X_t and Y_t.

2. Check if each asset is I(1):
   - The price series itself is nonstationary / random-walk-like.
   - The first differences, returns or price changes, are stationary.

3. Estimate the hedge ratio beta using regression:
      X_t = alpha + beta * Y_t + residual_t

   beta is chosen to minimize total squared residuals.
   Intuition: remove as much shared linear movement as possible.

4. Construct the spread:
      S_t = X_t - beta * Y_t

5. Test the spread for stationarity using a unit root test such as ADF.
   - If the spread is stationary, the pair may be cointegrated.
   - If the spread is nonstationary, the pair is probably not useful for spread arb.

6. Model the spread with AR(1):
      S_t = phi * S_{t-1} + noise_t

   Estimate phi by minimizing total squared prediction error.

7. Interpret phi:
   - |phi| < 1 means mean reverting.
   - phi close to 1 means slow reversion / almost random walk.
   - |phi| > 1 means unstable or explosive.

8. Use the stationary spread for trading signals:
   - compute z-score of spread,
   - enter when spread is far from mean,
   - exit when spread reverts toward mean.

Core idea:
Cointegration tells us whether a stable equilibrium spread exists.
AR(1) tells us how quickly deviations from that equilibrium decay.
"""

if __name__ == "__main__":
    # 1 and 2, known RandomWalks
    asset1 = RandomWalk(0, "asset1")
    asset2 = RandomWalk(0, "asset2")

    # 3
    n = 10000
    asset1.run(n)
    asset2.run(n)
    beta = get_hedge_ratio(asset1, asset2)
    print(f"Beta: {beta}")

    # 4, 5
    result = get_adf(asset1, asset2, beta)
    print(f"ADF p-value: {result[1]}")
    if result[0]:
        print("     => The spread is stationary by ADF unit root test.")

        # 6
        phi_2 = estimate_ar1_phi(asset1, asset2, beta)
        print(f"Phi_2: {phi_2}")
        if phi_2 < 0.9:
            print(
                "     => The spread converges fast enough to its mean to continue with Stats-Arb"
            )
            pass
