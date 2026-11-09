"""
Mesure du risque de prix du pétrole pour une entreprise acheteuse de carburant.

Situation : une compagnie de transport doit acheter l'équivalent de 50 000 barils de Brent dans 3 mois.
Questions auxquelles ce script répond :
- Comment le prix du Brent se comporte-t-il (tendance, volatilité, gros chocs) ?
- Combien l'entreprise peut-elle perdre si le prix monte ? (VaR, Expected Shortfall)
- Le résultat dépend-il de la méthode ? (historique, Monte Carlo simple, Monte Carlo GARCH)
- Que se passe-t-il lors de crises comme 2022 ? (stress tests)
- La VaR quotidienne du modèle est-elle fiable ? (backtesting, test de Kupiec)
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

warnings.filterwarnings("ignore")

TICKER = "BZ=F"            # Brent, contrat à terme le plus proche (Yahoo Finance)
START_DATE = "2010-01-01"
BARRELS = 50_000           # quantité à acheter
HORIZON = 63               # environ 3 mois de jours ouvrés
N_SIMULATIONS = 10_000
LEVELS = [0.95, 0.99]
DATA_FILE = Path("data/brent.csv")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# ============================================================================
# 1. Données
# ============================================================================
if DATA_FILE.exists():
    prices = pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")["Close"]
else:
    import yfinance as yf
    raw = yf.download(TICKER, start=START_DATE, progress=False, auto_adjust=False)
    if raw.empty:
        raise SystemExit("Yahoo Finance n'a renvoyé aucune donnée : réessaie plus tard.")
    prices = raw["Close"].squeeze().rename("Close")
    DATA_FILE.parent.mkdir(exist_ok=True)
    prices.to_csv(DATA_FILE, index_label="Date")

prices = prices[~prices.index.duplicated()].dropna()
prices = prices[prices > 0].sort_index()

# Contrôle qualité : grosse variation immédiatement annulée le lendemain = probable erreur
log_ret = np.log(prices).diff()
typical = log_ret.abs().rolling(60, min_periods=20).median().shift(1)
suspects = prices.index[(log_ret.abs() > 10 * typical)
                        & (np.sign(log_ret) != np.sign(log_ret.shift(-1)))
                        & (log_ret.shift(-1).abs() > 0.8 * log_ret.abs())]
prices[suspects] = np.nan
prices = prices.interpolate(method="time")
log_ret = np.log(prices).diff().dropna()

spot = prices.iloc[-1]
purchase_today = BARRELS * spot
print(f"Période : {prices.index[0]:%d/%m/%Y} -> {prices.index[-1]:%d/%m/%Y} ({len(prices)} jours)")
print(f"Jours suspects corrigés : {len(suspects)}")
for day in suspects:
    print(f"  {day:%d/%m/%Y}")
print(f"Prix du Brent aujourd'hui : {spot:.2f} $ | Achat de {BARRELS:,} barils au prix du jour : {purchase_today:,.0f} $\n")

print(f"Volatilité annualisée : {log_ret.std() * np.sqrt(252):.1%}")
print(f"Pire jour : {log_ret.min():+.1%} ({log_ret.idxmin():%d/%m/%Y}) | "
      f"Meilleur jour : {log_ret.max():+.1%} ({log_ret.idxmax():%d/%m/%Y})")
print(f"Kurtosis (excès) : {log_ret.kurtosis():.1f}  (0 pour une loi normale)\n")

# ============================================================================
# 2. Volatilité GARCH(1,1) avec loi de Student et ciblage de la variance
# ============================================================================
returns_pct = 100 * log_ret
model = arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
params = model.fit(disp="off").params.copy()
persistence = params["alpha[1]"] + params["beta[1]"]
# Si alpha + beta est presque égal à 1, les chocs ne se calment jamais et les simulations
# explosent : on plafonne la persistance à 0,99 en gardant le rapport entre alpha et beta.
MAX_PERSISTENCE = 0.99
if persistence > MAX_PERSISTENCE:
    print(f"Persistance estimée = {persistence:.3f}, plafonnée à {MAX_PERSISTENCE} pour la stabilité des simulations.")
    params["alpha[1]"] *= MAX_PERSISTENCE / persistence
    params["beta[1]"] *= MAX_PERSISTENCE / persistence
    persistence = MAX_PERSISTENCE
params["omega"] = returns_pct.var() * (1 - persistence)
garch = model.fix(params)

current_vol = garch.conditional_volatility.iloc[-1] * np.sqrt(252)
long_run_vol = returns_pct.std() * np.sqrt(252)
print(f"GARCH(1,1) Student : alpha = {params['alpha[1]']:.3f}, beta = {params['beta[1]']:.3f}, "
      f"persistance = {persistence:.3f}, nu = {params['nu']:.1f}")
print(f"Volatilité annualisée aujourd'hui : {current_vol:.1f} % | niveau de long terme : {long_run_vol:.1f} %")
print(f"-> Le marché est actuellement {'plus agité' if current_vol > long_run_vol else 'plus calme'} que d'habitude.\n")

# ============================================================================
# 3. Scénarios sur 3 mois et VaR / Expected Shortfall
# ============================================================================


def losses_from_returns(three_month_log_returns: np.ndarray) -> np.ndarray:
    """Surcoût en $ par rapport au prix du jour, pour chaque scénario de variation sur 3 mois."""
    return BARRELS * spot * (np.exp(three_month_log_returns) - 1)


def var_es(losses: np.ndarray, level: float) -> tuple[float, float]:
    var = np.quantile(losses, level)
    return var, losses[losses >= var].mean()


log_prices = np.log(prices)
three_month_hist = (log_prices.shift(-HORIZON) - log_prices).dropna()

mu, sigma = log_ret.mean(), log_ret.std()
gbm = rng.normal(mu - 0.5 * sigma ** 2, sigma, size=(N_SIMULATIONS, HORIZON)).sum(axis=1)

# Simulation GARCH écrite à la main, pour maîtriser entièrement le hasard (résultats reproductibles).
# Chaque jour : rendement = mu + volatilité du jour × choc de Student,
# puis variance du lendemain = omega + alpha × (choc)² + beta × variance du jour.
mu_g, omega, alpha, beta, nu = (params["mu"], params["omega"], params["alpha[1]"],
                                params["beta[1]"], params["nu"])
last_shock = returns_pct.iloc[-1] - mu_g
variance = np.full(N_SIMULATIONS, omega + alpha * last_shock ** 2
                   + beta * garch.conditional_volatility.iloc[-1] ** 2)
# Garde-fou : la volatilité simulée ne dépasse jamais le maximum observé dans l'historique (Covid, 2020).
max_variance = garch.conditional_volatility.max() ** 2
variance = np.minimum(variance, max_variance)
garch_3m = np.zeros(N_SIMULATIONS)
for _ in range(HORIZON):
    shocks = rng.standard_t(nu, N_SIMULATIONS) * np.sqrt((nu - 2) / nu)  # variance ramenée à 1
    daily = np.sqrt(variance) * shocks
    garch_3m += (mu_g + daily) / 100
    variance = np.minimum(omega + alpha * daily ** 2 + beta * variance, max_variance)

scenarios = {
    "Historique": losses_from_returns(three_month_hist.values),
    "Monte Carlo simple": losses_from_returns(gbm),
    "Monte Carlo GARCH": losses_from_returns(garch_3m),
}
rows = []
for method, losses in scenarios.items():
    row = {"Méthode": method}
    for level in LEVELS:
        var, es = var_es(losses, level)
        row[f"VaR {level:.0%}"], row[f"ES {level:.0%}"] = var, es
    rows.append(row)
results = pd.DataFrame(rows).set_index("Méthode")
print(f"Surcoût possible sur 3 mois en $ (achat au prix du jour : {purchase_today:,.0f} $)")
print(results.to_string(float_format=lambda x: f"{x:,.0f}"))
print("Lecture : VaR 95 % = surcoût dépassé dans seulement 5 % des scénarios ;"
      " ES 95 % = surcoût moyen dans ces 5 % de pires scénarios.\n")

# ============================================================================
# 4. Stress tests
# ============================================================================
# Le rebond d'avril 2020 (Brent à 19 $, puis +100 % en 3 mois) est un rattrapage après un krach :
# appliqué au prix d'aujourd'hui, il donnerait un prix irréaliste. On l'exclut du « pire épisode ».
rebound_2020 = (three_month_hist.index >= "2020-03-01") & (three_month_hist.index <= "2020-06-30")
worst_start = three_month_hist[~rebound_2020].idxmax()
ukraine_start = prices.index[prices.index.get_indexer([pd.Timestamp("2021-12-01")], method="bfill")[0]]
ukraine_shock = np.log(prices[ukraine_start:].iloc[:HORIZON + 1].max() / prices[ukraine_start])
stress = {
    "Brent +20 %": np.log(1.20),
    "Brent +40 % (choc géopolitique)": np.log(1.40),
    "Guerre en Ukraine (hausse max. en 3 mois dès déc. 2021)": ukraine_shock,
    f"Pire hausse sur 3 mois hors rebond 2020 (départ le {worst_start:%d/%m/%Y})": three_month_hist[~rebound_2020].max(),
}
print("Stress tests")
stress_rows = []
for name, shock in stress.items():
    loss = losses_from_returns(np.array([shock]))[0]
    stress_rows.append({"Scénario": name, "Prix ($)": spot * np.exp(shock), "Surcoût ($)": loss})
    print(f"  {name:<68} prix {spot * np.exp(shock):>7.2f} $ -> surcoût {loss:>12,.0f} $")

# ============================================================================
# 5. Backtesting de la VaR quotidienne à 95 % sur les 250 derniers jours (test de Kupiec)
# ============================================================================
nu = params["nu"]
q95 = stats.t.ppf(0.95, nu) * np.sqrt((nu - 2) / nu)
daily_var = (params["mu"] + q95 * garch.conditional_volatility) / 100
test = log_ret.index[-250:]
exceed = log_ret[test] > daily_var[test]
n, x = len(exceed), int(exceed.sum())
p_hat = x / n
lr = -2 * ((n - x) * np.log(0.95) + x * np.log(0.05)
           - ((n - x) * np.log(1 - p_hat) + (x * np.log(p_hat) if x > 0 else 0)))
p_value = 1 - stats.chi2.cdf(lr, 1)
print(f"\nBacktesting : {x} dépassements sur {n} jours, soit {p_hat:.1%} (attendu : 5 %)")
print(f"Test de Kupiec : p-value = {p_value:.2f} -> "
      f"{'modèle acceptable' if p_value > 0.05 else 'modèle rejeté'} (seuil 0,05)")

# ============================================================================
# 6. Sauvegardes et graphiques
# ============================================================================
results.to_csv("data/var_resultats.csv", index_label="Methode")
pd.DataFrame(stress_rows).to_csv("data/stress_tests.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(11, 8))
prices.plot(ax=axes[0], color="#111827", title="Prix du Brent ($ par baril)")
(garch.conditional_volatility * np.sqrt(252)).plot(ax=axes[1], color="#dc2626", label="Volatilité GARCH",
                                                  title="Volatilité annualisée GARCH (%)")
axes[1].axhline(long_run_vol, color="black", ls="--", lw=1, label="Niveau de long terme")
axes[1].legend()
for ax in axes:
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(FIGURES / "01_prix_volatilite.png", dpi=150)

fig, ax = plt.subplots(figsize=(11, 5))
losses_k = scenarios["Monte Carlo GARCH"] / 1000
lo, hi = np.quantile(losses_k, [0.001, 0.999])
ax.hist(np.clip(losses_k, lo, hi), bins=120, color="#93c5fd")
var95, es95 = var_es(losses_k, 0.95)
ax.axvline(0, color="black", lw=1)
ax.axvline(var95, color="#ea580c", lw=2, label=f"VaR 95 % = {var95:,.0f} k$")
ax.axvline(es95, color="#dc2626", lw=2, ls="--", label=f"ES 95 % = {es95:,.0f} k$")
ax.set_title(f"Surcoût possible dans 3 mois pour l'achat de {BARRELS:,} barils (Monte Carlo GARCH)")
ax.set_xlabel("Surcoût en milliers de $ (négatif = économie)")
ax.set_ylabel("Nombre de scénarios")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "02_var.png", dpi=150)
plt.show()
