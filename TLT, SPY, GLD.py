import pandas as pd, numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = [6,5]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
import seaborn as sns
sns.set()

tickers = ["TLT", "GLD", "SPY"]

años = 25
start = dt.date.today()-dt.timedelta(365*años)
end = dt.date.today()

data = yf.download(tickers, period="max")["Adj Close"]
data.dropna(inplace=True)
yields = data.pct_change()
risk_free = 0.00
#==============================================================================
#                  Combinaciones aleatorias
#==============================================================================
combinaciones = 2500
años = len(yields)/252
portfolio, returns, volatility, sharpe = [], [], [], []

for portfol in range(combinaciones):
    #ponderacion aleatoria
    weights = np.random.random_sample(len(tickers))
    weights = weights/weights.sum()
    
    portfolio.append(weights)
    
    yields2 = yields.copy()
    yields2["portfolio_daily_ret"] = (weights*yields).sum(axis=1)
    yields2["acum_ret"] = (1+yields2["portfolio_daily_ret"]).cumprod()-1
    retorno = ((1+yields2["acum_ret"][-1])**(1/años))-1
    returns.append(retorno)
    
    volatilidad = yields2["portfolio_daily_ret"].std() * np.sqrt(252)
    volatility.append(volatilidad)
    
    sharpe.append((retorno-risk_free)/volatilidad)

datos = pd.DataFrame({"retornos": returns,
                     "volatilidad": volatility,
                     "sharpe": sharpe,
                     "portfolio": portfolio})
#==============================================================================
#                  Combinaciones clasicas
#==============================================================================
yields3 = yields.copy()
yields3["100pct_SPY"] = (1+yields3["SPY"]).cumprod()-1
cagr_spy = ((1+yields3["100pct_SPY"][-1])**(1/años))-1
vol_spy = yields3["SPY"].std() * np.sqrt(252)
sharpe_spy = (cagr_spy-risk_free)/vol_spy

yields3["100pct_GLD"] = (1+yields3["GLD"]).cumprod()-1
cagr_gld = ((1+yields3["100pct_GLD"][-1])**(1/años))-1
vol_gld = yields3["GLD"].std() * np.sqrt(252)
sharpe_gld = (cagr_gld-risk_free)/vol_gld

yields3["100pct_TLT"] = (1+yields3["TLT"]).cumprod()-1
cagr_tlt = ((1+yields3["100pct_TLT"][-1])**(1/años))-1
vol_tlt = yields3["TLT"].std() * np.sqrt(252)
sharpe_tlt = (cagr_tlt-risk_free)/vol_tlt
#==============================================================================
opt_v = datos.iloc[datos["sharpe"].idxmax()]["volatilidad"]
opt_r = datos.iloc[datos["sharpe"].idxmax()]["retornos"]

min_risk_v = datos.iloc[datos["volatilidad"].idxmin()]["volatilidad"]
min_risk_r = datos.iloc[datos["volatilidad"].idxmin()]["retornos"]

max_ret_v = datos.iloc[datos["retornos"].idxmax()]["volatilidad"]
max_ret_r = datos.iloc[datos["retornos"].idxmax()]["retornos"]
#==============================================================================
#                  Grafico
#==============================================================================    
plt.title("RETORNO ENTRE NOVIEMBRE 2004 Y HOY", fontweight="bold",
          fontsize = 12)
plt.scatter(datos["volatilidad"], datos["retornos"], c= datos["sharpe"])
plt.scatter(vol_spy, cagr_spy, label="100% SPY", color="g",edgecolors="k")
plt.scatter(vol_gld, cagr_gld, label="100% GLD", color="y",edgecolors="k")
plt.scatter(vol_tlt, cagr_tlt, color="white", edgecolors="k")
plt.xlabel('VOLATILIDAD', fontsize = 12)
plt.ylabel('RETORNO', fontsize = 12)
plt.colorbar(label='SHARPE RATIO')
plt.scatter(min_risk_v, min_risk_r, color="salmon",edgecolors="k")
plt.legend(loc='upper right')
plt.annotate("Mínimo riesgo", weight ='bold', xy=(min_risk_v, min_risk_r), xytext=(.08, .105),
             arrowprops=dict(arrowstyle="->", color="k"))
plt.annotate("100% TLT", weight ='bold', xy=(vol_tlt, cagr_tlt), xytext=(.12, .07),
             arrowprops=dict(arrowstyle="->", color="k"))
plt.show()
