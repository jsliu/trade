# %%
# calculate expected share price move by implied earnings vol
import asyncio
import nest_asyncio
import datetime
import random
import math
import pandas as pd
import numpy as np
from scipy.stats import norm
from ib_async import IB, Stock, Option, Order

# get greeks
async def get_leg_data(ib, symbol, leg):
    expiry = leg["expiry"]
    strike = leg["strike"]
    right = leg["right"]
    action = leg["action"]
    qty = leg.get("quantity", 1)

    sign = 1 if action.upper() == "BUY" else -1
    multiplier = sign * qty

    contract = Option(symbol, expiry, strike, right, "SMART")
    await ib.qualifyContractsAsync(contract)

    ticker = ib.reqMktData(contract)

    for _ in range(20):
        await asyncio.sleep(0.25)
        if ticker.modelGreeks and ticker.bid and ticker.ask:
            g = ticker.modelGreeks
            mid = (ticker.bid + ticker.ask) / 2
            return {
                "gamma": g.gamma * multiplier,
                "theta": g.theta * multiplier,
                "delta": g.delta * multiplier,
                "vega": g.vega * multiplier,
                "premium": mid * 100 * multiplier,
                "iv": g.impliedVol
            }

    print(f"No data for {symbol} {strike}")
    return None

# monte carlo expected value
def monte_carlo_ev(spot, iv, days, payoff_func, sims=3000):

    dt = days / 365
    results = []

    for _ in range(sims):
        shock = np.random.normal(0, iv*np.sqrt(dt))
        price = spot * np.exp(shock)
        results.append(payoff_func(price))

    arr = np.array(results)
    return arr.mean()


def find_break_even(spot, payoff_func):

    prices = np.linspace(spot*0.7, spot*1.3, 200)

    for i in range(1, len(prices)):
        if payoff_func(prices[i-1]) * payoff_func(prices[i]) < 0:
            return prices[i]

    return None

# margin
async def get_margin_requirement(ib, contract):

    order = Order(
        action="BUY",
        orderType="MKT",
        totalQuantity=1,
        transmit=False
    )

    w = await ib.whatIfOrderAsync(contract, order)
    return abs(w.initMarginChange or 1)



# compute metrics for combo
async def evaluate_strategy(ib, symbol, legs, spot, iv_crush_pct=0.2):
    net_gamma = 0
    net_theta = 0
    net_delta = 0
    net_vega = 0
    net_premium = 0
    iv_used = None
    days = 30

    for leg in legs:
        data = await get_leg_data(ib, symbol, leg)
        if data is None:
            return None
        net_gamma += data["gamma"]
        net_theta += data["theta"]
        net_delta += data["delta"]
        net_vega += data["vega"]
        net_premium += data["premium"]
        iv_used = data["iv"]

    # payoff function
    def payoff(price):

        pnl = 0

        for leg in legs:
            sign = 1 if leg["action"] == "BUY" else -1
            qty = leg["quantity"]

            if leg["right"] == "C":
                intrinsic = max(price - leg["strike"], 0)
            else:
                intrinsic = max(leg["strike"] - price, 0)

            pnl += intrinsic * 100 * qty * sign

        return pnl - net_premium
    
    # expected value
    ev = monte_carlo_ev(spot, iv_used, days, payoff)
    
    # Efficiency
    efficiency = net_gamma / abs(net_theta) if net_theta != 0 else float('nan')
    efficiency_theta = net_theta / abs(net_gamma) if net_gamma != 0 else float('nan')
    efficiency_gamma_premium = net_gamma / abs(net_premium) if net_premium != 0 else float('nan')
    efficiency_expected_value = ev / abs(net_premium) if net_premium != 0 else float('nan')

    breakeven = find_break_even(spot, payoff)
    breakeven_pct = ((breakeven - spot)/spot) if breakeven else None

    # Break-even &
    breakeven_pct = abs(net_premium) / (abs(net_delta) * spot * 100) if abs(net_delta) > 1e-6 else float('inf')

    # PnL for ±1%, ±2%
    def pnl_for_move(move_pct):
        dS = spot * move_pct
        return net_delta * dS * 100 + 0.5 * net_gamma * (dS ** 2) * 100

    pnl_1_up  = pnl_for_move(0.01)
    pnl_1_dn  = pnl_for_move(-0.01)
    pnl_2_up  = pnl_for_move(0.02)
    pnl_2_dn  = pnl_for_move(-0.02)

    # Earnings PnL with IV crush
    def pnl_earnings(move_pct, iv_crush=iv_crush_pct):
        dS = spot * move_pct
        return net_delta * dS * 100 + 0.5 * net_gamma * (dS ** 2) * 100 + net_vega * (-iv_crush)

    earnings_pnl = pnl_earnings(0.01)

    return {
        "net_delta": net_delta,
        "net_gamma": net_gamma,
        "net_theta": net_theta,
        "net_vega": net_vega,
        "premium": net_premium,
        "efficiency_gamma_theta": efficiency,
        "efficiency_theta_gamma": efficiency_theta,
        "efficiency_gamma_premium": efficiency_gamma_premium,
        "efficiency_expected_value": efficiency_expected_value,
        "breakeven_pct": breakeven_pct,
        "pnl_1_up": pnl_1_up,
        "pnl_1_dn": pnl_1_dn,
        "pnl_2_up": pnl_2_up,
        "pnl_2_dn": pnl_2_dn,
        "earnings_pnl": earnings_pnl
    }

async def detect_regime(ib, symbol):
    stock = Stock(symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(stock)

    # ---- Historical Data ----
    bars = await ib.reqHistoricalDataAsync(
        stock,
        endDateTime='',
        durationStr='30 D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True
    )

    df = pd.DataFrame([b.__dict__ for b in bars])
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    rv = df["return"].std() * np.sqrt(252)

    trend = (df["close"].iloc[-1] / df["close"].iloc[-20] - 1)

    # ---- ATM IV ----
    chains = await ib.reqSecDefOptParamsAsync(
        stock.symbol, '', stock.secType, stock.conId
    )

    chain = chains[0]
    expiry = sorted(chain.expirations)[0]
    strike = min(chain.strikes, key=lambda x: abs(x - df["close"].iloc[-1]))

    opt = Option(symbol, expiry, strike, "C", "SMART")
    await ib.qualifyContractsAsync(opt)

    ticker = ib.reqMktData(opt)

    for _ in range(20):
        await asyncio.sleep(0.25)
        if ticker.modelGreeks:
            iv = ticker.modelGreeks.impliedVol
            break

    iv_ratio = iv / rv if rv > 0 else 1

    # ---- Regime Classification ----
    if iv_ratio > 1.3:
        regime = "HIGH_IV"
    elif iv_ratio < 0.8:
        regime = "LOW_IV"
    elif abs(trend) > 0.05:
        regime = "TRENDING"
    else:
        regime = "NEUTRAL"

    return {
        "regime": regime,
        "iv": iv,
        "rv": rv,
        "iv_ratio": iv_ratio,
        "trend": trend
    }

def dynamic_rank(results, regime):
    if regime == "HIGH_IV":
        key_func = lambda x: x[1]["efficiency_theta_gamma"]  # prefer theta sellers
        reverse = True

    elif regime == "LOW_IV":
        key_func = lambda x: x[1]["efficiency_gamma_theta"]  # prefer long gamma
        reverse = True

    elif regime == "TRENDING":
        key_func = lambda x: x[1]["pnl_2_up"] + x[1]["pnl_2_dn"]
        reverse = True

    else:  # NEUTRAL
        key_func = lambda x: x[1]["efficiency_gamma_premium"]
        reverse = True

    return sorted(results.items(), key=key_func, reverse=reverse)

async def main(symbol, strategies):

    ib = IB()
    try:
        await ib.connectAsync(
            '127.0.0.1',
            7496,  # TWS live
            clientId=random.randint(1000, 9999),
            timeout=5
        )

        # Use first leg symbol to get spot
        stock = Stock(symbol, "SMART", "USD")
        await ib.qualifyContractsAsync(stock)
        ticker_stock = ib.reqMktData(stock)

        for _ in range(10):
            await asyncio.sleep(0.2)
            spot = ticker_stock.last or ticker_stock.close
            if spot:
                break

        results = {}
        for name, legs in strategies.items():
            metrics = await evaluate_strategy(ib, symbol, legs, spot)
            if metrics:
                results[name] = metrics

        # Ranking by Gamma/Theta efficiency
        regime_data = await detect_regime(ib, symbol)
        regime = regime_data["regime"]

        print("\nDetected Regime:", regime)
        print("IV:", round(regime_data["iv"], 3),
              "RV:", round(regime_data["rv"], 3),
              "IV/RV:", round(regime_data["iv_ratio"], 2),
              "Trend:", round(regime_data["trend"], 3))

        ranked = dynamic_rank(results, regime)

        # print 
        print("\n Strategy Comparison:\n")
        for rank, (name, r) in enumerate(ranked, 1):
            print(f"{rank}. {name}")
            print(f"   Premium: {r['premium']:.2f}")
            print(f"   Efficiency Gamma/Theta: {r['efficiency_gamma_theta']:.4f}")
            print(f"   Efficiency Theta/Gamma: {r['efficiency_theta_gamma']:.4f}")
            print(f"   Efficiency Gamma/Premium: {r['efficiency_gamma_premium']:.6f}")
            print(f"   Break-even %: {r['breakeven_pct']:.2%}")
            print(f"   PnL +1%: {r['pnl_1_up']:.2f} | -1%: {r['pnl_1_dn']:.2f}")
            print(f"   PnL +2%: {r['pnl_2_up']:.2f} | -2%: {r['pnl_2_dn']:.2f}")
            print(f"   Earnings PnL (IV crush 20%): {r['earnings_pnl']:.2f}")
            print("")

        return results

    finally:
        if ib.isConnected():
            ib.disconnect()
# %%
if __name__ == "__main__":
    nest_asyncio.apply()
    symbol = "HL"
    strategies = {
    "Call Spread": [
        {"expiry": "20260220", "strike": 22.5, "right": "C", "action": "BUY",  "quantity": 1},
        {"expiry": "20260220", "strike": 25,   "right": "C", "action": "SELL", "quantity": 1},
    ],

    "Ratio Spread": [
        {"expiry": "20260220", "strike": 22.5, "right": "C", "action": "BUY",  "quantity": 1},
        {"expiry": "20260220", "strike": 25,   "right": "C", "action": "SELL", "quantity": 2},
    ],

    "Long Straddle": [
        {"expiry": "20260220", "strike": 22.5, "right": "C", "action": "BUY", "quantity": 1},
        {"expiry": "20260220", "strike": 22.5, "right": "P", "action": "BUY", "quantity": 1},
    ]
    }
    result = asyncio.run(main(symbol, strategies))
    # result = await main('short.csv')
    # pd.DataFrame(result, index=['ExpectedMove']).T.to_csv(f'expected move/expected_move_{datetime.datetime.now()}.csv')
# %%
