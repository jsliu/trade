
# Detects front/back volatility relationship (earnings compression / expansion)
async def get_term_structure(ib, symbol):
    stock = Stock(symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(stock)

    chains = await ib.reqSecDefOptParamsAsync(
        stock.symbol, '', stock.secType, stock.conId
    )

    chain = chains[0]
    expirations = sorted(chain.expirations)[:3]  # first 3 expiries
    strikes = sorted(chain.strikes)

    term_ivs = []

    for expiry in expirations:
        # ATM strike
        ticker_stock = ib.reqMktData(stock)
        await asyncio.sleep(1)
        spot = ticker_stock.last or ticker_stock.marketPrice()

        strike = min(strikes, key=lambda x: abs(x - spot))

        opt = Option(symbol, expiry, strike, "C", "SMART")
        await ib.qualifyContractsAsync(opt)

        ticker = ib.reqMktData(opt)
        await asyncio.sleep(1)

        if ticker.modelGreeks:
            term_ivs.append((expiry, ticker.modelGreeks.impliedVol))

    slope = term_ivs[-1][1] - term_ivs[0][1] if len(term_ivs) > 1 else 0

    if slope < -0.02:
        structure = "INVERTED (EARNINGS)"
    elif slope > 0.02:
        structure = "NORMAL CONTANGO"
    else:
        structure = "FLAT"

    return {
        "term_structure": structure,
        "slope": slope,
        "ivs": term_ivs
    }

# Measure put-call IV difference
async def detect_skew(ib, symbol, expiry):
    stock = Stock(symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(stock)

    chains = await ib.reqSecDefOptParamsAsync(
        stock.symbol, '', stock.secType, stock.conId
    )

    chain = chains[0]
    strikes = sorted(chain.strikes)

    ticker_stock = ib.reqMktData(stock)
    await asyncio.sleep(1)
    spot = ticker_stock.last or ticker_stock.marketPrice()

    atm = min(strikes, key=lambda x: abs(x - spot))
    otm_put = min(strikes, key=lambda x: abs(x - spot*0.95))
    otm_call = min(strikes, key=lambda x: abs(x - spot*1.05))

    contracts = [
        Option(symbol, expiry, otm_put, "P", "SMART"),
        Option(symbol, expiry, otm_call, "C", "SMART")
    ]

    await ib.qualifyContractsAsync(*contracts)

    skew_vals = []
    for c in contracts:
        t = ib.reqMktData(c)
        await asyncio.sleep(1)
        if t.modelGreeks:
            skew_vals.append(t.modelGreeks.impliedVol)

    skew = skew_vals[0] - skew_vals[1]

    return {
        "skew": skew,
        "type": "PUT_HEAVY" if skew > 0 else "CALL_HEAVY"
    }

# Find price where net gamma = 0
def gamma_flip_level(legs, spot):
    # approximate: find price where gamma exposure changes sign
    prices = np.linspace(spot*0.9, spot*1.1, 50)
    gamma_profile = []

    for price in prices:
        total_gamma = 0
        for leg in legs:
            sign = 1 if leg["side"] == "BUY" else -1
            total_gamma += sign * leg["gamma"] * leg["contracts"]
        gamma_profile.append(total_gamma)

    zero_cross = None
    for i in range(1, len(gamma_profile)):
        if gamma_profile[i-1] * gamma_profile[i] < 0:
            zero_cross = prices[i]
            break

    return zero_cross

# Real distribution-based PnL
def monte_carlo_ev(spot, iv, days, payoff_func, simulations=5000):
    dt = days / 365
    results = []

    for _ in range(simulations):
        shock = np.random.normal(0, iv*np.sqrt(dt))
        price = spot * np.exp(shock)
        pnl = payoff_func(price)
        results.append(pnl)

    return {
        "expected_value": np.mean(results),
        "std_dev": np.std(results),
        "prob_profit": np.mean(np.array(results) > 0)
    }
