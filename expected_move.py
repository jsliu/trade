# %%
# calculate expected share price move by implied earnings vol
import asyncio
import nest_asyncio
import datetime
import random
import math
import pandas as pd
from ib_async import IB, Stock, Option

# Request ATM options
async def get_atm_iv(ib, symbol, chain, expiry, spot):
    # strikes = sorted(chain.strikes)
    strikes = [s for s in chain.strikes if spot * 0.5 <= s <= spot * 1.5]
    if not strikes:
        print(f"No valid strikes for {symbol} {expiry}")
        return float('nan')

    atm = min(strikes, key=lambda x: abs(x - spot))

    opt = Option(symbol, expiry, atm, 'C', 'SMART')
    try:
        await ib.qualifyContractsAsync(opt)

        # reqMktData() is already awaitable, so don't need to add Async
        ticker = ib.reqMktData(opt)

        for _ in range(20):
            await asyncio.sleep(0.25)
            if ticker.modelGreeks:
                return ticker.modelGreeks.impliedVol
        # If we get here, IV not available
        print(f"No IV available for {symbol} {expiry} ATM={atm} — maybe market closed or illiquid")
        return float('nan')
    
    except Exception as e:
        print(f"Error fetching IV for {symbol} {expiry}: {e}")
        return float('nan')

# Compute expected move
async def compute_expected_move(ib, symbol, ann_factor=252):
    try:
        # ---- Stock ----
        stock = Stock(symbol, 'SMART', 'USD')
        await ib.qualifyContractsAsync(stock)


        # ---- Spot price (once) ----
        ticker_stock = ib.reqMktData(stock)

        for _ in range(10):
            await asyncio.sleep(0.2)
            spot = ticker_stock.last or ticker_stock.close
            if spot:
                break
        else:
            raise RuntimeError(f"Spot price not available for {symbol}")
            return float('nan') 
        
        # ---- Option chain ----
        chains = await ib.reqSecDefOptParamsAsync(
            stock.symbol, '', stock.secType, stock.conId
        )
        chain = chains[0]
        expires = sorted(chain.expirations)[:2]
        if len(expires) < 2:
            print(f"Not enough expiries for {symbol}")
            return float('nan')

        # First two expiries
        exp1, exp2 = expires

        # ---- ATM IVs ----
        iv1 = await get_atm_iv(ib, symbol, chain, exp1, spot)
        iv2 = await get_atm_iv(ib, symbol, chain, exp2, spot)

        # ---- Time to maturity ----
        d1 = datetime.datetime.strptime(exp1, '%Y%m%d')
        d2 = datetime.datetime.strptime(exp2, '%Y%m%d')
        today = datetime.datetime.now()

        T1 = (d1 - today).days / ann_factor
        T2 = (d2 - today).days / ann_factor

        # ---- Forward variance ----
        fwd_var = (T2 * iv2**2 - T1 * iv1**2) / (T2 - T1)
        sigma12 = math.sqrt(fwd_var)

        # ---- Earnings variance extraction ----
        dtE = 1 / ann_factor
        sigmaE_sq = T1 * (iv1**2 - sigma12**2) / dtE + sigma12**2
        sigmaE = math.sqrt(sigmaE_sq)

        # ---- Expected move ----
        expected_move = sigmaE / math.sqrt(ann_factor)
        return expected_move
    
    except Exception as e:
        print(f"Error computing expected move for {symbol}: {e}")
        return float('nan')

async def main(symbols, prices):
    # load symbols from csv
    # df = pd.read_csv(watchlist)
    # if "Financial Instrument" not in df.columns:
    #     raise ValueError("CSV must have a column named 'Financial Instrument'")
    # symbols = df["Financial Instrument"].to_list()

    ib = IB()
    try:
        await ib.connectAsync(
            '127.0.0.1',
            7496,  # TWS live
            clientId=random.randint(1000, 9999),
            timeout=5
        )

        # Get stocks from IB Watchlist
        # For ib_async, watchlists can be accessed via ib.reqManagedAccts or manually:
        # Here, just specify your symbols manually or implement reading your watchlist if you have the API
        # watchlist_symbols = ["AAPL", "MSFT", "GOOG"]  # replace with your watchlist symbols

        results = {}
        for symbol, price in zip(symbols, prices):
            move = await compute_expected_move(ib, symbol)
            if math.isnan(move):
                print(f"{symbol}: expected move not available")
            else:
                print(f"{symbol}: expected move = {move:.2%}: [{price * (1-move):.1f}, {price * (1+move):.1f}]")
            results[symbol] = move

        return results

    finally:
        if ib.isConnected():
            ib.disconnect()
# %%
if __name__ == "__main__":
    nest_asyncio.apply()
    symbols = ['HL', ]
    prices = [22.66, ]
    result = asyncio.run(main(symbols, prices))
    # result = await main('short.csv')
    # pd.DataFrame(result, index=['ExpectedMove']).T.to_csv(f'expected move/expected_move_{datetime.datetime.now()}.csv')
# %%
