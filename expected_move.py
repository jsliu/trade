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

        # ---- Expected earnings move ----
        earnings_move_pct = sigmaE
        earnings_move_abs = spot * earnings_move_pct

        # ---- Full move to expiration ----
        total_move_pct = iv1 * math.sqrt(T1)
        total_move_abs = spot * total_move_pct

        # ---- Diffusion component (non-earnings) ----
        diffusion_pct = sigma12 * math.sqrt(T1 - dtE)
        diffusion_abs = spot * diffusion_pct
      
        return {
                "spot": spot,
                "expiry1": exp1,
                "expiry2": exp2,
                "iv1": iv1,
                "iv2": iv2,
                "T1": T1,
                "T2": T2,
                "diffusion_vol": sigma12,
                "earnings_vol": sigmaE,
                "earnings_move_pct": earnings_move_pct,
                "earnings_move_abs": earnings_move_abs,
                "earnings_range": (spot - earnings_move_abs,
                                spot + earnings_move_abs),
                "total_move_pct": total_move_pct,
                "total_range": (spot - total_move_abs,
                                spot + total_move_abs),
                "diffusion_move_pct": diffusion_pct,
                "diffusion_move_abs": diffusion_abs,
                "diffusion_range": (spot - diffusion_abs,
                                    spot + diffusion_abs),

        }
    
    except Exception as e:
        print(f"Error computing expected move for {symbol}: {e}")
        return None

async def main(symbols):
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
        for symbol in symbols:
            data = await compute_expected_move(ib, symbol)

            if not data:
                print(f"{symbol}: not available")
                continue

            print(f"\n{symbol}")
            print(f"Spot: {data['spot']:.2f}")
            
            print(f"\n--- Earnings Jump ---")
            print(f"Move: {data['earnings_move_pct']:.2%}")
            print(f"Range: [{data['earnings_range'][0]:.2f}, "
                f"{data['earnings_range'][1]:.2f}]")

            print(f"\n--- Post-Earnings Diffusion ---")
            print(f"Move: {data['diffusion_move_pct']:.2%}")
            print(f"Range: [{data['diffusion_range'][0]:.2f}, "
                f"{data['diffusion_range'][1]:.2f}]")

            print(f"\n--- Total Move to Expiration ---")
            print(f"Move: {data['total_move_pct']:.2%}")
            print(f"Range: [{data['total_range'][0]:.2f}, "
                f"{data['total_range'][1]:.2f}]")

    finally:
        if ib.isConnected():
            ib.disconnect()
# %%
if __name__ == "__main__":
    nest_asyncio.apply()
    symbols = ['HL', ]
    asyncio.run(main(symbols))
# %%
