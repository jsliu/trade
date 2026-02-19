# %%
import asyncio
import nest_asyncio
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline
from ib_async import IB, Stock, Option

class VolLab:

    def __init__(self, ib: IB):
        self.ib = ib

    # -------------------------------------------------
    # 1️⃣ Spot Price
    # -------------------------------------------------
    async def get_spot(self, symbol: str):
        stock = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(stock)

        ticker = self.ib.reqMktData(stock)
        await asyncio.sleep(1)

        return ticker.last or ticker.marketPrice()


    # -------------------------------------------------
    # 2️⃣ Term Structure (ATM Call IVs)
    # -------------------------------------------------
    async def get_term_structure(self, symbol: str, n_expiries=24):

        stock = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(stock)

        chains = await self.ib.reqSecDefOptParamsAsync(
            stock.symbol, "", stock.secType, stock.conId
        )

        chain = chains[0]
        expirations = sorted(chain.expirations)[:n_expiries]
        strikes = sorted(chain.strikes)

        spot = await self.get_spot(symbol)

        results = []

        for expiry in expirations:

            # ATM strike
            strike = min(strikes, key=lambda x: abs(x - spot))

            opt = Option(symbol, expiry, strike, "C", "SMART")
            await self.ib.qualifyContractsAsync(opt)

            if not hasattr(opt, "conId") or not opt.conId:
                print(f"Skipping {symbol} {expiry} {strike}C — contract not available on IB")
                continue

            ticker = self.ib.reqMktData(opt)
            await asyncio.sleep(1)

            if ticker.modelGreeks:
                results.append({
                    "expiry": expiry,
                    "iv": ticker.modelGreeks.impliedVol
                })

        df = pd.DataFrame(results)

        if len(df) > 1:
            slope = df["iv"].iloc[-1] - df["iv"].iloc[0]
        else:
            slope = 0

        if slope < -0.02:
            structure = "INVERTED (Event Premium)"
        elif slope > 0.02:
            structure = "CONTANGO"
        else:
            structure = "FLAT"

        return df, slope, structure


    # -------------------------------------------------
    # 3️⃣ Skew (5% OTM Put vs Call)
    # -------------------------------------------------
    async def get_skew(self, symbol: str, expiry: str):

        stock = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(stock)

        chains = await self.ib.reqSecDefOptParamsAsync(
            stock.symbol, "", stock.secType, stock.conId
        )

        chain = chains[0]
        strikes = sorted(chain.strikes)

        spot = await self.get_spot(symbol)

        put_strike = min(strikes, key=lambda x: abs(x - spot * 0.95))
        call_strike = min(strikes, key=lambda x: abs(x - spot * 1.05))

        put = Option(symbol, expiry, put_strike, "P", "SMART")
        call = Option(symbol, expiry, call_strike, "C", "SMART")

        await self.ib.qualifyContractsAsync(put, call)

        put_ticker = self.ib.reqMktData(put)
        call_ticker = self.ib.reqMktData(call)

        await asyncio.sleep(1)

        put_iv = put_ticker.modelGreeks.impliedVol if put_ticker.modelGreeks else None
        call_iv = call_ticker.modelGreeks.impliedVol if call_ticker.modelGreeks else None

        if put_iv and call_iv:
            skew = put_iv - call_iv
        else:
            skew = None

        skew_type = None
        if skew is not None:
            skew_type = "PUT_HEAVY" if skew > 0 else "CALL_HEAVY"

        return {
            "put_iv": put_iv,
            "call_iv": call_iv,
            "skew": skew,
            "type": skew_type
        }


    # -------------------------------------------------
    # 4️⃣ Gamma Flip Level (Portfolio-Level Approx)
    # -------------------------------------------------
    def gamma_flip_level(self, legs: list, spot: float):

        prices = np.linspace(spot * 0.85, spot * 1.15, 100)
        gamma_profile = []

        for price in prices:
            total_gamma = 0
            for leg in legs:
                sign = 1 if leg["side"] == "BUY" else -1
                total_gamma += sign * leg["gamma"] * leg["contracts"]
            gamma_profile.append(total_gamma)

        zero_cross = None
        for i in range(1, len(gamma_profile)):
            if gamma_profile[i - 1] * gamma_profile[i] < 0:
                zero_cross = prices[i]
                break

        return zero_cross


    # -------------------------------------------------
    # 5️⃣ Plot Term Structure
    # -------------------------------------------------
    def plot_term_structure(self, df: pd.DataFrame, symbol: str):

        if df.empty:
            print("No IV data available.")
            return

            # x = numeric (months from now)
        df = df.sort_values("expiry")
        x = np.arange(len(df))
        y = df["iv"].values

        # Smooth spline
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=3)  # cubic spline
        y_smooth = spline(x_smooth)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_smooth, y_smooth, label="Smoothed IV", color="blue")
        plt.scatter(x, y, color="red", label="ATM IV (Monthly)")
        plt.grid(True, linestyle="--", alpha=0.3)

        # Optional: add mean line
        mean_iv = y.mean()
        plt.axhline(mean_iv, linestyle=":", color="green", alpha=0.7, label="Mean IV")

        plt.title(f"{symbol} 12-Month Smooth Term Structure")
        plt.xlabel("Month Index (0 = nearest)")
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------
    # 6️⃣ Full Vol Analysis
    # -------------------------------------------------
    async def full_vol_analysis(self, symbol: str):

        print(f"\nRunning Vol Lab for {symbol}\n")

        term_df, slope, structure = await self.get_term_structure(symbol)

        print("Term Structure:")
        print(term_df)
        print("Slope:", slope)
        print("Structure:", structure)

        skew = None

        if not term_df.empty:
            front_expiry = term_df["expiry"].iloc[0]
            skew = await self.get_skew(symbol, front_expiry)

            print("\nSkew:")
            print(skew)

        self.plot_term_structure(term_df, symbol)

        return {
            "term_structure": term_df,
            "slope": slope,
            "structure": structure,
            "skew": skew
        }

async def main():

    ib = IB()
    await ib.connectAsync("127.0.0.1", 
                          7496, 
                          clientId=random.randint(1000, 9999),
                          timeout=5)

    vol = VolLab(ib)

    result = await vol.full_vol_analysis("AAPL")

    ib.disconnect()

# %%
if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())

# %%
