# %%
# option_analytics.py
# This script fetches option data for a given symbol, calculates gamma exposure,
# and provides visualizations for net gamma exposure and call/put gamma exposure.   

import asyncio
import nest_asyncio
import random
from ib_async import IB, Stock, Option
import pandas as pd
import matplotlib.pyplot as plt


class OptionAnalytics:
    def __init__(self, symbol='CRCL', port=7496, client_id=1):
        self.symbol = symbol
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.underlying = None
        self.option_data = pd.DataFrame()

    async def connect(self):
        self.ib.connect('127.0.0.1', self.port, clientId=self.client_id)
        print("Connected to IB")

    async def disconnect(self):
        self.ib.disconnect()
        print("Disconnected from IB")

    async def fetch_option_chain_multiple_expiries(self, num_expiries=4, strike_range=(0, 1000)):
        self.underlying = Stock(self.symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(self.underlying)

        chains = self.ib.reqSecDefOptParams(
            self.underlying.symbol, '', self.underlying.secType, self.underlying.conId)
        chain = chains[0]

        expiries = sorted(chain.expirations)[:num_expiries]
        strikes = sorted([s for s in chain.strikes if strike_range[0] < s < strike_range[1]])

        all_data = []

        for expiry in expiries:
            print(f"Fetching data for expiry {expiry}...")

            # Create both calls and puts
            contracts = []
            for strike in strikes:
                contracts.append(Option(self.symbol, expiry, strike, 'C', 'SMART'))
                contracts.append(Option(self.symbol, expiry, strike, 'P', 'SMART'))

            self.ib.qualifyContracts(*contracts)
            tickers = [
                self.ib.reqMktData(contract, genericTickList='100', snapshot=False) 
                # self.ib.reqMktData(contract, snapshot=False) 
                for contract in contracts]
            # allow time for data to be fetched
            # await asyncio.sleep(1)
            for _ in range(100):
                if any(
                    self.get_option_oi(t) not in (None, 0)
                    for t in tickers
                    ):
                    break
                await asyncio.sleep(1)

            for contract, ticker in zip(contracts, tickers):
                greeks = getattr(ticker, 'modelGreeks', None)
                gamma = greeks.gamma if greeks else None
                # oi = getattr(ticker, 'optionOpenInterest', 0)
                oi = self.get_option_oi(ticker)

                if gamma is not None and oi is not None:
                    gex = gamma * oi * 100
                    all_data.append({
                        'Expiry': expiry,
                        'Strike': contract.strike,
                        'Right': contract.right,
                        'Gamma': gamma,
                        'Open Interest': oi,
                        'GEX': gex
                    })

        self.option_data = pd.DataFrame(all_data).sort_values(['Expiry', 'Strike', 'Right'])
        return self.option_data
    
    def get_option_oi(self, ticker):
        for dt in getattr(ticker, 'domTicks', []):
            # 27 = Call OI, 28 = Put OI
            if dt.tickType in (27, 28):
                return int(dt.size)
        return 0

    def net_gamma_exposure(self):
        if self.option_data.empty:
            print("No option data loaded.")
            return pd.DataFrame()

        # Sum across all expiries
        net = self.option_data.groupby('Strike')['GEX'].sum().reset_index()
        net.columns = ['Strike', 'Net GEX']
        return net

    def plot_net_gamma_exposure(self, title_suffix=''):
        net_df = self.net_gamma_exposure()
        if net_df.empty:
            print("No net GEX to plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.bar(net_df['Strike'], net_df['Net GEX'], color='darkorange')
        plt.title(f'Net Gamma Exposure by Strike for {self.symbol} {title_suffix}')
        plt.xlabel('Strike Price')
        plt.ylabel('Net Gamma Exposure')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_call_put_gamma_exposure(self, title_suffix=''):
        if self.option_data.empty:
            print("No option data loaded.")
            return

        # Separate calls and puts
        calls = self.option_data[self.option_data['Right'] == 'C']
        puts = self.option_data[self.option_data['Right'] == 'P']

        # Group and sum GEX by strike
        call_gex = calls.groupby('Strike')['GEX'].sum()
        put_gex = puts.groupby('Strike')['GEX'].sum()

        # Align both on same index
        all_strikes = sorted(set(call_gex.index).union(put_gex.index))
        call_gex = call_gex.reindex(all_strikes, fill_value=0)
        put_gex = put_gex.reindex(all_strikes, fill_value=0)

        bar_width = 0.4
        strikes = list(all_strikes)
        x = range(len(strikes))

        # Plot side-by-side bars
        plt.figure(figsize=(12, 6))
        plt.bar([i - bar_width/2 for i in x], call_gex.values, width=bar_width, label='Calls', color='steelblue')
        plt.bar([i + bar_width/2 for i in x], put_gex.values, width=bar_width, label='Puts', color='indianred')

        plt.title(f'Gamma Exposure by Strike for {self.symbol} {title_suffix}')
        plt.xlabel('Strike Price')
        plt.ylabel('Gamma Exposure')
        plt.xticks(ticks=x, labels=strikes, rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

async def main():
    client_id = random.randint(2, 10)  # Random client ID for the IB connection
    analytics = OptionAnalytics(symbol='AAPL', client_id=client_id)
    try:
        await analytics.connect()

        # Fetch both call and put chains
        await analytics.fetch_option_chain_multiple_expiries(num_expiries=4, strike_range=(180, 200))

    finally:
        await analytics.disconnect()    
    
    # Plot net GEX by strike
    analytics.plot_net_gamma_exposure()
    analytics.plot_call_put_gamma_exposure()


# %%
if __name__ == "__main__":
    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()
    # Run the main function 
    asyncio.run(main())
# %%
