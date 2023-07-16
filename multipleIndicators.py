from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override() 
import pandas as pd
import json
import openai
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt

# Following are the functions
def getData(ticker):
 data = pdr.get_data_yahoo(ticker, start=start_date, end= date.today())
 dataname= ticker
 SaveData(data, dataname)
# Create a data folder in your current dir.
def SaveData(df, filename):
 df.to_csv('./data/'+filename+'.csv')
##################################################################################################

class UserStrategy(bt.Strategy):
    params = (
        ('buy_criteria', {
            "price": {
                "conditions": [
                    {
                        "value": 900,
                        "operation": ">"
                    }
                ]
            },
            "indicators": {
                "conditions": [
                    {
                        "name": "rsi",
                        "value": 60,
                        "operation": ">"
                    },
                    {
                        "name": "stochastic",
                        "value": 80,
                        "operation": ">"
                    },
                    {
                        "name": "ema",
                        "value": 5,
                        "operation": ">"
                    }
                ]
            }
        }),
        ('sell_criteria', {
            "price": {
                "conditions": [
                    {
                        "value": 900,
                        "operation": "<"
                    }
                ]
            },
            "indicators": {
                "conditions": [
                    {
                        "name": "rsi",
                        "value": 60,
                        "operation": "<"
                    },
                    {
                        "name": "stochastic",
                        "value": 80,
                        "operation": "<"
                    },
                    {
                        "name": "ema",
                        "value": 5,
                        "operation": "<"
                    }
                ]
            }
        }),
        ('risk_per_trade', 500),  # Risk per trade in the account currency
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        # self.stochastic = bt.indicators.Stochastic(self.data.close, period=14)
        self.buy_indicators = []
        self.sell_indicators = []

        # Add price condition
        buy_price_conditions = self.params.buy_criteria.get('price', {}).get('conditions', [])
        for condition in buy_price_conditions:
            self.buy_indicators.append(('price', condition['value'], condition['operation']))
        # Add indicator conditions
        buy_indicator_conditions = self.params.buy_criteria.get('indicators', {}).get('conditions', [])
        for condition in buy_indicator_conditions:
            self.buy_indicators.append((condition['name'], condition['value'], condition['operation']))
            # Initiliaze EMAs
            if condition['name'] == "ema":
                ema_name = "ema-" + str(condition['value'])
                setattr(self.data, ema_name, bt.indicators.ExponentialMovingAverage(self.data.close, period=condition['value']))
        
        
        # Sell price condition
        sell_price_conditions = self.params.sell_criteria.get('price', {}).get('conditions', [])
        print("sell_price_conditions",sell_price_conditions)
        for condition in sell_price_conditions:
            self.sell_indicators.append(('price', condition['value'], condition['operation']))
        # Sell indicator conditions
        sell_indicator_conditions = self.params.sell_criteria.get('indicators', {}).get('conditions', [])
        print("sell_price_conditions",sell_indicator_conditions)
        for condition in sell_indicator_conditions:
            self.sell_indicators.append((condition['name'], condition['value'], condition['operation']))
            # Initiliaze EMAs
            if condition['name'] == "ema":
                ema_name = "ema-" + str(condition['value'])
                setattr(self.data, ema_name, bt.indicators.ExponentialMovingAverage(self.data.close, period=condition['value']))

        print("buy_indicators::", self.buy_indicators)
        print("sell_indicators::", self.sell_indicators)

    def next(self):
        if self.position:  # If already in a position, don't execute additional trades
            for name, value, operation in self.sell_indicators:
                if not self.isConditionSatisfied(name,value,operation):
                    return
            self.sell(size=self.position.size)
            return

        # Check indicator conditions
        for name, value, operation in self.buy_indicators:
            if not self.isConditionSatisfied(name,value,operation):
                return

        # Buy if all conditions are met
        self.buy(size=self.params.risk_per_trade / self.data.close)


##############################################################################################
    def isConditionSatisfied(self,name,value, operation):
        if name == 'price':
            if operation == '>' and self.data.close[0] <= value:
                return False
            if operation == '<' and self.data.close[0] >= value:
                return False            
        elif name == 'rsi':
            rsi = self.rsi[0]
            if operation == '>' and rsi <= value:
                return False
            if operation == '<' and rsi >= value:
                return False
        # elif name == 'stochastic':
        #     stochastic = self.stochastic[0]
        #     if operation == '>' and stochastic >= value:
        #         return False
        #     if operation == '<' and stochastic >= value:
        #         return False
        elif name == 'ema':
            ema = getattr(self.data, "ema-" + str(value))[0]
            if operation == '>' and self.data.close[0] <= ema:
                return False
            if operation == '<' and self.data.close[0] >= ema:
                return False
        return True

    def ema(self, data, period):
        return bt.indicators.ExponentialMovingAverage(data, period=period)
##############################################################################################

def run_backtest(data, capital, buyCriteria,sellCriteria):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(UserStrategy, risk_per_trade=500,buy_criteria= buyCriteria, sell_criteria=sellCriteria)

    # Convert the data to a format compatible with backtrader
    data = bt.feeds.PandasData(dataname=data)

    cerebro.adddata(data)
    cerebro.broker.setcash(capital)

    # Set the commission and slippage parameters (if needed)
    cerebro.broker.setcommission(commission=0.001)  # Example commission of 0.1%
    cerebro.broker.set_slippage_fixed(0.01)  # Example fixed slippage of 1 cent per share

    # Run the backtest
    cerebro.run()

    # Plot the chart with buy/sell signals and EMAs
    cerebro.plot(style='candlestick')

# ###############################################################################################
# Load OpenAI API credentials
with open('openai_credentials.json') as f:
    credentials = json.load(f)
    openai.api_key = credentials['api_key']

# Load system prompts from JSON file
with open('systemPrompt.json') as f:
    systemPrompt = json.load(f)['systemPrompt']


# ###############################################################################################
# ###############################################################################################
# ###############################################################################################
# ###############################################################################################

# response = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#        {
#         "role": "system",
#         "content": systemPrompt
#         },
#         {
#         "role": "user",
#         "content": "buy reliance if price goes above 2000 , rsi must be above 60 and stochastic above 55 and stock should be above 50 ema and sell position  below 13 ema"
#         }
#   ],
#   temperature=1,
#   max_tokens=256,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )
# json_data = response["choices"][0]["message"]["content"]
# data = json.loads(json_data)

# # We can get data by our choice by giving days bracket
start_date= "2017-01-01"

data = {'stockName': 'RELIANCE.NS', 'buyCriteria': {'price': {'conditions': [{'value': 2000, 'operation': '>'}]}, 'indicators': {'conditions': [{'name': 'rsi', 'value': 60, 'operation': '>'}, {'name': 'stochastic', 'value': 55, 'operation': '>'}, {'name': 'ema', 'value': 50, 'operation': '>'}]}}, 'sellCriteria': {'indicators': {'conditions': [{'name': 'ema', 'value': 13, 'operation': '<'}]}}}
print(data)
getData(data["stockName"])
# Read the CSV file into a pandas DataFrame
df = pd.read_csv('./data/'+data["stockName"]+'.csv')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Set initial capital and run the backtest
capital = 50000
run_backtest(df, capital,data['buyCriteria'],data['sellCriteria'])