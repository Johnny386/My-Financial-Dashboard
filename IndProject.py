# Libraries
import numpy as np                        # Array, Calculation
import pandas as pd                       # DataFrame
import matplotlib.pyplot as plt           # Visualization
import plotly.express as px               # Visualization
import plotly.graph_objects as go         # Visualization
from datetime import datetime, timedelta  # Date-time
import yfinance as yf
import streamlit as st
from newsapi import NewsApiClient


#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "majorHoldersBreakdown,"
                         "indexTrend,"
                         "defaultKeyStatistics,"
                         "majorHoldersBreakdown,"
                         "insiderHolders")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret


# Create a function that generates the list of S&P 500 stocks
def extractStocks():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500 = pd.read_html(url)[0]  # Read the first table on the page
    tickers = sp500['Symbol'].tolist()
    return tickers

# Create a function that will extract the info of a chosen stock
def getStockData(stockname, startDate, endDate):
    stockData = yf.Ticker(stockname).history(start=startDate, end=endDate)
    return stockData

# Get the company information
@st.cache_data
def GetCompanyInfo(ticker):
    """
    This function get the company information from Yahoo Finance.
    """        
    return YFinance(ticker).info

# Create function for the Monte Carlo Simulation
def MonteCarloSimulation(stock_database, simulations, time_horizon):
   
   """
    This function carries out the Monte Carlo Simulation.
    
    Input: dataframe of stock data, number of simulations needed, number of days wished to predict the price
    
    Output: a dataframe containing the daily price for all the simulation. The simulations are ordered from 1 to N
    """
# Run the simulation
   simulation_df = pd.DataFrame()

   for i in range(simulations):
    # The list to store the next stock price
        next_price = []
    # Create the next stock price
        last_price = stock_database['Close'].iloc[-1]
    
        for j in range(time_horizon):
        # Generate the random percentage change around the mean (0) and std (daily_volatility)
          future_return = np.random.normal(0, np.std(stock_database['Close'].pct_change()))

          # Generate the random future price
          future_price = last_price * (1 + future_return)

          # Save the price and go next
          next_price.append(future_price)
          last_price = future_price
    # Store the result of the simulation
        next_price_df = pd.Series(next_price).rename('sim' + str(i+1))
        simulation_df = pd.concat([simulation_df, next_price_df], axis=1)
   return simulation_df

def get_stock_news(stock_name):
    api_key="c8c81c12b46d483c815a3c48bc101ecd"
    newsapi = NewsApiClient(api_key=api_key)
    
    # Fetch articles related to the stock
    articles = newsapi.get_everything(q=stock_name, language="en", sort_by="relevancy")
    return articles["articles"]

## Header
col1,col2=st.columns([2,2])
col1.title("Welcome to my Financial Dashboard")
col2.image('C:/Users/jchreim/Downloads/background.webp', width=300)
# Create a selection box of all the S&P stocks for the user to choose
st.sidebar.title("Welcome to my financial dashboard")
st.sidebar.write("This dashboard is split into several tabs and each tab has its own function: ")
st.sidebar.write("In tab 1, you can see the summary of the S&P 500 stock of your choice.")
st.sidebar.write("In tab 2, you can see a chart showing the pric changes in details.")
st.sidebar.write("In tab 3, you can extract some statements of the chosen S&P 500 stock.")
st.sidebar.write("In tab 4, the Monte Carlo Simulation is performed.")
st.sidebar.write("In tab 5, all the news of the stock are seen.")

stock= st.sidebar.selectbox('Choose the stock:',extractStocks())
startDate = st.sidebar.date_input("Choose Start Date", datetime.today().date() - timedelta(days=60))
endDate = st.sidebar.date_input("Choose End Date", datetime.today().date())

#Store the dataframe of the chosen stock in a variable
stockData=getStockData(stock, startDate=startDate, endDate=endDate)

#Creating the Tabs
tab1, tab2, tab3, tab4, tab5=st.tabs(["Summary", "Chart","Financials","Monte Carlo simulation", "News"])


with tab1:
   

   #Getting stocks info
   stockinfo = yf.Ticker(stock)
   stock_info = stockinfo.info


   previousClose=stockData['Close'][-2]



   # Creating the Line plot
   fig=go.Figure()

   fig.add_trace(go.Bar( x=stockData.index, y=stockData['Volume'], name='Volume',yaxis='y1', 
                        marker_color='black', showlegend=False, opacity=0.6))

   fig.add_trace(go.Scatter(x=stockData.index, y=stockData['Close'],
                         fill='tozeroy', fillcolor='rgba(161,225,132,0.5)', line=dict(color='rgba(161,225,132,1)', width=1),
                         showlegend=False, yaxis='y2'))
   
   


   fin_buttons = [
     {'count': 1, 'label': "1M", 'step': 'month', 'stepmode': 'backward'},
     {'count': 3, 'label': "3M", 'step': 'month', 'stepmode': 'backward'},
     {'count': 6, 'label': "6M", 'step': 'month', 'stepmode': 'backward'},
     {'count': 1, 'label': "YTD", 'step': 'year', 'stepmode': 'todate'},
     {'count': 1, 'label': "1Y", 'step': 'year', 'stepmode': 'backward'},
     {'count': 3, 'label': "3Y", 'step': 'year', 'stepmode': 'backward'},
     {'count': 5, 'label': "5Y", 'step': 'year', 'stepmode': 'backward'},
     {'step': 'all', 'label': 'Max'}
    ]
   


   ## Adding the previous close line and point on the graph
   fig.add_shape(
       type="line",
       x0=stockData.index[0], x1=stockData.index[-1],
       y0=stockData['Close'][-2], y1=stockData['Close'][-2],
       line=dict(color="grey", width=2, dash="dash")
   )


   

   ## Adding the current close price on the graph
   fig.add_annotation(
        x=1,  # Position the annotation to the right
        xref="paper",
        yref="y2",
        y=stockData['Close'][-1],
        text=round(stockData['Close'][-1],2),
        showarrow=False,
        font=dict(size=16, color="black"),
        align="right",
        xanchor="left",
        yanchor="middle"
    )


   fig.update_layout (
    {'xaxis': {'rangeselector': {'buttons': fin_buttons}}},
    yaxis=dict(
        title="Volume",
        range=[0,1000000000]
    ),
    yaxis2=dict(
        title="Price",
        overlaying='y',  
        side='right'       # Move the y-axis to the right side
    ),
    title="Close Price and Volume Change of "+ stock_info.get("longName")
    )

   st.plotly_chart(fig)


   
   st.subheader("Some key figures")
   colA, colB= st.columns ([2,2]) #Creating two main columns
   

   # Creating the first table
   df=pd.DataFrame({
    'Attribute': ["Previous Close", "Open", "Volume", "Avg Volume"],
    'Value': [stockData['Close'][-2],stockData['Open'][-1], stockData['Volume'][-1], stockData['Volume'].mean()]
                   })
   
   df.set_index("Attribute", inplace=True)
   colA.table(df)



   # Creating the 2nd table
   # Get today's date
   today = datetime.now().date()

   # Filter the data for today
   today_data = yf.Ticker('aapl').history(start=datetime.now().date(), end=datetime.now().date())


   df2=pd.DataFrame({
    'Attribute': ["Bid", "Ask", "Low","High"],
    'Value': [stock_info['bid'],stock_info['ask'],stockData['Low'][-1],stockData['High'][-1]]
                   })
   
   df2.set_index("Attribute", inplace=True)
   colB.table(df2)


   colC, colD=st.columns(2)

   colC.markdown("##### Full Company Name: "+stock_info.get("longName"))
   colC.markdown("##### Sector: "+stock_info.get("sector"))
   colC.markdown("##### Industry: "+stock_info.get("industry"))
   colD.markdown("##### Website: "+stock_info.get("website"))
   colD.markdown("##### Country: "+stock_info.get("country"))
   colD.markdown("##### PE Ratio: "+ str(stock_info.get("trailingPE")))


   st.write('More Information on '+ "'"+stock_info.get("longName")+"'"+" can be found below")



   with st.expander('**1. Business Summary**'):
        st.markdown('<div style="text-align: justify;">' + \
                    GetCompanyInfo(stock)['longBusinessSummary'] + \
                    '</div><br>',
                    unsafe_allow_html=True)
   

   with st.expander ('**2. Shareholders Info**'):
       st.write("Major Holders")
       st.write(yf.Ticker(stock).major_holders)
       st.write("Institutional Holders")
       st.write(yf.Ticker(stock).institutional_holders)
       st.write("Mutual Fund Holders")
       st.write(yf.Ticker(stock).mutualfund_holders)


   with st.expander('**3. Stock Splits**'):
       st.write("Dates in which " + "'"+stock_info.get("longName") + "'"+" had to divide its shares into new shares to boost liquidity")
       st.write(yf.Ticker(stock).splits)


with tab2:
    colA, colB, colC, colD= st.columns ([1,1,1,1]) #Creating four columns

    #Asking the user to select a start and End Date
    tab2startDate = colA.date_input("Start Date", datetime.today().date() - timedelta(days=60))
    tab2endDate = colB.date_input("End Date", datetime.today().date())

    tab2stockData=getStockData(stock, startDate=tab2startDate, endDate=tab2endDate)


    fig2=go.Figure()

    price =tab2stockData['Close']
    volume=tab2stockData['Volume']

    # Asking the user to select the tme interval he wants to see
    # Resampling the data we will use according to the interval chosen
    interval = colC.selectbox("Select Time Interval:", ("Day", "Month", "Year"))
    if interval == "Day":
        resampled_price = price
        resampled_volume=volume
        x_data = resampled_price.index 
        y_data = resampled_price
        resampled_data = tab2stockData
        
    elif interval == "Month":
        resampled_price = price.resample("M").sum()
        resampled_volume=volume.resample("M").sum()
        x_data = resampled_price.index  
        y_data = resampled_price
        resampled_data = tab2stockData.resample("M").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
         })

    elif interval == "Year":
        resampled_price = price.resample("Y").sum()
        resampled_volume=volume.resample("Y").sum()
        x_data = resampled_price.index  
        y_data = resampled_price
        resampled_data = tab2stockData.resample("Y").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
        })

    # Asking the user to choose which plot he wants to see. By default it will be a line plot
    plot = colD.selectbox("Select your plot:", ("Line Plot", "Candlestick"))

    if plot=="Line Plot":

        fig2.add_trace(go.Scatter(x=x_data, y=y_data,
                         mode="lines", line=dict(color='rgba(49,154,0,1)'),
                         showlegend=False, yaxis='y2'))

    elif plot=="Candlestick":
        fig2.add_trace(go.Candlestick(x=x_data, open=resampled_data['Open'],high=resampled_data['High'],
                         low=resampled_data['Low'],
                         close=resampled_data['Close'],
                         showlegend=False, yaxis='y2'))
   
    else:
        fig2.add_trace(go.Scatter(x=x_data, y=y_data,
                         mode="lines", line=dict(color='rgba(49,154,0,1)'),
                         showlegend=False, yaxis='y2'))
    

    # Adding a bar chart showing the volume
    fig2.add_trace(go.Bar(x=resampled_data.index, y=resampled_volume, marker_color=np.where(resampled_data['Close'].pct_change() < 0, 'red', 'green'),
                  showlegend=False, yaxis='y1'))
    

    #Calculating the Moving Average on a 50 days window
    resampled_data['Moving Average'] = resampled_data['Close'].rolling(window=50).mean()

    # Line plot showing the moving average
    fig2.add_trace(go.Scatter(
    x=x_data,
    y=resampled_data['Moving Average'],
    mode="lines",
    line=dict(color='blue', width=1.5),
    name="Moving Average", yaxis='y2'  #Linking the line to the second yaxis
    ))
   
    #Adding the different duration of time
    fin_buttons = [
     {'count': 1, 'label': "1M", 'step': 'month', 'stepmode': 'backward'},
     {'count': 3, 'label': "3M", 'step': 'month', 'stepmode': 'backward'},
     {'count': 6, 'label': "6M", 'step': 'month', 'stepmode': 'backward'},
     {'count': 1, 'label': "YTD", 'step': 'year', 'stepmode': 'todate'},
     {'count': 1, 'label': "1Y", 'step': 'year', 'stepmode': 'backward'},
     {'count': 3, 'label': "3Y", 'step': 'year', 'stepmode': 'backward'},
     {'count': 5, 'label': "5Y", 'step': 'year', 'stepmode': 'backward'},
     {'step': 'all', 'label': 'Max'}
    ]

    
   
    fig2.update_layout (
    {'xaxis': {'rangeselector': {'buttons': fin_buttons}}},
    yaxis=dict(
        title="Volume",
        range=[0,1000000000]
    ),
    yaxis2=dict(
        title="Price",
        overlaying='y',  
        side='right'),
    title='Price & Volume Variation for '+stock_info.get('longName')
    )

    st.plotly_chart(fig2)

    
with tab3:
    colA, colB= st.columns ([1,1]) #Creating two columns for the selection boxes

    # Creating selection boxes for the report and the period
    financials=colA.selectbox(label="Select the financial Report you wish to see from "+stock_info.get("longName"), options=['Income Statement','Balance Sheet', 'Cash Flow'])
    period=colB.selectbox("Select how you wish to see the "+financials+" of "+ stock_info.get("longName"),['Annual','Quarterly'])
    if financials=='Income Statement':
       if period=='Quarterly':
           st.write(stockinfo.quarterly_financials)
       else:
           st.write(stockinfo.financials)
    elif financials=='Balance Sheet':
        if period=='Quarterly':
           st.write(stockinfo.quarterly_balanche_sheet)
        else:
           st.write(stockinfo.balance_sheet)
    elif financials=='Cash Flow':
        if period=='Quarterly':
           st.write(stockinfo.quarterly_cashflow)
        else:
           st.write(stockinfo.cashflow)
    else:
        st.write(stockinfo.financials)


with tab4:
   st.subheader('Monte Carlo Simulation')
   colE, colF= st.columns ([2,2]) #Creating two main columns
   simulations = colE.selectbox("Choose the number of simulations: ",[200,500,1000])
   time_horizon = colF.selectbox("Choose the number of days",[30,60,90])

   

   simulation_df=MonteCarloSimulation(stockData, simulations, time_horizon)
   

   # Get the ending price of the nth day
   ending_price = simulation_df.iloc[-1:, :].values[0, ]
   # Price at 95% confidence interval
   future_price_95ci = np.percentile(ending_price, 5)

   # Value at Risk
   # 95% of the time, the losses will not be more than 16.35 USD
   VaR = stockData['Close'].iloc[-1] - future_price_95ci
   st.write('**VaR at 95% confidence interval is:** ' + str(np.round(VaR, 2)) + ' **USD**')
   with st.expander('***What is Monte Carlo Simulation?***'):
    st.write("A Monte Carlo simulation is a way to model the probability of different outcomes in a process that cannot easily be predicted due to the intervention of random variables. It is a technique used to understand the impact of risk and uncertainty. Monte Carlo simulations can be applied to a range of problems in many fields, including investing, business, physics, and engineering. It is also referred to as a multiple probability simulation.")
    st.markdown("[Click here for more info](https://www.investopedia.com/terms/m/montecarlosimulation.asp)")

   with st.expander('***What is VaR?***'):
    st.write("Value at risk (VaR) is a statistic that quantifies the extent of possible financial losses within a firm, portfolio, or position over a specific time frame. This metric is most commonly used by investment and commercial banks to determine the extent and probabilities of potential losses in their institutional portfolios.")
    st.write("Risk managers use VaR to measure and control the level of risk exposure. One can apply VaR calculations to specific positions or whole portfolios or use them to measure firm-wide risk exposure.")
    st.markdown("[Click here for more info](https://www.investopedia.com/terms/v/var.asp)")



    # Plot the simulation stock price in the future
   fig3, ax = plt.subplots(figsize=(15, 10))
    # Plot the prices
   ax.plot(simulation_df)
   ax.axhline(y=stockData['Close'].iloc[-1], color='red')
   # Customize the plot
   ax.set_title('Monte Carlo simulation for ' + stock_info.get("longName")+ ' stock price in the next ' +str(time_horizon)+ ' days')
   ax.set_xlabel('Day')
   ax.set_ylabel('Price')
   ax.legend(['Current stock price is: ' + str(np.round(stockData['Close'].iloc[-1], 2))])
   ax.get_legend().legend_handles[0].set_color('red')
   
   
   st.pyplot (fig3)


with tab5:
    cols=st.columns (2)
    news=get_stock_news(stock_info.get("longName"))
    for idx, article in enumerate(news[:10]):  
       column = cols[idx % 2]
       with column:
         st.subheader(f"Title: {article['title']}")
         st.write(f"More details: {article['url']}")
         image_url = article.get("urlToImage")
         if image_url:
            st.image(image_url, use_column_width=True)
         else:
            st.write("No image found.")
         st.write("---")


    
    
    



