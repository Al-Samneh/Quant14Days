# Day 8: Market Microstructure
## This project will represent a market order book
class OrderBook:

    def __init__(self):
        self.bids = [] # List of bids
        self.asks = [] # List of asks

    def add_bid(self, price, quantity):
        """
        Add a bid to the order book
        """
        self.bids.append((price, quantity))
        self.sort_bids()
    

    def add_ask(self, price, quantity):
        """
        Add an ask to the order book
        """
        self.asks.append((price, quantity))
        self.sort_asks()

    def remove_bid(self, price, quantity):
        """
        Remove a bid from the order book
        """
        self.bids.remove((price, quantity))

    def remove_ask(self, price, quantity):
        """
        Remove an ask from the order book
        """
        self.asks.remove((price, quantity))

    def get_best_bid(self):
        """
        Get the best bid from the order book
        """
        if not self.bids:
            return None
        self.sort_bids()
        return self.bids[0]
    
    def get_best_ask(self):
        """
        Get the best ask from the order book
        """
        if not self.asks:
            return None
        self.sort_asks()
        return self.asks[0]

    def sort_bids(self):
        """
        Sort the bids by price in descending order
        """
        self.bids.sort(key=lambda x: x[0], reverse=True)

    def sort_asks(self):
        """
        Sort the asks by price in ascending order
        """
        self.asks.sort(key=lambda x: x[0])
    
    def print_order_book(self):
        """
        Print the order book
        """
        print("Bids:")
        for bid in self.bids:
            print(f"Price: {bid[0]}, Quantity: {bid[1]}")
        print("Asks:")
        for ask in self.asks:
            print(f"Price: {ask[0]}, Quantity: {ask[1]}")
    
    def mid_price(self):
        """
        Get the mid price from the order book
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid[0] + best_ask[0]) / 2
    
    def best_order_volumes(self):
        """
        Get the best order volumes from the order book
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        return (best_bid[1] if best_bid else 0, best_ask[1] if best_ask else 0)
    
    def total_volume(self):
        """
        Get the total volume from the order book
        """
        return sum(bid[1] for bid in self.bids) + sum(ask[1] for ask in self.asks)
    
    def best_bid_volume(self):
        """
        Get the best bid volume from the order book
        """
        best_bid = self.get_best_bid()
        return best_bid[1] if best_bid else 0
    
    def best_ask_volume(self):
        """
        Get the best ask volume from the order book
        """
        best_ask = self.get_best_ask()
        return best_ask[1] if best_ask else 0

    def order(self,quantity,action, type="limit",price=0):
        """
        Add an order to the order book
        """
        if type == "market": # If type of order is market
            if action == "bid": # Market buy
                while len(self.asks)>0 and quantity>0: # While there are asks and quantity is greater than 0
                    bestask = self.get_best_ask()
                    if bestask[1] > quantity: # If the best ask is greater than quantity
                        # I know I could have just updated lol
                        self.remove_ask(bestask[0], bestask[1]) # Remove the best ask
                        self.add_ask(bestask[0], bestask[1] - quantity) # Add the best ask minus quantity
                        print(f"Bought {quantity} at {bestask[0]}")
                        quantity = 0 # Set quantity to 0
                    else:
                        self.remove_ask(bestask[0], bestask[1]) # Remove the best ask
                        print(f"Bought {bestask[1]} at {bestask[0]}") # Print the best ask
                        quantity -= bestask[1]
                if quantity > 0:
                    print("Liquidity is insufficient") # Print liquidity is insufficient
            elif action == "ask": # Market sell
                while len(self.bids)>0 and quantity>0:
                    bestbid = self.get_best_bid() # Get the best bid
                    if bestbid[1] > quantity:
                        self.remove_bid(bestbid[0], bestbid[1]) # Remove the best bid
                        self.add_bid(bestbid[0], bestbid[1] - quantity) # Add the best bid minus quantity
                        print(f"Sold {quantity} at {bestbid[0]}") # Print the best bid
                        quantity = 0
                    else:
                        self.remove_bid(bestbid[0], bestbid[1]) # Remove the best bid   
                        print(f"Sold {bestbid[1]} at {bestbid[0]}") # Print the best bid
                        quantity -= bestbid[1]
                if quantity > 0:
                    print("Liquidity is insufficient") # Print liquidity is insufficient
            else:
                raise ValueError("Invalid action")
        elif type == "limit": # Limit order
            if action == "bid": # Limit buy
                # Match against asks if crossing
                while quantity > 0 and self.asks and price >= self.get_best_ask()[0]: # While quantity is greater than 0 and there are asks and price is greater than or equal to the best ask
                    bestask = self.get_best_ask()
                    trade_qty = min(quantity, bestask[1])
                    self.remove_ask(bestask[0], bestask[1]) # Remove the best ask
                    residual = bestask[1] - trade_qty
                    if residual > 0:
                        self.add_ask(bestask[0], residual) # Add the best ask minus quantity
                    print(f"Bought {trade_qty} at {bestask[0]}") # Print the best ask
                    quantity -= trade_qty
                if quantity > 0:
                    self.add_bid(price, quantity) # Add the price and quantity
            elif action == "ask":
                # Match against bids if crossing
                while quantity > 0 and self.bids and price <= self.get_best_bid()[0]: # While quantity is greater than 0 and there are bids and price is less than or equal to the best bid
                    bestbid = self.get_best_bid()
                    trade_qty = min(quantity, bestbid[1])
                    self.remove_bid(bestbid[0], bestbid[1]) # Remove the best bid
                    residual = bestbid[1] - trade_qty
                    if residual > 0:
                        self.add_bid(bestbid[0], residual) # Add the best bid minus quantity
                    print(f"Sold {trade_qty} at {bestbid[0]}")
                    quantity -= trade_qty
                if quantity > 0:
                    self.add_ask(price, quantity) # Add the price and quantity
            else:
                raise ValueError("Invalid action")
        else:
            raise ValueError("Invalid type")
    
    def cancel_order(self, quantity, action, type="limit", price=0):
        """
        Cancel an order from the order book
        """
        if type == "market": # Market order
            if action == "bid":
                self.remove_bid(price, quantity)
            elif action == "ask":
                self.remove_ask(price, quantity)
            else:
                raise ValueError("Invalid action")
        elif type == "limit":
            if action == "bid":
                self.remove_bid(price, quantity)
            elif action == "ask":
                self.remove_ask(price, quantity)
            else:
                raise ValueError("Invalid action")
        else:
            raise ValueError("Invalid type")

    def spread(self):
        """
        Get the spread from the order book
        """
        best_bid = self.get_best_bid() # Get the best bid
        best_ask = self.get_best_ask() # Get the best ask
        if best_bid is None or best_ask is None:
            return None
        return best_ask[0] - best_bid[0]

# ========== Streamlit App ==========
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Simulated Order Book", layout="wide") # Set the page config
st.title("📊 Simulated Market Order Book")

# Session state
if "order_book" not in st.session_state:
    st.session_state.order_book = OrderBook()

book = st.session_state.order_book

# Sidebar order input
st.sidebar.header("➕ Add New Order")
order_type = st.sidebar.selectbox("Order Type", ["limit", "market"]) # Select the order type
side = st.sidebar.radio("Side", ["bid", "ask"])
quantity = st.sidebar.number_input("Quantity", min_value=1, value=10, step=1) # Select the quantity
price = None
if order_type == "limit":
    price = st.sidebar.number_input("Price", min_value=0.01, value=100.00, step=0.01, format="%.2f") # Select the price

if st.sidebar.button("Submit Order"):
    if order_type == "limit":
        if side == "bid": # Limit buy
            book.order(quantity, "bid", type="limit", price=price)
        else:
            book.order(quantity, "ask", type="limit", price=price)
    else:
        # Market order
        if side == "bid":
            book.order(quantity, "bid", type="market")
        else:
            book.order(quantity, "ask", type="market")

# Display order book
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Order Book")
    bids_df = pd.DataFrame(book.bids, columns=["Price", "Quantity"])
    asks_df = pd.DataFrame(book.asks, columns=["Price", "Quantity"])
    st.write("**Bids**")
    st.dataframe(bids_df)
    st.write("**Asks**")
    st.dataframe(asks_df)

with col2:
    st.subheader("📊 Market Stats")
    _best_bid = book.get_best_bid()
    _best_ask = book.get_best_ask()
    _spread = book.spread()
    _mid = book.mid_price()
    st.metric("Best Bid", _best_bid[0] if _best_bid else "-")
    st.metric("Best Ask", _best_ask[0] if _best_ask else "-")
    st.metric("Spread", _spread if _spread is not None else "-")
    st.metric("Mid Price", _mid if _mid is not None else "-")
    st.metric("Total Volume", book.total_volume())

# Market depth chart
st.subheader("📉 Market Depth (Cumulative)")
smooth_curves = st.toggle("Smooth curves", value=False, help="Use smooth curves instead of steps (visual only)")

# Aggregate quantities by price level
from collections import defaultdict
bid_levels = defaultdict(int)
ask_levels = defaultdict(int)
for p, q in book.bids:
    bid_levels[p] += q
for p, q in book.asks:
    ask_levels[p] += q

bid_prices = []
bid_cum = []
ask_prices = []
ask_cum = []

if bid_levels:
    bids_sorted = sorted(bid_levels.items(), key=lambda x: -x[0])
    prices, qtys = zip(*bids_sorted)
    cum = pd.Series(qtys).cumsum().tolist()
    bid_prices = list(prices)
    bid_cum = cum

if ask_levels:
    asks_sorted = sorted(ask_levels.items(), key=lambda x: x[0])
    prices, qtys = zip(*asks_sorted)
    cum = pd.Series(qtys).cumsum().tolist()
    ask_prices = list(prices)
    ask_cum = cum

fig = go.Figure()

line_shape = "spline" if smooth_curves else "hv"

if bid_prices:
    fig.add_trace(
        go.Scatter(
            x=bid_prices,
            y=bid_cum,
            mode="lines",
            name="Bids",
            line=dict(color="#2ca02c", width=2),
            fill="tozeroy",
            fillcolor="rgba(44,160,44,0.2)",
            line_shape=line_shape,
        )
    )

if ask_prices:
    fig.add_trace(
        go.Scatter(
            x=ask_prices,
            y=ask_cum,
            mode="lines",
            name="Asks",
            line=dict(color="#d62728", width=2),
            fill="tozeroy",
            fillcolor="rgba(214,39,40,0.2)",
            line_shape=line_shape,
        )
    )

best_bid = book.get_best_bid()
best_ask = book.get_best_ask()

shapes = []
annotations = []
if best_bid:
    shapes.append(dict(type="line", x0=best_bid[0], x1=best_bid[0], y0=0, y1=1, xref="x", yref="paper", line=dict(color="#2ca02c", dash="dash")))
    annotations.append(dict(x=best_bid[0], y=1.02, xref="x", yref="paper", text="Best Bid", showarrow=False, font=dict(color="#2ca02c")))
if best_ask:
    shapes.append(dict(type="line", x0=best_ask[0], x1=best_ask[0], y0=0, y1=1, xref="x", yref="paper", line=dict(color="#d62728", dash="dash")))
    annotations.append(dict(x=best_ask[0], y=1.02, xref="x", yref="paper", text="Best Ask", showarrow=False, font=dict(color="#d62728")))

fig.update_layout(
    title="Market Depth",
    xaxis_title="Price",
    yaxis_title="Cumulative Quantity",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    shapes=shapes,
    annotations=annotations,
    margin=dict(l=40, r=20, t=60, b=40),
)

if not bid_prices and not ask_prices:
    fig.add_annotation(text="Market is empty", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, opacity=0.6)

st.plotly_chart(fig, use_container_width=True)