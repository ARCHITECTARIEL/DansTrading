import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import json
import os
from uuid import uuid4
from datetime import datetime, timedelta

# File paths
PROGRESS_FILE = "user_progress.json"
JOURNAL_FILE = "trade_journal.json"

# Topstep Combine rules ($50K account)
TOPSTEP_RULES = {
    "daily_loss_limit": 2000,
    "max_drawdown": 3000,
    "profit_target": 3000,
    "min_trading_days": 5,
    "max_position_size": 5,
    "consistency_rule": 0.3  # No single day >30% of profits
}

# Course structure (expanded to all lessons)
COURSE_CONTENT = {
    "Module 1: Market Structure & Price Behavior": [
        {
            "title": "Understanding Market Structure",
            "content": """
**Lesson 1: Understanding Market Structure**

Market structure defines trend direction using Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), and Lower Lows (LL).

**How to Read:**
1. Use 5-min or 15-min charts.
2. Mark 3-5 swing highs/lows.
3. HH/HL = uptrend; LH/LL = downtrend.
4. Break of HL/LH signals reversal.

**Example**: NQ rallies to 14520 (HH), pulls to 14470 (HL), pushes to 14560 (HH). Buy pullbacks to 14470.
            """,
            "quiz": [
                {
                    "question": "What confirms an uptrend?",
                    "options": ["LH/LL", "HH/HL", "Flat highs/lows"],
                    "answer": "HH/HL",
                    "feedback": "Correct! HH/HL indicate bullish momentum (Lesson 1)."
                },
                {
                    "question": "NQ makes HH at 14520, HL at 14470, then breaks below 14470. What’s likely?",
                    "options": ["Continuation", "Reversal", "Chop"],
                    "answer": "Reversal",
                    "feedback": "Correct! Breaking a HL suggests a potential trend change (Lesson 1)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ forms HH at 14520, HL at 14470. Price pulls to 14470 with a bullish engulfing. Action?",
                    "options": ["Buy", "Sell", "Wait"],
                    "answer": "Buy",
                    "feedback": "Correct! Buy at HL with confirmation, targeting next HH (Lesson 1)."
                }
            ]
        },
        {
            "title": "Spotting Supply & Demand Zones",
            "content": """
**Lesson 2: Spotting Supply & Demand Zones**

Zones show where institutions buy (demand) or sell (supply).

**Identification:**
1. Find tight consolidation.
2. Look for 3+ candle breakout.
3. Mark highest/lowest candle.
4. Retest with engulfing confirms.

**Example**: NQ consolidates 14200-14210, breaks to 14280, retests 14212 with bullish engulfing. Buy at 14215.
            """,
            "quiz": [
                {
                    "question": "What confirms a demand zone?",
                    "options": ["Break below", "Bullish engulfing", "Above VWAP"],
                    "answer": "Bullish engulfing",
                    "feedback": "Correct! Engulfing at zone confirms institutional buying (Lesson 2)."
                },
                {
                    "question": "ES consolidates 4550-4560, drops to 4520, retests 4558 with pin bar. Supply zone?",
                    "options": ["Yes", "No", "Unclear"],
                    "answer": "Yes",
                    "feedback": "Correct! Consolidation and rejection at 4558 form a supply zone (Lesson 2)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ at 14200-14210 breaks to 14280, retests 14212 with engulfing. Mark zone and trade.",
                    "options": ["Buy 14215", "Sell 14215", "Wait"],
                    "answer": "Buy 14215",
                    "feedback": "Correct! Buy at demand zone retest, targeting 14280 (Lesson 2)."
                }
            ]
        },
        {
            "title": "Liquidity Sweeps & Stop Hunts",
            "content": """
**Lesson 3: Liquidity Sweeps & Stop Hunts**

Stop hunts target retail stops beyond swing highs/lows.

**How to Trade:**
1. Mark swing highs/lows.
2. Wait for spike past level.
3. Enter on reversal candle.

**Example**: CL dips to 84.95, sweeps 85.10 low, forms bullish engulfing. Buy at 85.18, stop 84.85.
            """,
            "quiz": [
                {
                    "question": "Where do stop hunts occur?",
                    "options": ["At VWAP", "Beyond swing highs/lows", "In consolidation"],
                    "answer": "Beyond swing highs/lows",
                    "feedback": "Correct! Stop hunts target liquidity past key levels (Lesson 3)."
                }
            ],
            "scenarios": [
                {
                    "description": "ES spikes above 4565 to 4572, reverses with bearish engulfing. Trade?",
                    "options": ["Short 4563", "Long 4572", "Wait"],
                    "answer": "Short 4563",
                    "feedback": "Correct! Short after stop hunt reversal, targeting 4538 (Lesson 3)."
                }
            ]
        }
    ],
    "Module 2: Tools That Actually Work": [
        {
            "title": "VWAP - The Intraday Anchor",
            "content": """
**Lesson 4: VWAP - The Intraday Anchor**

VWAP shows fair value, used by institutions.

**Trading Styles:**
1. Trend: Buy pullbacks to VWAP.
2. Reversion: Fade extensions.

**Example**: NQ pulls to VWAP 14452, forms engulfing. Buy 14454, stop 14440.
            """,
            "quiz": [
                {
                    "question": "Price above VWAP indicates?",
                    "options": ["Bearish", "Bullish", "Neutral"],
                    "answer": "Bullish",
                    "feedback": "Correct! Above VWAP signals bullish bias (Lesson 4)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ pulls to VWAP 14452, forms engulfing. Trade?",
                    "options": ["Buy 14454", "Sell 14454", "Wait"],
                    "answer": "Buy 14454",
                    "feedback": "Correct! Buy at VWAP pullback, targeting 14500 (Lesson 4)."
                }
            ]
        },
        {
            "title": "Moving Averages (EMA/SMA)",
            "content": """
**Lesson 5: Moving Averages**

EMAs define trends and pullbacks.

**Strategies:**
1. Pullback: Buy EMA 21 in uptrend.
2. Cross: EMA 9 over 21 signals reversal.

**Example**: NQ pulls to EMA 21 at 14515, holds. Buy 14515, stop 14495.
            """,
            "quiz": [
                {
                    "question": "EMA 9 crossing above EMA 21 suggests?",
                    "options": ["Sell", "Buy", "Chop"],
                    "answer": "Buy",
                    "feedback": "Correct! Cross signals bullish momentum (Lesson 5)."
                }
            ],
            "scenarios": [
                {
                    "description": "ES trends down, EMA 9 crosses above EMA 21 at 4562. Trade?",
                    "options": ["Long 4562", "Short 4562", "Wait"],
                    "answer": "Long 4562",
                    "feedback": "Correct! Buy on EMA cross, targeting 4580 (Lesson 5)."
                }
            ]
        },
        {
            "title": "Fibonacci Pullbacks",
            "content": """
**Lesson 6: Fibonacci Pullbacks**

Fib levels (38.2%, 50%, 61.8%) spot pullbacks.

**How to Draw:**
1. Select impulse leg.
2. Draw low to high (bullish).
3. Buy at 61.8% with confirmation.

**Example**: NQ pulls to 61.8% at 14380, engulfing. Buy 14382, stop 14370.
            """,
            "quiz": [
                {
                    "question": "Key Fibonacci level for deep pullbacks?",
                    "options": ["38.2%", "50%", "61.8%"],
                    "answer": "61.8%",
                    "feedback": "Correct! 61.8% is the golden ratio for pullbacks (Lesson 6)."
                }
            ],
            "scenarios": [
                {
                    "description": "CL drops from 86.20 to 84.90, pulls to 38.2% at 85.40 with pin bar. Trade?",
                    "options": ["Short 85.38", "Long 85.38", "Wait"],
                    "answer": "Short 85.38",
                    "feedback": "Correct! Short at 38.2% rejection, targeting 84.60 (Lesson 6)."
                }
            ]
        },
        {
            "title": "VWAP + EMA Cross Strategy",
            "content": """
**Lesson 7: VWAP + EMA Cross**

Combines VWAP and EMA 9 for trend shifts.

**Setup:**
1. EMA 9 crosses VWAP.
2. Confirm with candle close.

**Example**: NQ EMA 9 crosses VWAP at 14424, engulfing. Buy 14426, stop 14414.
            """,
            "quiz": [
                {
                    "question": "What confirms VWAP+EMA cross?",
                    "options": ["Volume spike", "Engulfing candle", "Price at fib"],
                    "answer": "Engulfing candle",
                    "feedback": "Correct! Candle confirms momentum (Lesson 7)."
                }
            ],
            "scenarios": [
                {
                    "description": "ES EMA 9 breaks below VWAP at 4561, bearish candle. Trade?",
                    "options": ["Short 4561", "Long 4561", "Wait"],
                    "answer": "Short 4561",
                    "feedback": "Correct! Short on cross, targeting 4535 (Lesson 7)."
                }
            ]
        }
    ],
    "Module 3: Smart Money Plays": [
        {
            "title": "Open Range Breakouts (ORB)",
            "content": """
**Lesson 8: Open Range Breakouts**

ORB catches momentum after 15-30 min open range.

**Strategy:**
1. Mark 9:30-9:45 high/low.
2. Trade breakout with volume.

**Example**: NQ ORB 14380-14420, breaks 14422. Buy 14422, stop 14410.
            """,
            "quiz": [
                {
                    "question": "Best ORB time?",
                    "options": ["Lunch", "First 90 min", "Close"],
                    "answer": "First 90 min",
                    "feedback": "Correct! High volume at open drives ORB (Lesson 8)."
                }
            ],
            "scenarios": [
                {
                    "description": "ES ORB 4550-4570, breaks 4550 with bearish engulfing. Trade?",
                    "options": ["Short 4548", "Long 4550", "Wait"],
                    "answer": "Short 4548",
                    "feedback": "Correct! Short on ORB breakdown, targeting 4525 (Lesson 8)."
                }
            ]
        },
        {
            "title": "Power of 3",
            "content": """
**Lesson 9: Power of 3**

Cycle of accumulation, manipulation, distribution.

**Phases:**
1. Accumulation: Tight range.
2. Manipulation: Fake breakout.
3. Distribution: Real move.

**Example**: NQ ranges 14310-14330, spikes to 14340, drops to 14250. Short 14310.
            """,
            "quiz": [
                {
                    "question": "Manipulation phase involves?",
                    "options": ["Tight range", "Fake breakout", "Trend"],
                    "answer": "Fake breakout",
                    "feedback": "Correct! Fakeout grabs liquidity (Lesson 9)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ ranges 14310-14330, spikes to 14340, reverses. Trade?",
                    "options": ["Short 14310", "Long 14340", "Wait"],
                    "answer": "Short 14310",
                    "feedback": "Correct! Short after fakeout, targeting 14250 (Lesson 9)."
                }
            ]
        }
    ],
    "Module 4: Strategy Integration & Topstep Prep": [
        {
            "title": "Trade Setup Framework",
            "content": """
**Lesson 10: Trade Setup Framework**

Build a repeatable system.

**Steps:**
1. Bias: Check trend, VWAP.
2. Liquidity: Find stops.
3. Setup: Pick ORB, VWAP, etc.
4. Entry: Confirm with candle.

**Example**: NQ pulls to 61.8%+VWAP at 14392, engulfing. Buy 14392.
            """,
            "quiz": [
                {
                    "question": "First step in framework?",
                    "options": ["Entry", "Bias", "Setup"],
                    "answer": "Bias",
                    "feedback": "Correct! Bias sets trade direction (Lesson 10)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ at 61.8%+VWAP 14392, bullish engulfing. Trade?",
                    "options": ["Buy 14392", "Sell 14392", "Wait"],
                    "answer": "Buy 14392",
                    "feedback": "Correct! Buy with confluence, targeting 14440 (Lesson 10)."
                }
            ]
        },
        {
            "title": "Journaling & Backtesting",
            "content": """
**Lesson 11: Journaling & Backtesting**

Journal trades to improve.

**Journal Includes:**
- Date, setup, reason, emotion.
- Screenshots, outcome.

**Backtesting:**
- Test 50-100 trades.
- Measure win rate, R.

**Example**: ORB buy at 14522, +2.3R. Note confidence.
            """,
            "quiz": [
                {
                    "question": "Backtesting needs how many trades?",
                    "options": ["10-20", "50-100", "200+"],
                    "answer": "50-100",
                    "feedback": "Correct! 50-100 trades ensure reliability (Lesson 11)."
                }
            ],
            "scenarios": [
                {
                    "description": "You backtest VWAP pullbacks, win rate 60%. Action?",
                    "options": ["Trade live", "Test more", "Abandon"],
                    "answer": "Test more",
                    "feedback": "Correct! Refine with more data before live trading (Lesson 11)."
                }
            ]
        },
        {
            "title": "Topstep Combine Readiness",
            "content": """
**Lesson 12: Topstep Combine Readiness**

Prepare for evaluation.

**Checklist:**
- Stick to rules 20+ sessions.
- Trade 1-2 setups.
- Pass simulated test.

**Example**: Pass $50K Combine with $3,000 profit, no rule breaks.
            """,
            "quiz": [
                {
                    "question": "Key to Topstep success?",
                    "options": ["High leverage", "Discipline", "Many setups"],
                    "answer": "Discipline",
                    "feedback": "Correct! Discipline ensures rule adherence (Lesson 12)."
                }
            ],
            "scenarios": [
                {
                    "description": "You hit $2,000 profit in 3 days. Action?",
                    "options": ["Scale up", "Stay consistent", "Stop trading"],
                    "answer": "Stay consistent",
                    "feedback": "Correct! Consistency meets Topstep goals (Lesson 12)."
                }
            ]
        }
    ]
}

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=4)

def load_journal():
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, 'r') as f:
            return json.load(f)
    return []

def save_journal(journal):
    with open(JOURNAL_FILE, 'w') as f:
        json.dump(journal, f, indent=4)

def fetch_futures_data(symbol="NQ=F", period="1d", interval="5m"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data
    except Exception:
        st.error("Error fetching data. Using mock data.")
        return pd.DataFrame({
            'Open': [14500, 14510, 14505],
            'High': [14515, 14520, 14510],
            'Low': [14495, 14500, 14490],
            'Close': [14510, 14505, 14500],
            'Volume': [1000, 1200, 1100]
        }, index=pd.date_range(start=pd.Timestamp.now(), periods=3, freq='5min'))

def calculate_vwap(df):
    df['Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['PriceVolume'] = df['Price'] * df['Volume']
    df['CumulativePriceVolume'] = df['PriceVolume'].cumsum()
    df['CumulativeVolume'] = df['Volume'].cumsum()
    df['VWAP'] = df['CumulativePriceVolume'] / df['CumulativeVolume']
    return df

def plot_candlestick_chart(data, title, annotations=None):
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlesticks"
        ),
        go.Scatter(
            x=data.index,
            y=data['VWAP'],
            mode='lines',
            name='VWAP',
            line=dict(color='purple')
        )
    ])
    if annotations:
        for ann in annotations:
            if ann['type'] in ["Demand", "Supply"]:
                fig.add_hrect(
                    y0=ann['y0'], y1=ann['y1'],
                    fillcolor="green" if ann['type'] == "Demand" else "red",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    annotation_text=ann['type']
                )
            elif ann['type'] == "Fibonacci":
                for level in ann['levels']:
                    fig.add_hline(
                        y=level['price'],
                        line_dash="dash",
                        line_color="blue",
                        annotation_text=f"{level['label']}%"
                    )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    return fig

def backtest_setup(data, setup_type, params):
    trades = []
    if setup_type == "VWAP Pullback":
        for i in range(1, len(data)):
            if data['Close'].iloc[i-1] > data['VWAP'].iloc[i-1] and data['Close'].iloc[i] <= data['VWAP'].iloc[i]:
                entry = data['Close'].iloc[i]
                stop = entry - params['stop_distance']
                target = entry + params['target_distance']
                for j in range(i+1, len(data)):
                    if data['Low'].iloc[j] <= stop:
                        trades.append({"result": -1, "R": -1})
                        break
                    if data['High'].iloc[j] >= target:
                        trades.append({"result": 1, "R": params['target_distance'] / params['stop_distance']})
                        break
    win_rate = sum(1 for t in trades if t['result'] > 0) / len(trades) if trades else 0
    avg_r = sum(t['R'] for t in trades) / len(trades) if trades else 0
    return {"trades": len(trades), "win_rate": win_rate, "avg_r": avg_r}

def evaluate_topstep_metrics(sim):
    daily_pnl = {}
    max_equity = sim['balance']
    current_equity = sim['balance']
    trading_days = set()
    total_profit = 0

    for trade in sim['trades']:
        price_change = trade['exit_price'] - trade['entry_price'] if trade['type'] == 'Buy' else trade['entry_price'] - trade['exit_price']
        trade_pnl = price_change * trade['size'] * 20
        day = trade['exit_time'].split('T')[0]
        daily_pnl[day] = daily_pnl.get(day, 0) + trade_pnl
        current_equity += trade_pnl
        max_equity = max(max_equity, current_equity)
        trading_days.add(day)
        total_profit += trade_pnl

    for day, pnl in daily_pnl.items():
        if pnl < -TOPSTEP_RULES['daily_loss_limit']:
            return False, f"Violated daily loss limit on {day}"
        if total_profit > 0 and abs(pnl) / total_profit > TOPSTEP_RULES['consistency_rule']:
            return False, f"Violated consistency rule on {day}"

    drawdown = max_equity - current_equity
    if drawdown > TOPSTEP_RULES['max_drawdown']:
        return False, "Violated max drawdown ($3,000)"

    if total_profit < TOPSTEP_RULES['profit_target']:
        return False, f"Profit ${total_profit:.2f} below target ($3,000)"

    if len(trading_days) < TOPSTEP_RULES['min_trading_days']:
        return False, f"Only {len(trading_days)} trading days (need 5)"

    return True, "Passed Topstep Combine!"

def main():
    st.title("Pro Trader Futures Course Agent")
    st.write("Master futures trading with comprehensive lessons, simulations, and Topstep preparation.")

    # Initialize session state
    if 'module' not in st.session_state:
        st.session_state.module = None
    if 'lesson' not in st.session_state:
        st.session_state.lesson = None
    if 'trade_simulator' not in st.session_state:
        st.session_state.trade_simulator = {
            'balance': 100000,
            'positions': [],
            'trades': [],
            'equity': 100000
        }
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    if 'quiz_scores' not in st.session_state:
        st.session_state.quiz_scores = {}
    if 'scenario_scores' not in st.session_state:
        st.session_state.scenario_scores = {}

    # Load user data
    progress = load_progress()
    journal = load_journal()

    # Sidebar
    st.sidebar.header("Course Modules")
    for mod in COURSE_CONTENT.keys():
        if st.sidebar.button(mod, key=f"mod_{mod}"):
            st.session_state.module = mod
            st.session_state.lesson = None

    # Ticker
    st.sidebar.header("NQ Futures Ticker")
    try:
        ticker_data = fetch_futures_data(symbol="NQ=F", period="1d", interval="1m")
        latest = ticker_data.iloc[-1]
        st.sidebar.write(f"**Last Price**: ${latest['Close']:.2f}")
        st.sidebar.write(f"**
