import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import json
import os
from uuid import uuid4
from datetime import datetime, timedelta
import random

# File paths (Cloud-friendly)
PROGRESS_FILE = "user_progress.json"
JOURNAL_FILE = "trade_journal.json"

# Topstep rules ($50K account)
TOPSTEP_RULES = {
    "daily_loss_limit": 2000,
    "max_drawdown": 3000,
    "profit_target": 3000,
    "min_trading_days": 5,
    "max_position_size": 5,
    "consistency_rule": 0.3
}

# Course structure (deepened content)
COURSE_CONTENT = {
    "Module 1: Market Structure & Price Behavior": [
        {
            "title": "Understanding Market Structure",
            "content": """
**Lesson 1: Understanding Market Structure**

Market structure is the foundation of trend trading, revealing direction and reversals through Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), and Lower Lows (LL).

**Detailed Analysis:**
- **HH**: New peak above prior high (e.g., NQ from 14400 to 14520).
- **HL**: Pullback above prior low (e.g., 14470 vs. 14450).
- **LH/LL**: Bearish equivalents, signaling downtrends.
- Breaks in structure (e.g., price below HL) warn of reversals.

**How to Trade:**
1. Use 5-min or 15-min charts for futures (NQ, ES).
2. Zoom out for 3-5 swings.
3. Confirm trend with EMA 21 or VWAP.
4. Enter on pullbacks to HL/HH with engulfing candles.

**Examples:**
- **NQ Uptrend**: Rallies to 14520 (HH), pulls to 14470 (HL), pushes to 14560. Buy at 14472 (engulfing), stop 14455, target 14540.
- **ES Downtrend**: Drops to 4530 (LL), rallies to 4548 (LH), falls to 4515. Short 4546, stop 4552, target 4520.

**Common Mistakes:**
- Chasing highs without confirmation.
- Trading choppy structure (flat highs/lows).
- Ignoring VWAP/EMA confluence.

**Checklist:**
- Trend clear? Structure clean? Price at key level?
            """,
            "quiz": [
                {
                    "question": "What confirms an uptrend?",
                    "options": ["LH/LL", "HH/HL", "Flat highs/lows"],
                    "answer": "HH/HL",
                    "feedback": "Correct! HH/HL define bullish trends (Lesson 1)."
                },
                {
                    "question": "On the chart, NQ hits 14520 (HH), pulls to 14470 (HL), then breaks 14470. Action?",
                    "options": ["Buy", "Sell", "Wait"],
                    "answer": "Wait",
                    "feedback": "Correct! Breaking HL signals potential reversal; wait for confirmation (Lesson 1)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ forms HH at 14520, HL at 14470. Price pulls to 14470 with bullish engulfing. Step 1: Enter trade. Step 2: Set stop. Step 3: Set target.",
                    "steps": [
                        {"question": "Entry?", "options": ["Buy 14472", "Sell 14472", "Wait"], "answer": "Buy 14472"},
                        {"question": "Stop?", "options": ["14455", "14480", "14490"], "answer": "14455"},
                        {"question": "Target?", "options": ["14540", "14490", "14470"], "answer": "14540"}
                    ],
                    "feedback": "Correct! Buy at 14472, stop at 14455, target 14540 aligns with HL pullback (Lesson 1)."
                }
            ]
        },
        {
            "title": "Spotting Supply & Demand Zones",
            "content": """
**Lesson 2: Spotting Supply & Demand Zones**

Zones reveal institutional buying (demand) or selling (supply), driving price.

**In-Depth:**
- **Demand**: Buyers dominate at move’s base (e.g., 14200-14210).
- **Supply**: Sellers at rally tops (e.g., 4618 on ES).
- Strong moves (3+ candles) from consolidation mark zones.

**Trading Steps:**
1. Identify base on 5-min chart.
2. Mark breakout’s highest/lowest candle.
3. Wait for retest with rejection/engulfing.
4. Enter with stop below/above zone.

**Examples:**
- **NQ Demand**: Consolidates 14200-14210, breaks to 14280, retests 14212 (engulfing). Buy 14215, stop 14197, target 14275.
- **ES Supply**: Grinds to 4620, drops to 4590, retests 4618 (pin bar). Short 4616, stop 4624, target 4595.

**Common Mistakes:**
- Entering without confirmation.
- Trading multi-hit zones (weaker).
- Ignoring low-volume sessions.

**Checklist:**
- Sharp breakout? Clean retest? Confluence with VWAP?
            """,
            "quiz": [
                {
                    "question": "What confirms a supply zone?",
                    "options": ["Break above", "Bearish pin bar", "High volume"],
                    "answer": "Bearish pin bar",
                    "feedback": "Correct! Pin bar at zone shows seller control (Lesson 2)."
                },
                {
                    "question": "Chart shows NQ at 14200-14210, breaks to 14280, retests 14212. Demand zone?",
                    "options": ["Yes", "No", "Unclear"],
                    "answer": "Yes",
                    "feedback": "Correct! Consolidation and retest confirm demand (Lesson 2)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ consolidates 14200-14210, breaks to 14280, retests 14212 with engulfing. Step 1: Enter. Step 2: Set stop. Step 3: Set target.",
                    "steps": [
                        {"question": "Entry?", "options": ["Buy 14215", "Sell 14215", "Wait"], "answer": "Buy 14215"},
                        {"question": "Stop?", "options": ["14197", "14225", "14250"], "answer": "14197"},
                        {"question": "Target?", "options": ["14275", "14230", "14200"], "answer": "14275"}
                    ],
                    "feedback": "Correct! Buy at 14215, stop at 14197, target 14275 (Lesson 2)."
                }
            ]
        },
        # Add other lessons similarly (omitted for brevity)
    ],
    "Module 2: Tools That Actually Work": [
        {
            "title": "VWAP - The Intraday Anchor",
            "content": """
**Lesson 4: VWAP - The Intraday Anchor**

VWAP (Volume Weighted Average Price) is institutional fair value.

**Deep Dive:**
- Calculated from open, weighted by volume.
- Above VWAP = bullish; below = bearish.
- Acts as dynamic support/resistance.

**Strategies:**
1. **Trend Continuation**: Buy pullbacks with engulfing.
2. **Mean Reversion**: Fade extensions with low volume.

**Examples:**
- **NQ Trend**: Breaks VWAP at 14452, pulls back, holds with engulfing. Buy 14454, stop 14440, target 14500.
- **ES Fade**: Runs to 4568, VWAP at 4550, slows. Short 4566, stop 4572, target 4552.

**Common Mistakes:**
- Trading VWAP in chop without structure.
- Ignoring momentum trends.
- Using VWAP alone.

**Checklist:**
- Price above/below VWAP? Confirmation candle? Trend context?
            """,
            "quiz": [
                {
                    "question": "Price above VWAP suggests?",
                    "options": ["Bearish", "Bullish", "Neutral"],
                    "answer": "Bullish",
                    "feedback": "Correct! Bullish bias above VWAP (Lesson 4)."
                },
                {
                    "question": "Chart: NQ pulls to VWAP 14452, bullish engulfing. Trade?",
                    "options": ["Buy", "Sell", "Wait"],
                    "answer": "Buy",
                    "feedback": "Correct! Buy at VWAP pullback with confirmation (Lesson 4)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ breaks VWAP at 14450, pulls to 14452, engulfing forms. Step 1: Enter. Step 2: Stop. Step 3: Target.",
                    "steps": [
                        {"question": "Entry?", "options": ["Buy 14454", "Sell 14454", "Wait"], "answer": "Buy 14454"},
                        {"question": "Stop?", "options": ["14440", "14460", "14470"], "answer": "14440"},
                        {"question": "Target?", "options": ["14500", "14480", "14450"], "answer": "14500"}
                    ],
                    "feedback": "Correct! Buy at 14454, stop at 14440, target 14500 (Lesson 4)."
                }
            ]
        }
        # Add other lessons (EMA, Fibonacci, VWAP+EMA) with similar depth
    ]
    # Modules 3 and 4 added in previous artifact
}

def load_progress():
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_progress(progress):
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=4)
    except Exception:
        pass

def load_journal():
    try:
        if os.path.exists(JOURNAL_FILE):
            with open(JOURNAL_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_journal(journal):
    try:
        with open(JOURNAL_FILE, 'w') as f:
            json.dump(journal, f, indent=4)
    except Exception:
        pass

def fetch_futures_data(symbol="NQ=F", period="1d", interval="5m"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data
    except Exception:
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

def simulate_trading_session():
    setups = ["VWAP Pullback", "ORB Breakout", "Stop Hunt", "Fibonacci Retracement"]
    events = ["Normal Day", "High Volatility (FOMC)", "Low Volume"]
    session = []
    for _ in range(random.randint(3, 6)):
        setup = random.choice(setups)
        event = random.choice(events)
        price = 14500 + random.randint(-50, 50)
        session.append({
            "setup": setup,
            "event": event,
            "price": price,
            "description": f"{setup} at ${price:.2f} during {event}.",
            "action": random.choice(["Buy", "Sell", "Wait"])
        })
    return session

def main():
    st.title("Pro Trader Futures Course Agent")
    st.write("Comprehensive futures trading education with advanced simulations, analytics, and Topstep preparation.")

    # Session state
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
    if 'session_results' not in st.session_state:
        st.session_state.session_results = []

    # Load data
    progress = load_progress()
    journal = load_journal()

    # Sidebar
    st.sidebar.header("Modules")
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
        st.sidebar.write(f"**Change**: {(latest['Close'] - latest['Open']):.2f} ({((latest['Close'] - latest['Open']) / latest['Open'] * 100):.2f}%)")
    except Exception:
        st.sidebar.write("**Last Price**: $14500.00 (Mock)")

    # Dashboard
    st.sidebar.header("Dashboard")
    total_lessons = sum(len(lessons) for lessons in COURSE_CONTENT.values())
    completed = sum(1 for k, v in progress.items() if v)
    st.sidebar.progress(completed / total_lessons)
    st.sidebar.write(f"Completed {completed}/{total_lessons} lessons")
    quiz_avg = sum(st.session_state.quiz_scores.values()) / len(st.session_state.quiz_scores) if st.session_state.quiz_scores else 0
    st.sidebar.write(f"Quiz Score: {quiz_avg:.0%}")
    scenario_avg = sum(st.session_state.scenario_scores.values()) / len(st.session_state.scenario_scores) if st.session_state.scenario_scores else 0
    st.sidebar.write(f"Scenario Score: {scenario_avg:.0%}")
    topstep_pass, _ = evaluate_topstep_metrics(st.session_state.trade_simulator)
    st.sidebar.write(f"Topstep Ready: {'✅' if topstep_pass else '❌'}")

    # Learning path
    if quiz_avg < 0.8 or scenario_avg < 0.8:
        st.sidebar.write("**Next Step**: Review weak lessons (e.g., ORB if scenarios low).")

    # Main content
    if st.session_state.module:
        st.header(st.session_state.module)
        lessons = COURSE_CONTENT[st.session_state.module]
        st.subheader("Lessons")
        
        for i, lesson in enumerate(lessons):
            lesson_key = f"{st.session_state.module}_{lesson['title']}"
            is_complete = progress.get(lesson_key, False)
            label = f"{lesson['title']} {'✅' if is_complete else ''}"
            if st.button(label, key=f"lesson_{i}"):
                st.session_state.lesson = lesson['title']
                st.session_state.annotations = []

        if st.session_state.lesson:
            lesson_data = next((l for l in lessons if l['title'] == st.session_state.lesson), None)
            if lesson_data:
                st.subheader(lesson_data['title'])
                st.markdown(lesson_data['content'])

                # Chart
                st.subheader("Interactive Chart")
                try:
                    chart_data = fetch_futures_data(symbol="NQ=F", period="1d", interval="5m")
                    chart_data = calculate_vwap(chart_data)
                    fig = plot_candlestick_chart(chart_data, f"NQ Futures ({lesson_data['title']})", st.session_state.annotations)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.error("Error loading chart.")

                # Annotations
                st.subheader("Annotate Chart")
                with st.form("annotation_form"):
                    ann_type = st.selectbox("Type", ["Demand", "Supply", "Fibonacci"])
                    if ann_type in ["Demand", "Supply"]:
                        y0 = st.number_input("Lower Price", min_value=0.0, value=14200.0)
                        y1 = st.number_input("Upper Price", min_value=0.0, value=14210.0)
                    else:
                        low = st.number_input("Swing Low", min_value=0.0, value=14360.0)
                        high = st.number_input("Swing High", min_value=0.0, value=14420.0)
                    submit = st.form_submit_button("Add")
                    if submit:
                        if ann_type in ["Demand", "Supply"]:
                            st.session_state.annotations.append({
                                'type': ann_type,
                                'y0': y0,
                                'y1': y1
                            })
                            # Feedback
                            if lesson_data['title'] == "Spotting Supply & Demand Zones" and ann_type == "Demand":
                                if 14200 <= y0 <= y1 <= 14210:
                                    st.success("Correct! Demand zone at 14200-14210.")
                                else:
                                    st.error("Incorrect. Demand zone should be 14200-14210.")
                        else:
                            diff = high - low
                            levels = [
                                {"label": "38.2", "price": high - diff * 0.382},
                                {"label": "50.0", "price": high - diff * 0.5},
                                {"label": "61.8", "price": high - diff * 0.618}
                            ]
                            st.session_state.annotations.append({
                                'type': "Fibonacci",
                                'levels': levels
                            })
                            if lesson_data['title'] == "Fibonacci Pullbacks":
                                if abs(high - 14420) < 10 and abs(low - 14360) < 10:
                                    st.success("Correct! Fibonacci levels align with 14360-14420 swing.")
                                else:
                                    st.error("Incorrect. Use swing low 14360, high 14420.")
                        st.experimental_rerun()
                
                if st.button("Clear Annotations"):
                    st.session_state.annotations = []
                    st.experimental_rerun()

                # Chart-based quiz
                if lesson_data['title'] in ["Spotting Supply & Demand Zones", "Fibonacci Pullbacks"]:
                    st.subheader("Chart Quiz")
                    with st.form("chart_quiz_form"):
                        st.write("Annotate the chart with a Demand zone (14200-14210) or Fibonacci (14360-14420), then submit.")
                        submit = st.form_submit_button("Check Annotation")
                        if submit:
                            correct = False
                            for ann in st.session_state.annotations:
                                if lesson_data['title'] == "Spotting Supply & Demand Zones" and ann['type'] == "Demand":
                                    if 14200 <= ann['y0'] <= ann['y1'] <= 14210:
                                        correct = True
                                if lesson_data['title'] == "Fibonacci Pullbacks" and ann['type'] == "Fibonacci":
                                    levels = {l['label']: l['price'] for l in ann['levels']}
                                    if abs(levels['61.8'] - (14420 - (14420-14360)*0.618)) < 10:
                                        correct = True
                            st.session_state.quiz_scores[f"{lesson_data['title']}_chart"] = 1 if correct else 0
                            if correct:
                                st.success("Correct! Annotation matches expected zone/levels.")
                            else:
                                st.error("Incorrect. Review zone or Fibonacci placement.")

                # Simulator
                st.subheader("Topstep Simulator")
                sim = st.session_state.trade_simulator
                st.write(f"**Balance**: ${sim['balance']:.2f}")
                st.write(f"**Equity**: ${sim['equity']:.2f}")
                if sim['positions']:
                    st.write("**Positions**:")
                    for pos in sim['positions']:
                        st.write(f"- {pos['type']} NQ at ${pos['entry_price']:.2f} (Size: {pos['size']})")
                
                topstep_pass, topstep_message = evaluate_topstep_metrics(sim)
                st.write(f"**Topstep Status**: {'✅ Pass' if topstep_pass else '❌ Fail'} - {topstep_message}")

                with st.form("trade_form"):
                    size = st.number_input("Size", min_value=1, max_value=TOPSTEP_RULES['max_position_size'], value=1)
                    stop_loss = st.number_input("Stop Loss (points)", min_value=0.0, value=15.0)
                    take_profit = st.number_input("Take Profit (points)", min_value=0.0, value=30.0)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        buy = st.form_submit_button("Buy NQ")
                    with col2:
                        sell = st.form_submit_button("Sell NQ")
                    with col3:
                        close = st.form_submit_button("Close Positions")
                    with col4:
                        reset = st.form_submit_button("Reset Combine")
                    
                    if buy:
                        latest_price = ticker_data.iloc[-1]['Close'] if 'ticker_data' in locals() else 14500
                        if len(sim['positions']) < TOPSTEP_RULES['max_position_size']:
                            sim['positions'].append({
                                'type': 'Long',
                                'entry_price': latest_price,
                                'size': size,
                                'stop_loss': latest_price - stop_loss,
                                'take_profit': latest_price + take_profit,
                                'entry_time': str(datetime.now())
                            })
                            sim['trades'].append({
                                'type': 'Buy',
                                'entry_price': latest_price,
                                'size': size,
                                'exit_price': latest_price,
                                'entry_time': str(datetime.now()),
                                'exit_time': str(datetime.now())
                            })
                            st.session_state.trade_simulator = sim
                            st.experimental_rerun()
                    if sell:
                        latest_price = ticker_data.iloc[-1]['Close'] if 'ticker_data' in locals() else 14500
                        if len(sim['positions']) < TOPSTEP_RULES['max_position_size']:
                            sim['positions'].append({
                                'type': 'Short',
                                'entry_price': latest_price,
                                'size': size,
                                'stop_loss': latest_price + stop_loss,
                                'take_profit': latest_price - take_profit,
                                'entry_time': str(datetime.now())
                            })
                            sim['trades'].append({
                                'type': 'Sell',
                                'entry_price': latest_price,
                                'size': size,
                                'exit_price': latest_price,
                                'entry_time': str(datetime.now()),
                                'exit_time': str(datetime.now())
                            })
                            st.session_state.trade_simulator = sim
                            st.experimental_rerun()
                    if close:
                        latest_price = ticker_data.iloc[-1]['Close'] if 'ticker_data' in locals() else 14500
                        for pos in sim['positions']:
                            for trade in sim['trades']:
                                if trade['entry_price'] == pos['entry_price'] and trade['type'] == ('Buy' if pos['type'] == 'Long' else 'Sell'):
                                    if latest_price <= pos['stop_loss'] or latest_price >= pos['take_profit'] if pos['type'] == 'Long' else latest_price >= pos['stop_loss'] or latest_price <= pos['take_profit']:
                                        trade['exit_price'] = pos['stop_loss'] if (latest_price <= pos['stop_loss'] if pos['type'] == 'Long' else latest_price >= pos['stop_loss']) else pos['take_profit']
                                    else:
                                        trade['exit_price'] = latest_price
                                    trade['exit_time'] = str(datetime.now())
                        sim['positions'] = []
                        sim['equity'] = sim['balance'] + sum(
                            (t['exit_price'] - t['entry_price'] if t['type'] == 'Buy' else t['entry_price'] - t['exit_price']) * t['size'] * 20
                            for t in sim['trades']
                        )
                        st.session_state.trade_simulator = sim
                        st.experimental_rerun()
                    if reset:
                        st.session_state.trade_simulator = {
                            'balance': 100000,
                            'positions': [],
                            'trades': [],
                            'equity': 100000
                        }
                        st.experimental_rerun()

                # Trading session
                st.subheader("Mock Trading Session")
                if st.button("Start Session"):
                    st.session_state.session_results = simulate_trading_session()
                if st.session_state.session_results:
                    for i, event in enumerate(st.session_state.session_results):
                        st.write(f"**Event {i+1}**: {event['description']}")
                        action = st.radio("Action:", ["Buy", "Sell", "Wait"], key=f"session_{i}")
                        if st.button("Submit Action", key=f"submit_session_{i}"):
                            correct = action == event['action']
                            st.session_state.session_results[i]['correct'] = correct
                            if correct:
                                st.success(f"Correct! {action} aligns with {event['setup']}.")
                            else:
                                st.error(f"Incorrect. {event['action']} was optimal for {event['setup']}.")
                    score = sum(1 for r in st.session_state.session_results if r.get('correct', False)) / len(st.session_state.session_results)
                    st.write(f"Session Score: {score:.0%}")

                # Scenarios
                if lesson_data.get('scenarios'):
                    st.subheader("Practice Scenarios")
                    for i, scenario in enumerate(lesson_data['scenarios']):
                        st.write(f"**Scenario {i+1}**: {scenario['description']}")
                        answers = []
                        for j, step in enumerate(scenario['steps']):
                            answer = st.radio(step['question'], step['options'], key=f"scenario_{i}_{j}_{lesson_data['title']}")
                            answers.append(answer)
                        if st.button("Submit Scenario", key=f"submit_scenario_{i}_{lesson_data['title']}"):
                            correct = all(a == s['answer'] for a, s in zip(answers, scenario['steps']))
                            st.session_state.scenario_scores[f"{lesson_data['title']}_{i}"] = 1 if correct else 0
                            if correct:
                                st.success(scenario['feedback'])
                                st.write("🎉 Scenario Master Badge")
                            else:
                                st.error(f"Incorrect. {scenario['feedback']}")

                # Quizzes
                if lesson_data['quiz']:
                    st.subheader("Quiz")
                    for i, q in enumerate(lesson_data['quiz']):
                        st.write(q['question'])
                        answer = st.radio("", q['options'], key=f"quiz_{i}_{lesson_data['title']}_{uuid4()}")
                        if st.button("Check", key=f"check_{i}_{lesson_data['title']}_{uuid4()}"):
                            st.session_state.quiz_scores[f"{lesson_data['title']}_{i}"] = 1 if answer == q['answer'] else 0
                            if answer == q['answer']:
                                st.success(q['feedback'])
                                st.write("🎉 Quiz Master Badge")
                            else:
                                st.error(f"Incorrect. {q['feedback']}")

                # Journal
                st.subheader("Trade Journal")
                with st.form(key=f"journal_form_{lesson_data['title']}"):
                    entry = st.text_area("Notes:")
                    setup = st.selectbox("Setup", ["ORB", "VWAP", "Stop Hunt", "Fibonacci", "Other"])
                    emotion = st.selectbox("Emotion", ["Confident", "Hesitant", "Impulsive"])
                    r_outcome = st.number_input("R Outcome", min_value=-10.0, max_value=10.0, value=0.0)
                    submit = st.form_submit_button("Log")
                    if submit and entry:
                        journal.append({
                            'date': str(datetime.now()),
                            'lesson': lesson_data['title'],
                            'entry': entry,
                            'setup': setup,
                            'emotion': emotion,
                            'r_outcome': r_outcome
                        })
                        save_journal(journal)
                        st.success("Entry saved!")
                
                st.write("**Journal Analysis**:")
                setups = {}
                emotions = {}
                wins = losses = 0
                for entry in journal:
                    if entry['lesson'] == lesson_data['title']:
                        setups[entry['setup']] = setups.get(entry['setup'], 0) + 1
                        emotions[entry['emotion']] = emotions.get(entry['emotion'], 0) + 1
                        if entry['r_outcome'] > 0:
                            wins += 1
                        elif entry['r_outcome'] < 0:
                            losses += 1
                st.write(f"Setups: {setups}")
                st.write(f"Emotions: {emotions}")
                win_rate = wins / (wins + losses) if wins + losses > 0 else 0
                st.write(f"Win Rate: {win_rate:.0%}")
                if win_rate < 0.5:
                    st.write("**Tip**: Review losing setups for patterns.")

                # Backtesting
                st.subheader("Backtest Setups")
                with st.form("backtest_form"):
                    setup_type = st.selectbox("Setup", ["VWAP Pullback", "ORB Breakout"])
                    stop_distance = st.number_input("Stop (points)", min_value=1.0, value=15.0)
                    target_distance = st.number_input("Target (points)", min_value=1.0, value=30.0)
                    submit = st.form_submit_button("Run")
                    if submit:
                        results = backtest_setup(chart_data, setup_type, {
                            "stop_distance": stop_distance,
                            "target_distance": target_distance
                        })
                        st.write(f"Trades: {results['trades']}")
                        st.write(f"Win Rate: {results['win_rate']:.0%}")
                        st.write(f"Average R: {results['avg_r']:.2f}")
                        if results['win_rate'] < 0.5:
                            st.write("**Tip**: Adjust stop/target or test more data.")

                # Mark complete
                lesson_key = f"{st.session_state.module}_{lesson_data['title']}"
                if st.button("Mark Complete", key=f"complete_{lesson_key}"):
                    progress[lesson_key] = True
                    save_progress(progress)
                    st.success("Lesson completed!")
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
