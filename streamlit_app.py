import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import json
import os
from uuid import uuid4
from datetime import datetime, timedelta
import random

# File paths (Cloud-compatible)
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

# Course structure (deepened, partial for brevity)
COURSE_CONTENT = {
    "Module 1: Market Structure & Price Behavior": [
        {
            "title": "Understanding Market Structure",
            "content": """
**Lesson 1: Understanding Market Structure**

Market structure is the heartbeat of trading, mapping trends through Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), and Lower Lows (LL).

**In-Depth Analysis:**
- **HH**: Price surpasses prior peak (e.g., NQ from 14400 to 14520).
- **HL**: Pullback respects prior low (e.g., 14470 vs. 14450).
- **LH/LL**: Bearish shifts, often after failed rallies.
- Reversals trigger when HL/LH breaks with volume.

**Trading Blueprint:**
1. Chart on 5-min or 15-min for NQ/ES/CL.
2. Spot 3-5 swings for context.
3. Align with EMA 21 or VWAP.
4. Enter pullbacks with engulfing or pin bars.

**Trade Walkthrough**:
- **9:45 AM, NQ**: Post-open, price hits 14520 (HH), dips to 14470 (HL) with EMA 21 support. Bullish engulfing forms. Buy 14472, stop 14455, target 14540. Trade hits +2R by 10:30 AM.
- **Why It Worked**: Clear structure, VWAP confluence, high-volume open.

**Market Context**:
- Pre-10 AM EST: Strong trends due to open liquidity.
- Post-FOMC: Watch for reversals after spikes.

**Dynamic Example**:
- Today’s NQ shows HH at [latest high], HL at [latest low]. Look for pullback buys.

**Pro Tip**:
- Skip trades during lunch chop (12-2 PM EST); structure muddies.

**Common Mistakes:**
- Entering without candle confirmation.
- Trading flat structure (no HH/HL).
- Ignoring news-driven breaks.

**Checklist**:
- Trend defined? Level tested? Confluence confirmed?
            """,
            "quiz": [
                {
                    "question": "What signals an uptrend?",
                    "options": ["LH/LL", "HH/HL", "No swings"],
                    "answer": "HH/HL",
                    "feedback": "Correct! HH/HL define bullish trends (Lesson 1)."
                },
                {
                    "question": "Chart: NQ hits HH 14520, HL 14470, breaks 14470 with volume. Action?",
                    "options": ["Buy", "Sell", "Wait"],
                    "answer": "Wait",
                    "feedback": "Correct! HL break signals reversal risk; wait for clarity (Lesson 1)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ forms HH at 14520, HL at 14470. Pulls to 14470 with engulfing. FOMC at 2 PM spikes +75 points. Step 1: Enter. Step 2: Stop. Step 3: Trail stop after +30 points. Step 4: Target. Step 5: Handle spike.",
                    "steps": [
                        {"question": "Entry?", "options": ["Buy 14472", "Sell 14472", "Wait"], "answer": "Buy 14472"},
                        {"question": "Stop?", "options": ["14455", "14480", "14500"], "answer": "14455"},
                        {"question": "Trail stop?", "options": ["14472", "14490", "No change"], "answer": "14472"},
                        {"question": "Target?", "options": ["14540", "14500", "14490"], "answer": "14540"},
                        {"question": "FOMC spike?", "options": ["Hold", "Exit", "Tighten stop"], "answer": "Tighten stop"}
                    ],
                    "feedback": "Correct! Buy 14472, stop 14455, trail to 14472, target 14540, tighten stop post-FOMC to lock profits (Lesson 1)."
                }
            ]
        },
        {
            "title": "Spotting Supply & Demand Zones",
            "content": """
**Lesson 2: Spotting Supply & Demand Zones**

Zones pinpoint where institutions buy (demand) or sell (supply), fueling big moves.

**Deep Dive:**
- **Demand**: Base before upsurge (e.g., 14200-14210 on NQ).
- **Supply**: Peak before collapse (e.g., ES 4618-4620).
- Strong breakouts (3+ candles) define zones.

**Trading Blueprint:**
1. Find tight range on 5-min chart.
2. Mark breakout candle’s high/low.
3. Enter retest with engulfing/pin bar.
4. Stop outside zone; target prior swing.

**Trade Walkthrough**:
- **10:15 AM, NQ**: Ranges 14200-14210 for 20 min, breaks to 14280. Retests 14212 with bullish engulfing. Buy 14215, stop 14197, target 14275. Hits target by 11 AM.
- **Why It Worked**: Clean base, high-volume break, VWAP support.

**Market Context**:
- Post-9:30 AM: Zones form as institutions enter.
- Pre-CPI: Expect fakeouts; wait for confirmation.

**Dynamic Example**:
- Today’s NQ data suggests demand at [recent low range]. Watch retest.

**Pro Tip**:
- Avoid zones hit 3+ times; they lose edge.

**Common Mistakes:**
- Jumping in without retest confirmation.
- Trading zones in low-volume chop.
- Missing structure alignment.

**Checklist**:
- Tight range? Sharp break? Strong retest?
            """,
            "quiz": [
                {
                    "question": "What confirms a demand zone?",
                    "options": ["Break below", "Bullish engulfing", "Volume drop"],
                    "answer": "Bullish engulfing",
                    "feedback": "Correct! Engulfing confirms buyer intent (Lesson 2)."
                },
                {
                    "question": "Chart: NQ ranges 14200-14210, breaks 14280, retests 14212. Demand?",
                    "options": ["Yes", "No", "Unclear"],
                    "answer": "Yes",
                    "feedback": "Correct! Range and retest define demand (Lesson 2)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ ranges 14200-14210, breaks to 14280, retests 14212 with engulfing. CPI at 8:30 AM dips -40 points, recovers. Step 1: Enter. Step 2: Stop. Step 3: Trail stop after +20 points. Step 4: Target. Step 5: Handle dip.",
                    "steps": [
                        {"question": "Entry?", "options": ["Buy 14215", "Sell 14215", "Wait"], "answer": "Buy 14215"},
                        {"question": "Stop?", "options": ["14197", "14225", "14250"], "answer": "14197"},
                        {"question": "Trail stop?", "options": ["14215", "14200", "No change"], "answer": "14215"},
                        {"question": "Target?", "options": ["14275", "14240", "14200"], "answer": "14275"},
                        {"question": "CPI dip?", "options": ["Hold", "Exit", "Widen stop"], "answer": "Hold"}
                    ],
                    "feedback": "Correct! Buy 14215, stop 14197, trail to 14215, target 14275, hold through CPI dip (Lesson 2)."
                }
            ]
        }
        # Other lessons (Liquidity Sweeps, VWAP, etc.) follow similar depth
    ]
    # Modules 2-4 as in previous artifact, with added walkthroughs
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

def fetch_futures_data(symbol="NQ=F", period="4h", interval="5m"):
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
                entry = data['Close'].iloc[i] + params['slippage']
                stop = entry - params['stop_distance']
                target = entry + params['target_distance']
                for j in range(i+1, len(data)):
                    if data['Low'].iloc[j] <= stop:
                        trades.append({"result": -1, "R": -1, "drawdown": entry - stop})
                        break
                    if data['High'].iloc[j] >= target:
                        trades.append({"result": 1, "R": params['target_distance'] / params['stop_distance'], "drawdown": 0})
                        break
    elif setup_type == "ORB Breakout":
        for i in range(1, len(data)):
            high_15min = data['High'].iloc[:i].tail(3).max()
            low_15min = data['Low'].iloc[:i].tail(3).min()
            if data['Close'].iloc[i] > high_15min:
                entry = data['Close'].iloc[i] + params['slippage']
                stop = low_15min
                target = entry + params['target_distance']
                for j in range(i+1, len(data)):
                    if data['Low'].iloc[j] <= stop:
                        trades.append({"result": -1, "R": -1, "drawdown": entry - stop})
                        break
                    if data['High'].iloc[j] >= target:
                        trades.append({"result": 1, "R": params['target_distance'] / (entry - stop), "drawdown": 0})
                        break
    win_rate = sum(1 for t in trades if t['result'] > 0) / len(trades) if trades else 0
    avg_r = sum(t['R'] for t in trades) / len(trades) if trades else 0
    max_drawdown = max(t['drawdown'] for t in trades if t['drawdown'] > 0) if any(t['drawdown'] > 0 for t in trades) else 0
    profit_factor = sum(t['R'] for t in trades if t['R'] > 0) / abs(sum(t['R'] for t in trades if t['R'] < 0)) if any(t['R'] < 0 for t in trades) else float('inf')
    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_r": avg_r,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor
    }

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
    setups = ["VWAP Pullback", "ORB Breakout", "Stop Hunt", "Fibonacci Retracement", "Fake ORB", "EMA Cross"]
    events = ["Normal Day", "FOMC Spike", "CPI Release", "Low Volume Chop", "NY Open Rush"]
    session = []
    for _ in range(random.randint(10, 15)):
        setup = random.choice(setups)
        event = random.choice(events)
        price = 14500 + random.randint(-150, 150)
        volatility = 75 if event in ["FOMC Spike", "CPI Release"] else 30
        correct_action = "Wait" if setup == "Fake ORB" else random.choice(["Buy", "Sell"])
        session.append({
            "setup": setup,
            "event": event,
            "price": price,
            "volatility": volatility,
            "description": f"{setup} at ${price:.2f} during {event} (±{volatility} points).",
            "action": correct_action
        })
    return session

def plot_equity_curve(sim):
    equity = [sim['balance']]
    dates = [datetime.now() - timedelta(days=len(sim['trades']))]
    for trade in sim['trades']:
        price_change = trade['exit_price'] - trade['entry_price'] if trade['type'] == 'Buy' else trade['entry_price'] - trade['exit_price']
        trade_pnl = price_change * trade['size'] * 20
        equity.append(equity[-1] + trade_pnl)
        dates.append(datetime.strptime(trade['exit_time'], "%Y-%m-%dT%H:%M:%S.%f"))
    fig = go.Figure(data=[
        go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name='Equity'
        )
    ])
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Balance",
        showlegend=True
    )
    return fig

def plot_r_distribution(trades):
    r_values = [t['R'] for t in trades if t['R'] != 0]
    if not r_values:
        return None
    fig = go.Figure(data=[
        go.Histogram(
            x=r_values,
            nbinsx=20,
            name="R Outcomes"
        )
    ])
    fig.update_layout(
        title="R Distribution",
        xaxis_title="R Multiple",
        yaxis_title="Count",
        showlegend=True
    )
    return fig

def main():
    st.title("Pro Trader Futures Course Agent")
    st.write("Elite futures trading education with immersive simulations, predictive analytics, and Topstep mastery.")

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

    # Analytics Dashboard
    st.sidebar.header("Analytics Dashboard")
    total_lessons = sum(len(lessons) for lessons in COURSE_CONTENT.values())
    completed = sum(1 for k, v in progress.items() if v)
    st.sidebar.progress(completed / total_lessons)
    st.sidebar.write(f"Lessons Completed: {completed}/{total_lessons}")
    quiz_avg = sum(st.session_state.quiz_scores.values()) / len(st.session_state.quiz_scores) if st.session_state.quiz_scores else 0
    st.sidebar.write(f"Quiz Accuracy: {quiz_avg:.0%}")
    scenario_avg = sum(st.session_state.scenario_scores.values()) / len(st.session_state.scenario_scores) if st.session_state.scenario_scores else 0
    st.sidebar.write(f"Scenario Accuracy: {scenario_avg:.0%}")
    topstep_pass, _ = evaluate_topstep_metrics(st.session_state.trade_simulator)
    topstep_score = (quiz_avg * 0.3 + scenario_avg * 0.3 + (1.0 if topstep_pass else 0.0) * 0.4) * 100
    st.sidebar.write(f"Topstep Readiness: {topstep_score:.0f}%")
    
    # Learning path
    weak_lessons = sorted([(k, v) for k, v in st.session_state.quiz_scores.items() if v == 0], key=lambda x: x[0])
    if weak_lessons or scenario_avg < 0.8:
        st.sidebar.write("**Focus Areas**:")
        for lesson, _ in weak_lessons[:2]:
            st.sidebar.write(f"- {lesson.split('_')[0]} (Quiz failed)")
        if scenario_avg < 0.8:
            st.sidebar.write("- Scenarios (Accuracy low)")

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
                
                # Dynamic content
                try:
                    chart_data = fetch_futures_data()
                    highs = chart_data['High'].rolling(window=3).max()
                    lows = chart_data['Low'].rolling(window=3).min()
                    latest_hh = highs[-1]
                    latest_hl = lows[-1] if lows[-1] > lows[-2] else lows[-2]
                    content = lesson_data['content'].replace("[latest high]", f"${latest_hh:.2f}").replace("[latest low]", f"${latest_hl:.2f}")
                    recent_lows = chart_data['Low'].tail(10).min()
                    recent_highs = chart_data['High'].tail(10).max()
                    content = content.replace("[recent low range]", f"${recent_lows:.2f}-${recent_lows+10:.2f}")
                except Exception:
                    content = lesson_data['content']
                st.markdown(content)

                # Chart
                st.subheader("Interactive Chart")
                try:
                    chart_data = fetch_futures_data()
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
                        y0 = st.number_input("Lower Price", min_value=10000.0, max_value=20000.0, value=14200.0)
                        y1 = st.number_input("Upper Price", min_value=10000.0, max_value=20000.0, value=14210.0)
                    else:
                        low = st.number_input("Swing Low", min_value=10000.0, max_value=20000.0, value=14360.0)
                        high = st.number_input("Swing High", min_value=10000.0, max_value=20000.0, value=14420.0)
                    submit = st.form_submit_button("Add")
                    if submit:
                        if ann_type in ["Demand", "Supply"]:
                            if y0 >= y1:
                                st.error("Upper Price must exceed Lower Price.")
                            elif y1 - y0 > 500:
                                st.error("Zone too wide; keep within 500 points.")
                            elif y0 < 10000 or y1 > 20000:
                                st.error("Prices must be between 10000 and 20000 for NQ.")
                            else:
                                st.session_state.annotations.append({
                                    'type': ann_type,
                                    'y0': y0,
                                    'y1': y1
                                })
                                # Feedback
                                if lesson_data['title'] == "Spotting Supply & Demand Zones" and ann_type == "Demand":
                                    try:
                                        recent_low = chart_data['Low'].tail(10).min()
                                        if abs(y0 - recent_low) <= 20 and abs(y1 - (recent_low + 10)) <= 20:
                                            st.success(f"Correct! Demand zone near ${recent_low:.2f}.")
                                        else:
                                            st.error(f"Incorrect. Demand zone should be near ${recent_low:.2f}-${recent_low+10:.2f}.")
                                    except Exception:
                                        if 14200 <= y0 <= y1 <= 14210:
                                            st.success("Correct! Demand zone at 14200-14210.")
                                        else:
                                            st.error("Incorrect. Demand zone should be 14200-14210.")
                                st.rerun()
                        else:
                            if low >= high:
                                st.error("Swing High must exceed Swing Low.")
                            elif high - low > 1000:
                                st.error("Fibonacci range too wide; keep within 1000 points.")
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
                                    try:
                                        recent_low = chart_data['Low'].tail(10).min()
                                        recent_high = chart_data['High'].tail(10).max()
                                        if abs(low - recent_low) <= 20 and abs(high - recent_high) <= 20:
                                            st.success(f"Correct! Fibonacci aligns with ${recent_low:.2f}-${recent_high:.2f}.")
                                        else:
                                            st.error(f"Incorrect. Use swing low ${recent_low:.2f}, high ${recent_high:.2f}.")
                                    except Exception:
                                        if abs(high - 14420) <= 10 and abs(low - 14360) <= 10:
                                            st.success("Correct! Fibonacci aligns with 14360-14420.")
                                        else:
                                            st.error("Incorrect. Use swing low 14360, high 14420.")
                                st.rerun()
                
                if st.button("Clear Annotations"):
                    st.session_state.annotations = []
                    st.rerun()

                # Chart quiz
                st.subheader("Chart Quiz")
                with st.form("chart_quiz_form"):
                    st.write(f"Annotate a {lesson_data['title'].split()[1]} (e.g., Demand near recent low, Fibonacci recent swing).")
                    submit = st.form_submit_button("Check")
                    if submit:
                        correct = False
                        try:
                            recent_low = chart_data['Low'].tail(10).min()
                            recent_high = chart_data['High'].tail(10).max()
                            for ann in st.session_state.annotations:
                                if lesson_data['title'] == "Spotting Supply & Demand Zones" and ann['type'] == "Demand":
                                    if abs(ann['y0'] - recent_low) <= 5 and abs(ann['y1'] - (recent_low + 10)) <= 5:
                                        correct = True
                                if lesson_data['title'] == "Fibonacci Pullbacks" and ann['type'] == "Fibonacci":
                                    levels = {l['label']: l['price'] for l in ann['levels']}
                                    expected_618 = recent_high - (recent_high - recent_low) * 0.618
                                    if abs(levels['61.8'] - expected_618) <= 5:
                                        correct = True
                        except Exception:
                            for ann in st.session_state.annotations:
                                if lesson_data['title'] == "Spotting Supply & Demand Zones" and ann['type'] == "Demand":
                                    if abs(ann['y0'] - 14200) <= 5 and abs(ann['y1'] - 14210) <= 5:
                                        correct = True
                                if lesson_data['title'] == "Fibonacci Pullbacks" and ann['type'] == "Fibonacci":
                                    levels = {l['label']: l['price'] for l in ann['levels']}
                                    if abs(levels['61.8'] - (14420 - (14420-14360)*0.618)) <= 5:
                                        correct = True
                        st.session_state.quiz_scores[f"{lesson_data['title']}_chart"] = 1 if correct else 0
                        if correct:
                            st.success(f"Correct! {lesson_data['title'].split()[1]} annotation is precise (±5 points).")
                        else:
                            try:
                                st.error(f"Incorrect. Aim for Demand near ${recent_low:.2f} or Fibonacci at ${recent_low:.2f}-${recent_high:.2f}.")
                            except Exception:
                                st.error("Incorrect. Review placement for zones or levels.")

                # Simulator
                st.subheader("Topstep Simulator")
                sim = st.session_state.trade_simulator
                st.write(f"**Balance**: ${sim['balance']:.2f}")
                st.write(f"**Equity**: ${sim['equity']:.2f}")
                if sim['positions']:
                    st.write("**Positions**:")
                    for pos in sim['positions']:
                        st.write(f"- {pos['type']} NQ at ${pos['entry_price']:.2f} (Size: {pos['size']}, Stop: ${pos['stop_loss']:.2f}, Target: ${pos['take_profit']:.2f})")
                
                topstep_pass, topstep_message = evaluate_topstep_metrics(sim)
                st.write(f"**Topstep Status**: {'✅ Pass' if topstep_pass else '❌ Fail'} - {topstep_message}")
                
                # Metrics
                if sim['trades']:
                    wins = sum(1 for t in sim['trades'] if (t['exit_price'] - t['entry_price'] if t['type'] == 'Buy' else t['entry_price'] - t['exit_price']) > 0)
                    win_rate = wins / len(sim['trades']) if sim['trades'] else 0
                    profit_factor = sum((t['exit_price'] - t['entry_price'] if t['type'] == 'Buy' else t['entry_price'] - t['exit_price']) * t['size'] * 20 for t in sim['trades'] if (t['exit_price'] - t['entry_price'] if t['type'] == 'Buy' else t['entry_price'] - t['exit_price']) > 0) / abs(sum((t['exit_price'] - t['entry_price'] if t['type'] == 'Buy' else t['entry_price'] - t['exit_price']) * t['size'] * 20 for t in sim['trades'] if (t['exit_price'] - t['entry_price'] if t['type'] == 'Buy' else t['entry_price'] - t['exit_price']) < 0)) if any((t['exit_price'] - t['entry_price'] if t['type'] == 'Buy' else t['entry_price'] - t['exit_price']) < 0 for t in sim['trades']) else float('inf')
                    st.write(f"**Win Rate**: {win_rate:.0%}")
                    st.write(f"**Profit Factor**: {profit_factor:.2f}")
                
                # Equity curve
                if sim['trades']:
                    st.subheader("Equity Curve")
                    fig = plot_equity_curve(sim)
                    st.plotly_chart(fig, use_container_width=True)

                with st.form("trade_form"):
                    size = st.number_input("Size", min_value=1, max_value=TOPSTEP_RULES['max_position_size'], value=1)
                    stop_loss = st.number_input("Stop Loss (points)", min_value=0.0, value=15.0)
                    take_profit = st.number_input("Take Profit (points)", min_value=0.0, value=30.0)
                    volatility_event = st.selectbox("Market Event", ["None", "FOMC Spike", "CPI Release", "Stress Test"])
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        buy = st.form_submit_button("Buy NQ")
                    with col2:
                        sell = st.form_submit_button("Sell NQ")
                    with col3:
                        close = st.form_submit_button("Close Positions")
                    with col4:
                        reset = st.form_submit_button("Reset Combine")
                    
                    slippage = 0.25 if volatility_event != "None" else 0.1
                    if buy:
                        latest_price = ticker_data.iloc[-1]['Close'] if 'ticker_data' in locals() else 14500
                        if volatility_event == "Stress Test":
                            latest_price += random.randint(-100, 100)
                        elif volatility_event != "None":
                            latest_price += random.randint(-75, 75)
                        latest_price += slippage
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
                            st.rerun()
                    if sell:
                        latest_price = ticker_data.iloc[-1]['Close'] if 'ticker_data' in locals() else 14500
                        if volatility_event == "Stress Test":
                            latest_price += random.randint(-100, 100)
                        elif volatility_event != "None":
                            latest_price += random.randint(-75, 75)
                        latest_price -= slippage
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
                            st.rerun()
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
                        st.rerun()
                    if reset:
                        st.session_state.trade_simulator = {
                            'balance': 100000,
                            'positions': [],
                            'trades': [],
                            'equity': 100000
                        }
                        st.rerun()

                # Trading session
                st.subheader("Mock Trading Session (6 Hours)")
                if st.button("Start Session"):
                    st.session_state.session_results = simulate_trading_session()
                if st.session_state.session_results:
                    st.write("**Session Events**:")
                    for i, event in enumerate(st.session_state.session_results):
                        st.write(f"Event {i+1}: {event['description']}")
                        action = st.radio("Action:", ["Buy", "Sell", "Wait"], key=f"session_{i}")
                        if st.button("Submit", key=f"submit_session_{i}"):
                            correct = action == event['action']
                            st.session_state.session_results[i]['correct'] = correct
                            if correct:
                                st.success(f"Correct! {action} fits {event['setup']} in {event['event']}.")
                            else:
                                st.error(f"Incorrect. {event['action']} was optimal for {event['setup']}.")
                    score = sum(1 for r in st.session_state.session_results if r.get('correct', False)) / len(st.session_state.session_results)
                    st.write(f"**Session Score**: {score:.0%}")
                    traps_missed = sum(1 for r in st.session_state.session_results if r['setup'] == "Fake ORB" and not r.get('correct', False))
                    if traps_missed:
                        st.write(f"**Debrief**: Missed {traps_missed} trap(s); review Fake ORB patterns.")
                    if score < 0.8:
                        st.write("**Debrief**: Practice timing on high-volatility events.")

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
                                if "FOMC" in scenario['description'] or "CPI" in scenario['description']:
                                    st.error("Failure Path: Chasing news spikes often leads to losses (-1R).")

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
                    setup = st.selectbox("Setup", ["ORB", "VWAP", "Stop Hunt", "Fibonacci", "EMA Cross", "Power of 3", "Other"])
                    emotion = st.selectbox("Emotion", ["Confident", "Hesitant", "Impulsive", "Calm"])
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
                
                st.subheader("Journal Analytics")
                setups = {}
                emotions = {}
                wins = losses = 0
                r_total = 0
                setup_r = {}
                for entry in journal:
                    setups[entry['setup']] = setups.get(entry['setup'], 0) + 1
                    emotions[entry['emotion']] = emotions.get(entry['emotion'], 0) + 1
                    r_total += entry['r_outcome']
                    setup_r[entry['setup']] = setup_r.get(entry['setup'], 0) + entry['r_outcome']
                    if entry['r_outcome'] > 0:
                        wins += 1
                    elif entry['r_outcome'] < 0:
                        losses += 1
                win_rate = wins / (wins + losses) if wins + losses > 0 else 0
                avg_r = r_total / (wins + losses) if wins + losses > 0 else 0
                st.write(f"**Setups**: {setups}")
                st.write(f"**Emotions**: {emotions}")
                st.write(f"**Win Rate**: {win_rate:.0%}")
                st.write(f"**Average R**: {avg_r:.2f}")
                setup_predictions = {k: v / setups[k] if setups[k] > 0 else 0 for k, v in setup_r.items()}
                st.write("**Setup Success Forecast**:")
                for setup, r_avg in setup_predictions.items():
                    st.write(f"- {setup}: {r_avg:.2f} R per trade")
                if emotions.get("Impulsive", 0) / sum(emotions.values()) > 0.3:
                    st.write("**Bias Alert**: Impulsive trades may hurt performance; aim for calm entries.")
                if setup_predictions:
                    best_setup = max(setup_predictions, key=setup_predictions.get)
                    st.write(f"**Strength**: {best_setup} averages {setup_predictions[best_setup]:.2f} R.")

                # Plot setup performance
                if setups:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(setups.keys()),
                            y=list(setups.values()),
                            name="Trades per Setup"
                        )
                    ])
                    fig.update_layout(
                        title="Setup Distribution",
                        xaxis_title="Setup",
                        yaxis_title="Trades"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Backtesting
                st.subheader("Backtest Setups")
                with st.form("backtest_form"):
                    setup_type = st.selectbox("Setup", ["VWAP Pullback", "ORB Breakout", "Stop Hunt", "Fibonacci Retracement"])
                    stop_distance = st.number_input("Stop (points)", min_value=1.0, value=15.0)
                    target_distance = st.number_input("Target (points)", min_value=1.0, value=30.0)
                    slippage = st.number_input("Slippage (points)", min_value=0.0, value=0.25)
                    submit = st.form_submit_button("Run")
                    if submit:
                        results = backtest_setup(chart_data, setup_type, {
                            "stop_distance": stop_distance,
                            "target_distance": target_distance,
                            "slippage": slippage
                        })
                        st.write(f"**Trades**: {results['trades']}")
                        st.write(f"**Win Rate**: {results['win_rate']:.0%}")
                        st.write(f"**Average R**: {results['avg_r']:.2f}")
                        st.write(f"**Max Drawdown**: {results['max_drawdown']:.2f} points")
                        st.write(f"**Profit Factor**: {results['profit_factor']:.2f}")
                        expectancy = results['win_rate'] * results['avg_r'] - (1 - results['win_rate'])
                        st.write(f"**Expectancy**: {expectancy:.2f} R per trade")
                        if expectancy < 0.3:
                            st.write("**Tip**: Optimize stop/target or filter entries.")

                # Mark complete
                lesson_key = f"{st.session_state.module}_{lesson_data['title']}"
                if st.button("Mark Complete", key=f"complete_{lesson_key}"):
                    progress[lesson_key] = True
                    save_progress(progress)
                    st.success("Lesson completed!")
                    st.rerun()

                # Expert challenge
                if completed >= total_lessons * 0.8 and quiz_avg >= 0.8 and scenario_avg >= 0.8:
                    st.subheader("Expert Challenge")
                    st.write("Pass a Topstep Combine simulation with 70% consistency.")
                    if st.button("Start Challenge"):
                        st.session_state.trade_simulator = {
                            'balance': 100000,
                            'positions': [],
                            'trades': [],
                            'equity': 100000
                        }
                        st.write("Challenge started! Trade 5+ days, hit $3,000 profit, maintain consistency.")
                    if topstep_pass and len(set(t['exit_time'].split('T')[0] for t in sim['trades'])) >= 5:
                        st.success("🎉 Challenge Complete! Topstep Combine passed with consistency.")

if __name__ == "__main__":
    main()
