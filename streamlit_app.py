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

# Topstep Combine rules for $50K account (simplified)
TOPSTEP_RULES = {
    "daily_loss_limit": 2000,  # $2,000 max daily loss
    "max_drawdown": 3000,      # $3,000 trailing drawdown
    "profit_target": 3000,     # $3,000 profit goal
    "min_trading_days": 5,     # Minimum 5 days
    "max_position_size": 5     # Max 5 contracts
}

# Course structure with enhanced quizzes and scenarios
COURSE_CONTENT = {
    "Module 1: Market Structure & Price Behavior": [
        {
            "title": "Understanding Market Structure",
            "content": """
**Lesson 1: Understanding Market Structure (Higher Highs, Higher Lows / Lower Highs, Lower Lows)**

Market structure is the backbone of technical analysis. It helps identify trend direction and reversals.

**Key Definitions:**
- **Higher High (HH)**: A new price peak above the previous high (bullish)
- **Higher Low (HL)**: A pullback above the previous low (bullish)
- **Lower High (LH)**: A rally below the last high (bearish)
- **Lower Low (LL)**: A new low below the previous low (bearish)

**How to Read:**
1. Use a 5-min chart for intraday futures.
2. Identify 3-5 swing highs/lows.
3. Higher highs/lows = uptrend; lower highs/lows = downtrend.
4. Break of HL signals reversal.
            """,
            "quiz": [
                {
                    "question": "What indicates a bullish market structure?",
                    "options": ["Lower Highs and Lower Lows", "Higher Highs and Higher Lows", "Flat Highs and Lows"],
                    "answer": "Higher Highs and Higher Lows",
                    "feedback": "Correct! Higher Highs and Higher Lows confirm an uptrend (Lesson 1)."
                },
                {
                    "question": "On the chart below, price makes a new high at 14520, pulls back to 14470 (above prior low), and rallies to 14560. Is this bullish?",
                    "options": ["Yes", "No", "Unclear"],
                    "answer": "Yes",
                    "feedback": "Correct! This shows a Higher High (14520 to 14560) and Higher Low (14470), confirming a bullish structure (Lesson 1)."
                }
            ],
            "scenarios": [
                {
                    "description": "Price forms HH at 14520 and HL at 14470 on NQ futures. A pullback approaches 14470 with a bullish engulfing candle. What should you do?",
                    "options": ["Buy", "Sell", "Wait"],
                    "answer": "Buy",
                    "feedback": "Correct! A bullish engulfing at a Higher Low (14470) is a high-probability buy setup in an uptrend (Lesson 1)."
                }
            ]
        },
        {
            "title": "Spotting Supply & Demand Zones",
            "content": """
**Lesson 2: Spotting Supply & Demand Zones**

Supply and demand zones are where institutions drive price movement.

**What is a Zone?**
- **Demand Zone**: Buyers push price up (bottom of move).
- **Supply Zone**: Sellers push price down (top of rally).

**How to Identify:**
1. Find a tight consolidation (base).
2. Look for a strong move away (3+ candles).
3. Mark the highest (supply) or lowest (demand) candle.
4. Wait for retest with confirmation (e.g., engulfing).
            """,
            "quiz": [
                {
                    "question": "What confirms a demand zone?",
                    "options": ["Price breaking below", "Bullish engulfing at zone", "Price above VWAP"],
                    "answer": "Bullish engulfing at zone",
                    "feedback": "Correct! A bullish engulfing candle at the demand zone confirms buyer strength (Lesson 2)."
                },
                {
                    "question": "Price consolidates at 14200-14210, breaks to 14280, then returns to 14212 with a bullish engulfing. Is this a demand zone setup?",
                    "options": ["Yes", "No", "Unclear"],
                    "answer": "Yes",
                    "feedback": "Correct! Consolidation followed by a strong move and retest with engulfing confirms a demand zone (Lesson 2)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ consolidates at 14200-14210 for 15 minutes, breaks to 14280, and pulls back to 14212. The chart shows a bullish engulfing. Mark the demand zone and decide.",
                    "options": ["Buy at 14215", "Sell at 14215", "Wait"],
                    "answer": "Buy at 14215",
                    "feedback": "Correct! Buy at 14215 after the bullish engulfing in the demand zone, targeting 14280 (Lesson 2)."
                }
            ]
        }
    ],
    "Module 2: Tools That Actually Work": [
        {
            "title": "VWAP - The Intraday Anchor",
            "content": """
**Lesson 4: VWAP - The Intraday Anchor**

VWAP (Volume Weighted Average Price) gauges fair value.

**How It Works:**
- Averages price by volume from open.
- Above VWAP = bullish; below = bearish.

**Trading Styles:**
1. **Trend Continuation**: Buy pullbacks to VWAP.
2. **Mean Reversion**: Fade extended moves to VWAP.
            """,
            "quiz": [
                {
                    "question": "What does price above VWAP indicate?",
                    "options": ["Bearish bias", "Bullish bias", "Neutral bias"],
                    "answer": "Bullish bias",
                    "feedback": "Correct! Price above VWAP signals bullish momentum (Lesson 4)."
                },
                {
                    "question": "Price pulls back to VWAP at 14452, forms a bullish engulfing, and holds above VWAP. Is this a buy setup?",
                    "options": ["Yes", "No", "Unclear"],
                    "answer": "Yes",
                    "feedback": "Correct! A bullish engulfing at VWAP in an uptrend is a trend continuation setup (Lesson 4)."
                }
            ],
            "scenarios": [
                {
                    "description": "NQ breaks above VWAP at 14450, pulls back to 14452, and forms a bullish engulfing. What’s the trade?",
                    "options": ["Buy at 14454", "Sell at 14454", "Wait"],
                    "answer": "Buy at 14454",
                    "feedback": "Correct! Buy at 14454 after VWAP pullback confirmation, targeting 14500 (Lesson 4)."
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
            fig.add_hrect(
                y0=ann['y0'], y1=ann['y1'],
                fillcolor="green" if ann['type'] == "Demand" else "red",
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=ann['type'],
                annotation_position="top left"
            )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    return fig

def evaluate_topstep_metrics(sim):
    # Calculate daily P&L and drawdown
    daily_pnl = 0
    max_equity = sim['balance']
    current_equity = sim['balance']
    trading_days = set()
    
    for trade in sim['trades']:
        price_change = trade['exit_price'] - trade['entry_price'] if trade['type'] == 'Buy' else trade['entry_price'] - trade['exit_price']
        trade_pnl = price_change * trade['size'] * 20  # NQ point value ~$20
        daily_pnl += trade_pnl
        current_equity += trade_pnl
        max_equity = max(max_equity, current_equity)
        trading_days.add(trade['exit_time'].split('T')[0])
        
        # Check daily loss limit
        if daily_pnl < -TOPSTEP_RULES['daily_loss_limit']:
            return False, "Violated daily loss limit ($2,000)"
        
        # Check max drawdown
        drawdown = max_equity - current_equity
        if drawdown > TOPSTEP_RULES['max_drawdown']:
            return False, "Violated max drawdown ($3,000)"
    
    # Check profit target
    if current_equity - sim['balance'] < TOPSTEP_RULES['profit_target']:
        return False, f"Profit ${current_equity - sim['balance']:.2f} below target ($3,000)"
    
    # Check trading days
    if len(trading_days) < TOPSTEP_RULES['min_trading_days']:
        return False, f"Only {len(trading_days)} trading days (need 5)"
    
    return True, "Passed Topstep rules!"

def main():
    st.title("Pro Trader Futures Course Agent")
    st.write("Master futures trading with interactive scenarios and Topstep-ready simulations.")

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

    # Load user data
    progress = load_progress()
    journal = load_journal()

    # Sidebar navigation
    st.sidebar.header("Course Modules")
    modules = list(COURSE_CONTENT.keys())
    for mod in modules:
        if st.sidebar.button(mod, key=f"mod_{mod}"):
            st.session_state.module = mod
            st.session_state.lesson = None

    # Real-time ticker
    st.sidebar.header("Real-Time Ticker (NQ Futures)")
    try:
        ticker_data = fetch_futures_data(symbol="NQ=F", period="1d", interval="1m")
        latest = ticker_data.iloc[-1]
        st.sidebar.write(f"**Last Price**: ${latest['Close']:.2f}")
        st.sidebar.write(f"**Change**: {(latest['Close'] - latest['Open']):.2f} ({((latest['Close'] - latest['Open']) / latest['Open'] * 100):.2f}%)")
        st.sidebar.write(f"**High**: ${latest['High']:.2f} | **Low**: ${latest['Low']:.2f}")
    except Exception:
        st.sidebar.error("Error fetching ticker data. Using mock data.")
        st.sidebar.write("**Last Price**: $14500.00 (Mock)")
        st.sidebar.write("**Change**: +50.00 (+0.35%)")
        st.sidebar.write("**High**: $14550.00 | **Low**: $14450.00")

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
                st.session_state.annotations = []  # Reset annotations

        if st.session_state.lesson:
            lesson_data = next((l for l in lessons if l['title'] == st.session_state.lesson), None)
            if lesson_data:
                st.subheader(lesson_data['title'])
                st.markdown(lesson_data['content'])

                # Interactive chart with annotations
                st.subheader("Practice Chart")
                try:
                    chart_data = fetch_futures_data(symbol="NQ=F", period="1d", interval="5m")
                    chart_data = calculate_vwap(chart_data)
                    fig = plot_candlestick_chart(chart_data, f"NQ Futures ({lesson_data['title']})", st.session_state.annotations)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.error("Error loading chart. Using placeholder.")
                    st.write("*(Placeholder: Candlestick chart with VWAP)*")

                # Annotation controls
                st.subheader("Annotate Chart")
                with st.form("annotation_form"):
                    zone_type = st.selectbox("Zone Type", ["Demand", "Supply"])
                    y0 = st.number_input("Lower Price (y0)", min_value=0.0, value=14200.0)
                    y1 = st.number_input("Upper Price (y1)", min_value=0.0, value=14210.0)
                    submit = st.form_submit_button("Add Zone")
                    if submit:
                        st.session_state.annotations.append({
                            'type': zone_type,
                            'y0': y0,
                            'y1': y1
                        })
                        st.experimental_rerun()
                
                if st.button("Clear Annotations"):
                    st.session_state.annotations = []
                    st.experimental_rerun()

                # Trading simulator with Topstep rules
                st.subheader("Trading Simulator (Topstep Rules)")
                sim = st.session_state.trade_simulator
                st.write(f"**Balance**: ${sim['balance']:.2f}")
                st.write(f"**Equity**: ${sim['equity']:.2f}")
                if sim['positions']:
                    st.write("**Open Positions**:")
                    for pos in sim['positions']:
                        st.write(f"- {pos['type']} NQ at ${pos['entry_price']:.2f} (Size: {pos['size']})")
                
                # Topstep metrics
                topstep_pass, topstep_message = evaluate_topstep_metrics(sim)
                st.write(f"**Topstep Status**: {'✅ Pass' if topstep_pass else '❌ Fail'} - {topstep_message}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    size = st.number_input("Size (Contracts)", min_value=1, max_value=TOPSTEP_RULES['max_position_size'], value=1, key="size")
                    if st.button("Buy NQ", key=f"buy_{lesson_data['title']}"):
                        latest_price = ticker_data.iloc[-1]['Close'] if 'ticker_data' in locals() else 14500
                        if len(sim['positions']) < TOPSTEP_RULES['max_position_size']:
                            sim['positions'].append({
                                'type': 'Long',
                                'entry_price': latest_price,
                                'size': size,
                                'entry_time': str(datetime.now())
                            })
                            sim['trades'].append({
                                'type': 'Buy',
                                'entry_price': latest_price,
                                'size': size,
                                'exit_price': latest_price,  # Placeholder
                                'entry_time': str(datetime.now()),
                                'exit_time': str(datetime.now())
                            })
                            sim['equity'] += 0  # Update later with P&L
                            st.session_state.trade_simulator = sim
                            st.experimental_rerun()
                with col2:
                    if st.button("Sell NQ", key=f"sell_{lesson_data['title']}"):
                        latest_price = ticker_data.iloc[-1]['Close'] if 'ticker_data' in locals() else 14500
                        if len(sim['positions']) < TOPSTEP_RULES['max_position_size']:
                            sim['positions'].append({
                                'type': 'Short',
                                'entry_price': latest_price,
                                'size': size,
                                'entry_time': str(datetime.now())
                            })
                            sim['trades'].append({
                                'type': 'Sell',
                                'entry_price': latest_price,
                                'size': size,
                                'exit_price': latest_price,  # Placeholder
                                'entry_time': str(datetime.now()),
                                'exit_time': str(datetime.now())
                            })
                            sim['equity'] += 0  # Update later with P&L
                            st.session_state.trade_simulator = sim
                            st.experimental_rerun()
                with col3:
                    if st.button("Close All Positions", key=f"close_{lesson_data['title']}"):
                        latest_price = ticker_data.iloc[-1]['Close'] if 'ticker_data' in locals() else 14500
                        for pos in sim['positions']:
                            for trade in sim['trades']:
                                if trade['entry_price'] == pos['entry_price'] and trade['type'] == ('Buy' if pos['type'] == 'Long' else 'Sell'):
                                    trade['exit_price'] = latest_price
                                    trade['exit_time'] = str(datetime.now())
                        sim['positions'] = []
                        st.session_state.trade_simulator = sim
                        st.experimental_rerun()

                # Scenarios
                if lesson_data.get('scenarios'):
                    st.subheader("Practice Scenarios")
                    for i, scenario in enumerate(lesson_data['scenarios']):
                        st.write(f"**Scenario {i+1}**: {scenario['description']}")
                        choice = st.radio("Your Decision:", scenario['options'], key=f"scenario_{i}_{lesson_data['title']}")
                        if st.button("Submit Decision", key=f"submit_scenario_{i}_{lesson_data['title']}"):
                            if choice == scenario['answer']:
                                st.success(scenario['feedback'])
                                st.write("🎉 Earned: Scenario Master Badge")
                            else:
                                st.error(f"Incorrect. {scenario['feedback']}")

                # Enhanced quizzes
                if lesson_data['quiz']:
                    st.subheader("Enhanced Quiz")
                    for q in lesson_data['quiz']:
                        st.write(q['question'])
                        answer = st.radio("", q['options'], key=f"quiz_{q['question']}_{uuid4()}")
                        if st.button("Check Answer", key=f"check_{q['question']}_{uuid4()}"):
                            if answer == q['answer']:
                                st.success(q['feedback'])
                                st.write("🎉 Earned: Quiz Master Badge")
                            else:
                                st.error(f"Incorrect. {q['feedback']}")

                # Journaling
                st.subheader("Trade Journal")
                with st.form(key=f"journal_form_{lesson_data['title']}"):
                    journal_entry = st.text_area("Log your trade or notes:")
                    submit = st.form_submit_button("Add to Journal")
                    if submit and journal_entry:
                        journal.append({
                            'date': str(datetime.now()),
                            'lesson': lesson_data['title'],
                            'entry': journal_entry,
                            'setup': st.session_state.lesson,
                            'emotion': 'N/A'
                        })
                        save_journal(journal)
                        st.success("Journal entry saved!")
                
                st.write("**Recent Journal Entries**:")
                for entry in journal[-3:]:
                    if entry['lesson'] == lesson_data['title']:
                        st.write(f"- {entry['date']}: {entry['entry']} (Setup: {entry['setup']})")

                # Mark lesson as complete
                lesson_key = f"{st.session_state.module}_{lesson_data['title']}"
                if st.button("Mark as Complete", key=f"complete_{lesson_key}"):
                    progress[lesson_key] = True
                    save_progress(progress)
                    st.success("Lesson marked as complete!")
                    st.experimental_rerun()

    else:
        st.write("Select a module from the sidebar to begin.")

    # Progress overview
    st.sidebar.header("Your Progress")
    total_lessons = sum(len(lessons) for lessons in COURSE_CONTENT.values())
    completed = sum(1 for k, v in progress.items() if v)
    st.sidebar.progress(completed / total_lessons)
    st.sidebar.write(f"Completed {completed}/{total_lessons} lessons")

if __name__ == "__main__":
    main()
