import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import json
import os
from uuid import uuid4
from datetime import datetime

# File paths
PROGRESS_FILE = "user_progress.json"
JOURNAL_FILE = "trade_journal.json"

# Course structure (abridged for brevity, focusing on charting-relevant lessons)
COURSE_CONTENT = {
    "Module 1: Market Structure & Price Behavior": [
        {
            "title": "Understanding Market Structure",
            "content": """
**Lesson 1: Understanding Market Structure (Higher Highs, Higher Lows / Lower Highs, Lower Lows)**

Market structure is the backbone of technical analysis. It helps identify trend direction and potential reversals.

**Key Definitions:**
- **Higher High (HH)**: A new price peak above the previous high (bullish)
- **Higher Low (HL)**: A pullback above the previous low (bullish)
- **Lower High (LH)**: A rally below the last high (bearish)
- **Lower Low (LL)**: A new low below the previous low (bearish)

**How to Read Market Structure:**
1. Use a 5-min or 15-min chart for intraday futures.
2. Identify 3-5 swing highs and lows.
3. Higher highs/lows = uptrend; lower highs/lows = downtrend.
4. A break in structure (e.g., price takes out HL) signals a reversal.

**Practice**: Use the chart below to identify HH/HL or LH/LL patterns.
            """,
            "quiz": [
                {
                    "question": "What indicates a bullish market structure?",
                    "options": ["Lower Highs and Lower Lows", "Higher Highs and Higher Lows", "Flat Highs and Lows"],
                    "answer": "Higher Highs and Higher Lows"
                },
                {
                    "question": "What signals a potential reversal in an uptrend?",
                    "options": ["Price makes a new HH", "Price breaks below a HL", "Price stays above VWAP"],
                    "answer": "Price breaks below a HL"
                }
            ]
        },
        {
            "title": "Spotting Supply & Demand Zones",
            "content": """
**Lesson 2: Spotting Supply & Demand Zones**

Supply and demand zones are where institutions place large orders, driving price movement.

**What is a Supply or Demand Zone?**
- **Demand Zone**: Where buyers pushed price up (bottom of a move).
- **Supply Zone**: Where sellers pushed price down (top of a rally).

**How to Identify:**
1. Find a base (tight candle consolidation).
2. Look for a strong move away (3+ candles).
3. Mark the highest (supply) or lowest (demand) candle.
4. Wait for price to retest with confirmation (e.g., engulfing candle).

**Practice**: Use the chart tool to mark a demand zone on NQ futures.
            """,
            "quiz": [
                {
                    "question": "What confirms a demand zone?",
                    "options": ["Price breaking below the zone", "Bullish engulfing candle at the zone", "Price staying above VWAP"],
                    "answer": "Bullish engulfing candle at the zone"
                }
            ]
        }
    ],
    "Module 2: Tools That Actually Work": [
        {
            "title": "VWAP - The Intraday Anchor",
            "content": """
**Lesson 4: VWAP - The Intraday Anchor**

VWAP (Volume Weighted Average Price) is used by institutions to gauge fair value.

**How VWAP Works:**
- Averages price weighted by volume from the open.
- Above VWAP = bullish; below VWAP = bearish.

**Trading Styles:**
1. **Trend Continuation**: Buy pullbacks to VWAP with confirmation.
2. **Mean Reversion**: Fade extended moves back to VWAP.

**Practice**: Use the chart to identify a VWAP pullback setup.
            """,
            "quiz": [
                {
                    "question": "What does price above VWAP indicate?",
                    "options": ["Bearish bias", "Bullish bias", "Neutral bias"],
                    "answer": "Bullish bias"
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
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    return data

def calculate_vwap(df):
    # VWAP = (Cumulative (Price * Volume)) / Cumulative Volume
    df['Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['PriceVolume'] = df['Price'] * df['Volume']
    df['CumulativePriceVolume'] = df['PriceVolume'].cumsum()
    df['CumulativeVolume'] = df['Volume'].cumsum()
    df['VWAP'] = df['CumulativePriceVolume'] / df['CumulativeVolume']
    return df

def plot_candlestick_chart(data, title="NQ Futures Chart"):
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
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    return fig

def main():
    st.title("Pro Trader Futures Course Agent")
    st.write("Master futures trading and prepare for the Topstep Combine with interactive tools.")

    # Initialize session state
    if 'module' not in st.session_state:
        st.session_state.module = None
    if 'lesson' not in st.session_state:
        st.session_state.lesson = None
    if 'trade_simulator' not in st.session_state:
        st.session_state.trade_simulator = {'balance': 100000, 'positions': [], 'trades': []}

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
    except Exception as e:
        st.sidebar.error("Error fetching ticker data. Using mock data.")
        st.sidebar.write("**Last Price**: $14500.00 (Mock)")
        st.sidebar.write("**Change**: +50.00 (+0.35%)")
        st.sidebar.write("**High**: $14550.00 | **Low**: $14450.00")

    # Main content
    if st.session_state.module:
        st.header(st.session_state.module)
        lessons = COURSE_CONTENT[st.session_state.module]
        st.subheader("Lessons")
        
        # Lesson buttons
        for i, lesson in enumerate(lessons):
            lesson_key = f"{st.session_state.module}_{lesson['title']}"
            is_complete = progress.get(lesson_key, False)
            label = f"{lesson['title']} {'✅' if is_complete else ''}"
            if st.button(label, key=f"lesson_{i}"):
                st.session_state.lesson = lesson['title']

        # Display lesson content
        if st.session_state.lesson:
            lesson_data = next((l for l in lessons if l['title'] == st.session_state.lesson), None)
            if lesson_data:
                st.subheader(lesson_data['title'])
                st.markdown(lesson_data['content'])

                # Interactive chart
                st.subheader("Practice Chart")
                try:
                    chart_data = fetch_futures_data(symbol="NQ=F", period="1d", interval="5m")
                    chart_data = calculate_vwap(chart_data)
                    fig = plot_candlestick_chart(chart_data, f"NQ Futures with VWAP ({lesson_data['title']})")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error("Error loading chart. Using placeholder.")
                    st.write("*(Placeholder: Candlestick chart with VWAP and annotation tools)*")

                # Trading simulator
                st.subheader("Trading Simulator")
                sim = st.session_state.trade_simulator
                st.write(f"**Balance**: ${sim['balance']:.2f}")
                if sim['positions']:
                    st.write("**Open Positions**:")
                    for pos in sim['positions']:
                        st.write(f"- {pos['type']} NQ at ${pos['entry_price']:.2f} (Size: {pos['size']})")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Buy NQ", key=f"buy_{lesson_data['title']}"):
                        latest_price = ticker_data.iloc[-1]['Close'] if 'ticker_data' in locals() else 14500
                        sim['positions'].append({
                            'type': 'Long',
                            'entry_price': latest_price,
                            'size': 1,
                            'entry_time': str(datetime.now())
                        })
                        sim['trades'].append({
                            'type': 'Buy',
                            'price': latest_price,
                            'size': 1,
                            'time': str(datetime.now())
                        })
                        st.session_state.trade_simulator = sim
                        st.experimental_rerun()
                with col2:
                    if st.button("Sell NQ", key=f"sell_{lesson_data['title']}"):
                        latest_price = ticker_data.iloc[-1]['Close'] if 'ticker_data' in locals() else 14500
                        sim['positions'].append({
                            'type': 'Short',
                            'entry_price': latest_price,
                            'size': 1,
                            'entry_time': str(datetime.now())
                        })
                        sim['trades'].append({
                            'type': 'Sell',
                            'price': latest_price,
                            'size': 1,
                            'time': str(datetime.now())
                        })
                        st.session_state.trade_simulator = sim
                        st.experimental_rerun()

                # Quiz section
                if lesson_data['quiz']:
                    st.subheader("Quick Quiz")
                    for q in lesson_data['quiz']:
                        st.write(q['question'])
                        answer = st.radio("", q['options'], key=f"quiz_{q['question']}_{uuid4()}")
                        if st.button("Check Answer", key=f"check_{q['question']}_{uuid4()}"):
                            if answer == q['answer']:
                                st.success("Correct!")
                                # Award badge (mock)
                                st.write("🎉 Earned: Quiz Master Badge")
                            else:
                                st.error(f"Incorrect. The correct answer is: {q['answer']}")

                # Journaling
                st.subheader("Trade Journal")
                with st.form(key=f"journal_form_{lesson_data['title']}"):
                    journal_entry = st.text_area("Log your trade or notes (e.g., setup, emotions):")
                    submit = st.form_submit_button("Add to Journal")
                    if submit and journal_entry:
                        journal.append({
                            'date': str(datetime.now()),
                            'lesson': lesson_data['title'],
                            'entry': journal_entry,
                            'setup': st.session_state.lesson,
                            'emotion': 'N/A'  # Placeholder for future input
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

    # Placeholder for progress dashboard
    st.sidebar.header("Performance Dashboard")
    st.sidebar.write("*(Future feature: Visualize quiz scores, simulator performance, and Topstep readiness)*")

if __name__ == "__main__":
    main()
