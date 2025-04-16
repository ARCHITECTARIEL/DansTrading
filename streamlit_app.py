import streamlit as st
import json
import os
from uuid import uuid4

# File to store user progress
PROGRESS_FILE = "user_progress.json"

# Course structure based on the provided document
COURSE_CONTENT = {
    "Module 1: Market Structure & Price Behavior": [
        {
            "title": "Understanding Market Structure",
            "content": """
**Lesson 1: Understanding Market Structure (Higher Highs, Higher Lows / Lower Highs, Lower Lows)**

Market structure is the backbone of technical analysis and trend trading. It tells you exactly what direction the market is moving and when that direction might be changing.

**Key Definitions:**
- **Higher High (HH)**: A new price peak above the previous high (bullish)
- **Higher Low (HL)**: A pullback that stays above the previous low (bullish)
- **Lower High (LH)**: A rally that doesn't exceed the last high (bearish)
- **Lower Low (LL)**: A new low below the previous low (bearish)

**How to Read Market Structure:**
1. Start on a clean chart (5-min or 15-min for intraday futures).
2. Zoom out to see 3-5 prior swing highs and lows.
3. Mark swings: Higher highs and lows indicate an uptrend; lower highs and lows indicate a downtrend.
4. Watch for a break in structure (e.g., price takes out a HL or LH) for potential reversals.

**Pro Tip**: Combine structure with EMA 21 or VWAP to stay aligned with the trend.
            """,
            "quiz": [
                {
                    "question": "What indicates a bullish market structure?",
                    "options": ["Lower Highs and Lower Lows", "Higher Highs and Higher Lows", "Flat Highs and Lows"],
                    "answer": "Higher Highs and Higher Lows"
                }
            ]
        },
        {
            "title": "Spotting Supply & Demand Zones",
            "content": """
**Lesson 2: Spotting Supply & Demand Zones**

Supply and demand zones are where institutions move the market with large orders.

**What is a Supply or Demand Zone?**
- **Demand Zone**: Area where buyers pushed price up (bottom of a move).
- **Supply Zone**: Area where sellers dominated, pushing price down (top of a rally).

**How to Identify a Zone:**
1. Look for a base (tight consolidation of candles).
2. Watch for a strong, fast move away (3+ candles).
3. Mark the highest (supply) or lowest (demand) candle in the base.
4. Wait for price to return and test the zone with confirmation (e.g., engulfing candle).

**Pro Tip**: Use a rectangle tool to mark the full body and wicks of the base zone.
            """,
            "quiz": [
                {
                    "question": "What confirms a demand zone?",
                    "options": ["Price breaking below the zone", "Bullish engulfing candle at the zone", "Price staying above VWAP"],
                    "answer": "Bullish engulfing candle at the zone"
                }
            ]
        },
        # Add more lessons as needed
    ],
    "Module 2: Tools That Actually Work": [
        {
            "title": "VWAP - The Intraday Anchor",
            "content": """
**Lesson 4: VWAP - The Intraday Anchor**

VWAP (Volume Weighted Average Price) is a tool used by institutions to judge fair value.

**How VWAP Works:**
- Calculates the average price weighted by volume from the open.
- Above VWAP = bullish bias; below VWAP = bearish bias.

**Why It's Powerful:**
- Not laggy like moving averages.
- Trusted by pros, funds, and algos.
- Acts as dynamic support/resistance.

**Trading Styles:**
1. **Trend Continuation**: Enter on a pullback to VWAP with a confirmation candle.
2. **Mean Reversion Fade**: Fade extended moves back to VWAP in choppy sessions.
            """,
            "quiz": [
                {
                    "question": "What does price above VWAP indicate?",
                    "options": ["Bearish bias", "Bullish bias", "Neutral bias"],
                    "answer": "Bullish bias"
                }
            ]
        },
        # Add more lessons (EMA, Fibonacci, etc.)
    ],
    # Add Module 3 and 4 as needed
}

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=4)

def main():
    st.title("Pro Trader Futures Course Agent")
    st.write("Learn to trade futures and pass the Topstep Combine with confidence.")

    # Initialize session state for navigation
    if 'module' not in st.session_state:
        st.session_state.module = None
    if 'lesson' not in st.session_state:
        st.session_state.lesson = None

    # Load user progress
    progress = load_progress()

    # Sidebar for module selection
    st.sidebar.header("Course Modules")
    modules = list(COURSE_CONTENT.keys())
    for mod in modules:
        if st.sidebar.button(mod, key=f"mod_{mod}"):
            st.session_state.module = mod
            st.session_state.lesson = None

    # Main content area
    if st.session_state.module:
        st.header(st.session_state.module)
        lessons = COURSE_CONTENT[st.session_state.module]
        st.subheader("Lessons")
        
        # Display lesson buttons for the selected module
        for i, lesson in enumerate(lessons):
            lesson_key = f"{st.session_state.module}_{lesson['title']}"
            is_complete = progress.get(lesson_key, False)
            label = f"{lesson['title']} {'✅' if is_complete else ''}"
            if st.button(label, key=f"lesson_{i}"):
                st.session_state.lesson = lesson['title']

        # Display selected lesson content
        if st.session_state.lesson:
            lesson_data = next((l for l in lessons if l['title'] == st.session_state.lesson), None)
            if lesson_data:
                st.subheader(lesson_data['title'])
                st.markdown(lesson_data['content'])

                # Quiz section
                if lesson_data['quiz']:
                    st.subheader("Quick Quiz")
                    for q in lesson_data['quiz']:
                        st.write(q['question'])
                        answer = st.radio("", q['options'], key=f"quiz_{q['question']}_{uuid4()}")
                        if st.button("Check Answer", key=f"check_{q['question']}_{uuid4()}"):
                            if answer == q['answer']:
                                st.success("Correct!")
                            else:
                                st.error(f"Incorrect. The correct answer is: {q['answer']}")

                # Mark lesson as complete
                lesson_key = f"{st.session_state.module}_{lesson_data['title']}"
                if st.button("Mark as Complete", key=f"complete_{lesson_key}"):
                    progress[lesson_key] = True
                    save_progress(progress)
                    st.success("Lesson marked as complete!")
                    st.experimental_rerun()

                # Placeholder for charting tool
                st.write("*(Future feature: Interactive chart for practicing market structure)*")

    else:
        st.write("Select a module from the sidebar to begin.")

    # Progress overview
    st.sidebar.header("Your Progress")
    total_lessons = sum(len(lessons) for lessons in COURSE_CONTENT.values())
    completed = sum(1 for k, v in progress.items() if v)
    st.sidebar.write(f"Completed {completed}/{total_lessons} lessons")

if __name__ == "__main__":
    main()
