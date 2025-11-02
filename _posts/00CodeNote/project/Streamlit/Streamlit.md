# Streamlit

## Streamlit overveiw

- open-source Python framework designed to turn data scripts into interactive web apps with minimal code — no frontend or web framework knowledge required.

- You write a normal Python script, sprinkle in some Streamlit commands (for charts, tables, widgets, text, etc.), then run it via: `streamlit run the_script.py`
- and a local server spins up; the app appears in the browser.

- As you change and save the code, Streamlit detects it and prompts you to rerun — giving a very quick feedback loop (code → app → modify → code → etc.).

- Because of this, Streamlit is especially good for:
- Data visualization and exploration (charts, tables, maps).
- Interactive dashboards or data apps (with filters, user inputs, controls).
- Rapid prototyping / minimum viable products (MVPs) for ML / data-driven tools.

---

## Main Programming Model & Concepts

- The re-run model / data flow
- Every time you launch the app or a user interacts with a widget (slider, button, input, etc.), Streamlit re-runs the entire script from top to bottom.

- This design means you don’t need to manually manage UI state or “which part of the page changed” — Streamlit handles that for you.

- For expensive or long-running computations, you can use caching (via decorators like @st.cache_data) so that repeated runs don’t re-compute everything.

- ● Displaying data & UI elements
- Use built-in functions like st.write(), st.text(), st.line_chart(), st.map(), etc. to show text, tables, charts, maps.

- Streamlit supports “magic” — meaning if you simply write a variable or DataFrame on its own line (instead of calling st.write()), Streamlit will render it automatically.

- ● Widgets & interactivity
- Widgets like st.slider, st.selectbox, st.button, st.text_input, etc. let you collect user input. The value(s) from these widgets can be used like normal Python variables.

- To maintain state (e.g. store user selections across runs), you can use widget keys or the session-state API.

- ● Layout controls
- Layout helpers: st.sidebar lets you build a sidebar for controls (filters, options), while keeping main screen for primary output.

- You can also use st.columns for multi-column layouts, and st.expander to hide/show parts of content.

- For long operations or to show progress, functions like st.progress() or st.spinner() can provide feedback.


---

## Example: Build a Simple Data-App

- A prototypical example from the docs (their “Create an app” tutorial) goes roughly:

- import streamlit as st
- import pandas as pd

- st.title("My first data app")

- df = pd.read_csv("data.csv")
- st.write(df)                # show raw data
- st.line_chart(df["value"])  # draw a line chart

- x = st.slider("Pick a number", 0, 100)
- st.write("You picked:", x, "— double that is", x * 2)
- Then run:
- streamlit run my_app.py
- Every time you change something in the script and save → Streamlit reloads, and the app view updates.
- You can easily extend this: add filters (sliders, selectboxes), live charts, tables, maps, caching for expensive computations, layout tweaks, sidebars, etc.
- 🔎 What’s New / Recent Features (as of latest docs)
- According to the “What’s new” section on the docs homepage, recent updates to Streamlit include:
- Streamlit Docs
- Custom Components v2 — ability to create frameless custom UI with bidirectional data flow using st.components.v2.component.
- Streamlit Docs
- Theming support (light & dark mode) — you can configure both light and dark themes in the app.
- Streamlit Docs
- More granular spacing controls (vertical/horizontal) via st.space.
- Streamlit Docs
- Flex container support — charts & dataframes now accept width/height parameters to play nicely in flexible container layouts.
- Streamlit Docs
- Improved widget identity via keys — more widgets use key-based identity to avoid unwanted resets when parameters change.
- Streamlit Docs
- These improvements make it easier to build polished, responsive, customizable interfaces — not just quick prototypes.
