"""
CricketIQ Visualization Package
==================================
All chart functions return matplotlib.Figure objects.
Streamlit renders them with st.pyplot(fig).

Modules:
    style           — shared theme, colours, make_figure(), empty_chart()
    batting_charts  — phase bars, season trend, scoring map, dismissal donut
    bowling_charts  — phase economy, over-by-over, wicket types, season trend
    matchup_charts  — summary tiles, phase comparison, recent form, dominance gauge
    team_charts     — win rates, head-to-head, toss impact, venue bars
"""
