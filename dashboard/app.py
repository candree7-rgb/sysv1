"""
OB Scalper Trading Dashboard
============================
Real-time trading performance dashboard connected to Supabase.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from supabase import create_client

# Page config
st.set_page_config(
    page_title="OB Scalper Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better dark mode styling
st.markdown("""
<style>
    /* Main metrics styling */
    [data-testid="metric-container"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 15px;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 28px;
    }

    /* Positive/Negative colors */
    .metric-positive {
        color: #00d4aa !important;
    }
    .metric-negative {
        color: #ff6b6b !important;
    }

    /* Card styling */
    .stats-card {
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2d3748;
    }

    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .sub-header {
        color: #888;
        margin-top: 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Table styling */
    .dataframe {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# SUPABASE CONNECTION
# ============================================

@st.cache_resource
def get_supabase_client():
    """Initialize Supabase client"""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')

    if not url or not key:
        return None

    return create_client(url, key)


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_trades(start_date: datetime = None, end_date: datetime = None):
    """Load trades from Supabase with optional date filter"""
    client = get_supabase_client()

    if not client:
        return pd.DataFrame()

    try:
        query = client.table('trades').select('*').not_.is_('exit_time', 'null')

        if start_date:
            query = query.gte('entry_time', start_date.isoformat())
        if end_date:
            query = query.lte('entry_time', end_date.isoformat())

        result = query.order('entry_time', desc=True).execute()

        if result.data:
            df = pd.DataFrame(result.data)
            # Convert timestamps
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            return df

        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading trades: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_open_trades():
    """Load currently open trades"""
    client = get_supabase_client()

    if not client:
        return pd.DataFrame()

    try:
        result = client.table('trades').select('*').is_('exit_time', 'null').execute()

        if result.data:
            df = pd.DataFrame(result.data)
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            return df

        return pd.DataFrame()

    except Exception as e:
        return pd.DataFrame()


# ============================================
# CALCULATIONS
# ============================================

def calculate_stats(df: pd.DataFrame) -> dict:
    """Calculate trading statistics"""
    if df.empty:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_duration': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'tp1_rate': 0,
            'tp2_rate': 0,
            'sl_rate': 0,
        }

    wins = df[df['is_win'] == True]
    losses = df[df['is_win'] == False]

    total_trades = len(df)
    win_count = len(wins)
    loss_count = len(losses)

    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    # PnL calculations
    total_pnl = df['realized_pnl'].sum() if 'realized_pnl' in df.columns else 0
    avg_win = wins['realized_pnl'].mean() if len(wins) > 0 and 'realized_pnl' in wins.columns else 0
    avg_loss = losses['realized_pnl'].mean() if len(losses) > 0 and 'realized_pnl' in losses.columns else 0

    # Profit factor
    gross_profit = wins['realized_pnl'].sum() if len(wins) > 0 and 'realized_pnl' in wins.columns else 0
    gross_loss = abs(losses['realized_pnl'].sum()) if len(losses) > 0 and 'realized_pnl' in losses.columns else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    # Duration
    if 'duration_minutes' in df.columns:
        avg_duration = df['duration_minutes'].mean() if not df['duration_minutes'].isna().all() else 0
    else:
        avg_duration = 0

    # Best/Worst
    best_trade = df['realized_pnl'].max() if 'realized_pnl' in df.columns and len(df) > 0 else 0
    worst_trade = df['realized_pnl'].min() if 'realized_pnl' in df.columns and len(df) > 0 else 0

    # TP/SL rates
    tp1_hits = len(df[df['tp1_hit'] == True]) if 'tp1_hit' in df.columns else 0
    tp2_hits = len(df[df['tp2_hit'] == True]) if 'tp2_hit' in df.columns else 0
    sl_hits = len(df[df['exit_reason'] == 'sl']) if 'exit_reason' in df.columns else 0

    tp1_rate = (tp1_hits / total_trades * 100) if total_trades > 0 else 0
    tp2_rate = (tp2_hits / total_trades * 100) if total_trades > 0 else 0
    sl_rate = (sl_hits / total_trades * 100) if total_trades > 0 else 0

    return {
        'total_trades': total_trades,
        'wins': win_count,
        'losses': loss_count,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_duration': avg_duration,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'tp1_rate': tp1_rate,
        'tp2_rate': tp2_rate,
        'sl_rate': sl_rate,
    }


def calculate_equity_curve(df: pd.DataFrame, initial_equity: float = 10000) -> pd.DataFrame:
    """Calculate cumulative equity curve"""
    if df.empty or 'realized_pnl' not in df.columns:
        return pd.DataFrame()

    # Sort by exit time
    df_sorted = df.sort_values('exit_time').copy()
    df_sorted['cumulative_pnl'] = df_sorted['realized_pnl'].cumsum()
    df_sorted['equity'] = initial_equity + df_sorted['cumulative_pnl']

    return df_sorted[['exit_time', 'equity', 'cumulative_pnl', 'symbol', 'realized_pnl']]


# ============================================
# UI COMPONENTS
# ============================================

def render_header():
    """Render dashboard header"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<p class="main-header">📊 OB Scalper Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Real-time trading performance</p>', unsafe_allow_html=True)

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def render_time_filter() -> tuple:
    """Render time period filter in sidebar"""
    st.sidebar.markdown("### ⏰ Time Period")

    period = st.sidebar.radio(
        "Select period",
        ["1W", "1M", "3M", "6M", "1Y", "ALL", "Custom"],
        horizontal=True,
        label_visibility="collapsed"
    )

    now = datetime.utcnow()

    if period == "1W":
        start_date = now - timedelta(days=7)
        end_date = now
    elif period == "1M":
        start_date = now - timedelta(days=30)
        end_date = now
    elif period == "3M":
        start_date = now - timedelta(days=90)
        end_date = now
    elif period == "6M":
        start_date = now - timedelta(days=180)
        end_date = now
    elif period == "1Y":
        start_date = now - timedelta(days=365)
        end_date = now
    elif period == "ALL":
        start_date = None
        end_date = None
    else:  # Custom
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("From", value=now - timedelta(days=30))
            start_date = datetime.combine(start_date, datetime.min.time())
        with col2:
            end_date = st.date_input("To", value=now)
            end_date = datetime.combine(end_date, datetime.max.time())

    return start_date, end_date


def render_kpis(stats: dict):
    """Render KPI metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        pnl_color = "normal" if stats['total_pnl'] >= 0 else "inverse"
        st.metric(
            "Total PnL",
            f"${stats['total_pnl']:,.2f}",
            delta=f"{stats['total_trades']} trades",
            delta_color=pnl_color
        )

    with col2:
        st.metric(
            "Win Rate",
            f"{stats['win_rate']:.1f}%",
            delta=f"{stats['wins']}W / {stats['losses']}L"
        )

    with col3:
        st.metric(
            "Profit Factor",
            f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "∞",
            delta="Target: >1.5"
        )

    with col4:
        st.metric(
            "Avg Win",
            f"${stats['avg_win']:,.2f}",
            delta=f"Best: ${stats['best_trade']:,.2f}"
        )

    with col5:
        st.metric(
            "Avg Loss",
            f"${stats['avg_loss']:,.2f}",
            delta=f"Worst: ${stats['worst_trade']:,.2f}",
            delta_color="inverse"
        )


def render_equity_curve(df: pd.DataFrame):
    """Render equity curve chart"""
    equity_df = calculate_equity_curve(df)

    if equity_df.empty:
        st.info("No trade data available for equity curve")
        return

    fig = go.Figure()

    # Main equity line
    fig.add_trace(go.Scatter(
        x=equity_df['exit_time'],
        y=equity_df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#00d4aa', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.1)',
        hovertemplate='<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>'
    ))

    # Add trade markers
    wins = equity_df[equity_df['realized_pnl'] > 0]
    losses = equity_df[equity_df['realized_pnl'] <= 0]

    fig.add_trace(go.Scatter(
        x=wins['exit_time'],
        y=wins['equity'],
        mode='markers',
        name='Wins',
        marker=dict(color='#00d4aa', size=8, symbol='circle'),
        hovertemplate='<b>%{text}</b><br>$%{customdata:,.2f}<extra></extra>',
        text=wins['symbol'],
        customdata=wins['realized_pnl']
    ))

    fig.add_trace(go.Scatter(
        x=losses['exit_time'],
        y=losses['equity'],
        mode='markers',
        name='Losses',
        marker=dict(color='#ff6b6b', size=8, symbol='circle'),
        hovertemplate='<b>%{text}</b><br>$%{customdata:,.2f}<extra></extra>',
        text=losses['symbol'],
        customdata=losses['realized_pnl']
    ))

    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Date',
        yaxis_title='Equity ($)',
        template='plotly_dark',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.update_xaxes(gridcolor='#2d3748', zeroline=False)
    fig.update_yaxes(gridcolor='#2d3748', zeroline=False)

    st.plotly_chart(fig, use_container_width=True)


def render_exit_distribution(stats: dict):
    """Render exit type distribution pie chart"""

    labels = ['TP1 Only', 'Full TP (TP2)', 'Stop Loss', 'Break Even']

    # Calculate values (approximate based on rates)
    tp1_only = stats['tp1_rate'] - stats['tp2_rate']  # TP1 hit but not TP2
    tp2_full = stats['tp2_rate']
    sl = stats['sl_rate']
    be = 100 - tp1_only - tp2_full - sl  # Remaining is BE

    values = [max(0, tp1_only), max(0, tp2_full), max(0, sl), max(0, be)]
    colors = ['#00d4aa', '#00ff88', '#ff6b6b', '#ffd93d']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo='percent',
        textfont_size=14,
        hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>'
    )])

    fig.update_layout(
        title='Exit Distribution',
        template='plotly_dark',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_daily_pnl(df: pd.DataFrame):
    """Render daily PnL bar chart"""
    if df.empty or 'exit_time' not in df.columns:
        st.info("No data for daily PnL")
        return

    # Group by date
    df['date'] = df['exit_time'].dt.date
    daily = df.groupby('date')['realized_pnl'].sum().reset_index()
    daily['color'] = daily['realized_pnl'].apply(lambda x: '#00d4aa' if x >= 0 else '#ff6b6b')

    fig = go.Figure(data=[go.Bar(
        x=daily['date'],
        y=daily['realized_pnl'],
        marker_color=daily['color'],
        hovertemplate='<b>%{x}</b><br>PnL: $%{y:,.2f}<extra></extra>'
    )])

    fig.update_layout(
        title='Daily PnL',
        xaxis_title='Date',
        yaxis_title='PnL ($)',
        template='plotly_dark',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.update_xaxes(gridcolor='#2d3748', zeroline=False)
    fig.update_yaxes(gridcolor='#2d3748', zeroline=True, zerolinecolor='#4a5568')

    st.plotly_chart(fig, use_container_width=True)


def render_trade_table(df: pd.DataFrame):
    """Render recent trades table"""
    if df.empty:
        st.info("No trades to display")
        return

    # Select and format columns
    display_cols = ['symbol', 'direction', 'entry_price', 'exit_price',
                    'realized_pnl', 'exit_reason', 'duration_minutes', 'entry_time']

    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].head(20).copy()

    # Format columns
    if 'entry_price' in display_df.columns:
        display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:,.6f}" if pd.notna(x) else "-")
    if 'exit_price' in display_df.columns:
        display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:,.6f}" if pd.notna(x) else "-")
    if 'realized_pnl' in display_df.columns:
        display_df['realized_pnl'] = display_df['realized_pnl'].apply(
            lambda x: f"+${x:,.2f}" if x > 0 else f"-${abs(x):,.2f}" if pd.notna(x) else "-"
        )
    if 'duration_minutes' in display_df.columns:
        display_df['duration_minutes'] = display_df['duration_minutes'].apply(
            lambda x: f"{int(x)}m" if pd.notna(x) else "-"
        )
    if 'entry_time' in display_df.columns:
        display_df['entry_time'] = display_df['entry_time'].dt.strftime('%m/%d %H:%M')

    # Rename columns
    display_df.columns = ['Symbol', 'Dir', 'Entry', 'Exit', 'PnL', 'Reason', 'Duration', 'Time'][:len(available_cols)]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def render_symbol_performance(df: pd.DataFrame):
    """Render performance by symbol"""
    if df.empty:
        return

    # Group by symbol
    symbol_stats = df.groupby('symbol').agg({
        'realized_pnl': ['sum', 'count', 'mean'],
        'is_win': 'sum'
    }).round(2)

    symbol_stats.columns = ['Total PnL', 'Trades', 'Avg PnL', 'Wins']
    symbol_stats['Win Rate'] = (symbol_stats['Wins'] / symbol_stats['Trades'] * 100).round(1)
    symbol_stats = symbol_stats.sort_values('Total PnL', ascending=False).head(10)

    # Bar chart
    fig = go.Figure(data=[go.Bar(
        x=symbol_stats.index,
        y=symbol_stats['Total PnL'],
        marker_color=symbol_stats['Total PnL'].apply(lambda x: '#00d4aa' if x >= 0 else '#ff6b6b'),
        hovertemplate='<b>%{x}</b><br>PnL: $%{y:,.2f}<br>Win Rate: %{customdata}%<extra></extra>',
        customdata=symbol_stats['Win Rate']
    )])

    fig.update_layout(
        title='Top Symbols by PnL',
        xaxis_title='Symbol',
        yaxis_title='Total PnL ($)',
        template='plotly_dark',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_open_positions(df: pd.DataFrame):
    """Render currently open positions"""
    if df.empty:
        st.info("No open positions")
        return

    for _, row in df.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            direction_emoji = "🟢" if row.get('direction') == 'long' else "🔴"
            st.markdown(f"**{direction_emoji} {row.get('symbol', 'N/A')}**")

        with col2:
            st.markdown(f"Entry: ${row.get('entry_price', 0):,.6f}")

        with col3:
            entry_time = row.get('entry_time')
            if pd.notna(entry_time):
                duration = datetime.utcnow() - entry_time.to_pydatetime()
                mins = int(duration.total_seconds() / 60)
                st.markdown(f"⏱️ {mins}m")


# ============================================
# MAIN APP
# ============================================

def main():
    # Check Supabase connection
    client = get_supabase_client()

    if not client:
        st.error("⚠️ Supabase not configured!")
        st.markdown("""
        Please set environment variables:
        - `SUPABASE_URL`
        - `SUPABASE_KEY`
        """)
        return

    # Header
    render_header()

    # Sidebar
    st.sidebar.markdown("## 🎛️ Filters")
    start_date, end_date = render_time_filter()

    # Load data
    with st.spinner("Loading trades..."):
        df = load_trades(start_date, end_date)
        open_trades = load_open_trades()

    # Calculate stats
    stats = calculate_stats(df)

    # Sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    st.sidebar.metric("Total Trades", stats['total_trades'])
    st.sidebar.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    st.sidebar.metric("Profit Factor", f"{stats['profit_factor']:.2f}")

    # Open positions in sidebar
    if not open_trades.empty:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔓 Open Positions")
        render_open_positions(open_trades)

    # Main content
    st.markdown("---")

    # KPIs
    render_kpis(stats)

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns([2, 1])

    with col1:
        render_equity_curve(df)

    with col2:
        render_exit_distribution(stats)

    # Charts row 2
    col1, col2 = st.columns(2)

    with col1:
        render_daily_pnl(df)

    with col2:
        render_symbol_performance(df)

    st.markdown("---")

    # Recent trades
    st.markdown("### 📋 Recent Trades")
    render_trade_table(df)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<p style='text-align: center; color: #666;'>Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
