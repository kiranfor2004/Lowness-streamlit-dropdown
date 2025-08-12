import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(layout="wide")

# -------------------------------
# App Title
# -------------------------------
st.markdown("<h1 style='text-align: center;'>📊 LOWESS + RSI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>TradingView-style interactive charts</p>", unsafe_allow_html=True)

# -------------------------------
# Data Source Selector
# -------------------------------
st.markdown("---")

# Add file upload option and refresh button
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    data_source = st.selectbox(
        "📈 Select Data Source",
        options=["Futures", "Index"],
        index=0,  # Default to Futures
        help="Choose between Futures.csv or Index.csv data"
    )

with col2:
    uploaded_file = st.file_uploader(
        "Or upload a CSV file",
        type=['csv'],
        help="Upload your own CSV file with OHLCV data"
    )

with col3:
    st.write("")  # Empty space for alignment
    if st.button("🔄 Refresh", help="Clear cache and reload data"):
        st.cache_data.clear()
        st.rerun()

# -------------------------------
# Load and Clean Data
# -------------------------------
@st.cache_data
def load_data(source, _cache_key=None):
    filename = f"{source.lower()}.csv"  # futures.csv or index.csv
    
    try:
        # First try to read as a standard CSV
        df = pd.read_csv(filename, encoding='latin1', low_memory=False)
        
        # Auto-detect column structure based on number of columns
        num_cols = len(df.columns)
        
        if num_cols >= 6:  # Minimum OHLCV + date/time
            # Check if first row contains headers or data
            first_row_is_data = True
            try:
                # Try to convert first few values to numbers
                for i in range(2, min(6, num_cols)):
                    pd.to_numeric(df.iloc[0, i])
            except:
                first_row_is_data = False
            
            if first_row_is_data and not any(col.lower() in ['date', 'time', 'open', 'high', 'low', 'close'] for col in df.columns):
                # No headers detected, assign column names based on structure
                if num_cols == 7:  # Index format: date, time, open, high, low, close, volume
                    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'Volume']
                elif num_cols == 11:  # Futures format with VIX
                    df.columns = [
                        'date', 'time', 'open', 'high', 'low', 'close', 'Volume',
                        'vix_open', 'vix_high', 'vix_low', 'vix_close'
                    ]
                elif num_cols >= 6:  # Generic OHLCV
                    base_cols = ['date', 'time', 'open', 'high', 'low', 'close']
                    if num_cols > 6:
                        base_cols.extend([f'col_{i}' for i in range(7, num_cols + 1)])
                    df.columns = base_cols[:num_cols]
            
            # Standardize column names (handle case variations)
            df.columns = [col.lower().strip() for col in df.columns]
            
        else:
            st.error(f"❌ '{filename}' has insufficient columns (found {num_cols}, need at least 6)")
            return None
            
    except FileNotFoundError:
        st.error(f"❌ '{filename}' not found. Please ensure the file exists in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error reading '{filename}': {str(e)}")
        return None

    # Clean and convert data
    try:
        # Handle different date/time combinations
        if 'date' in df.columns and 'time' in df.columns:
            # Try different date formats
            date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
            time_formats = ['%H:%M:%S', '%H:%M']
            
            datetime_created = False
            for date_fmt in date_formats:
                for time_fmt in time_formats:
                    try:
                        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), 
                                                      format=f'{date_fmt} {time_fmt}')
                        datetime_created = True
                        break
                    except:
                        continue
                if datetime_created:
                    break
            
            if not datetime_created:
                # Last resort: let pandas auto-detect
                try:
                    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), 
                                                  infer_datetime_format=True, errors='coerce')
                except:
                    st.error(f"❌ Could not parse date/time columns in '{filename}'")
                    return None
        
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        
        else:
            st.error(f"❌ No date/time columns found in '{filename}'")
            return None
        
        # Convert numeric columns (handle different column structures)
        required_cols = ['open', 'high', 'low', 'close']
        optional_cols = ['volume', 'vix_open', 'vix_high', 'vix_low', 'vix_close']
        
        # Check for required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns in '{filename}': {missing_cols}")
            return None
        
        # Convert all numeric columns
        all_numeric_cols = required_cols + [col for col in optional_cols if col in df.columns]
        for col in all_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add volume column if missing (set to 0)
        if 'volume' not in df.columns:
            df['volume'] = 0
            st.warning(f"⚠️ No volume data found in '{filename}', using zeros")
        
        # Clean data
        df.dropna(subset=['close', 'datetime'], inplace=True)
        
        if df.empty:
            st.error(f"❌ No valid data found after cleaning '{filename}'.")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"❌ Error processing data from '{filename}': {str(e)}")
        return None

# Load data based on selected source or uploaded file
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1', low_memory=False)
        data_source_display = f"Uploaded: {uploaded_file.name}"
        
        # Standardize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Try to standardize column names
        if len(df.columns) >= 6:  # Minimum required columns
            # Handle datetime
            if 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), 
                                              errors='coerce', infer_datetime_format=True)
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            df.dropna(subset=['close', 'datetime'], inplace=True)
            
            if df.empty:
                st.error("❌ No valid data found in uploaded file.")
                st.stop()
        else:
            st.error(f"❌ Uploaded file must have at least 6 columns (found {len(df.columns)})")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error reading uploaded file: {str(e)}")
        st.stop()
else:
    # Use a cache key that includes the data source to force reload when changing
    cache_key = f"{data_source}_{hash(data_source)}"
    df = load_data(data_source, _cache_key=cache_key)
    data_source_display = data_source

if df is None:
    st.stop()

# Add data preview section
with st.expander("🔍 Data Preview & Info", expanded=False):
    st.write("**Column Information:**")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 else 'N/A' for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)
    
    st.write("**First 5 Rows:**")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write("**Data Summary:**")
    st.write(f"- Total rows: {len(df):,}")
    if 'datetime' in df.columns:
        st.write(f"- Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    if 'close' in df.columns:
        st.write(f"- Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")

# Display data info with clear indication of current source
if data_source == "Futures":
    icon = "🔮"
elif data_source == "Index":
    icon = "📈"
else:
    icon = "📁"
    
st.info(f"{icon} **Currently showing: {data_source_display}** | {len(df):,} records")

# -------------------------------
# Date Filter
# -------------------------------
st.markdown("---")
min_date = df['datetime'].min().date()
max_date = df['datetime'].max().date()

col1, col2, _ = st.columns([1, 1, 4])
with col1:
    from_date = st.date_input("📅 From Date", min_date, min_value=min_date, max_value=max_date, key=f"from_{data_source}_{hash(str(df.shape))}")
with col2:
    to_date = st.date_input("📅 To Date", max_date, min_value=min_date, max_value=max_date, key=f"to_{data_source}_{hash(str(df.shape))}")

if from_date > to_date:
    st.error("❌ 'From Date' cannot be after 'To Date'")
    st.stop()

mask = (df['datetime'] >= pd.Timestamp(from_date)) & (df['datetime'] <= pd.Timestamp(to_date) + pd.Timedelta(days=1))
df_filtered = df[mask].copy()
if df_filtered.empty:
    st.warning(f"⚠️ No data found between {from_date} and {to_date}")
    st.stop()

df_filtered.sort_values('datetime', inplace=True)

# Convert Datetime to string for category x-axis (removes gaps)
df_filtered['TimeLabel'] = df_filtered['datetime'].dt.strftime('%Y-%m-%d %H:%M')

# -------------------------------
# RSI Calculation
# -------------------------------
delta = df_filtered['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df_filtered['RSI'] = 100 - (100 / (1 + rs))

# -------------------------------
# LOWESS Calculation
# -------------------------------
x = np.arange(len(df_filtered))
y = df_filtered['close'].values
df_filtered['LOWESS'] = lowess(y, x, frac=0.1, it=3)[:, 1]
residuals = y - df_filtered['LOWESS']

window = min(50, len(df_filtered))
rolling_std = pd.Series(residuals).rolling(window=window, center=True).std()

df_filtered['Upper_Band_1'] = df_filtered['LOWESS'] + rolling_std
df_filtered['Upper_Band_2'] = df_filtered['LOWESS'] + 2 * rolling_std
df_filtered['Lower_Band_1'] = df_filtered['LOWESS'] - rolling_std
df_filtered['Lower_Band_2'] = df_filtered['LOWESS'] - 2 * rolling_std

# -------------------------------
# Combined Chart (Price + Volume + RSI)
# -------------------------------
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.65, 0.15, 0.2],
    subplot_titles=(f"{data_source_display} - Price & LOWESS Channel ({from_date} to {to_date})", "Volume", "RSI (14)")
)

# --- Price Chart ---
# Set colors based on data source for visual distinction
price_colors = {
    'Futures': {'up': '#26a69a', 'down': '#ef5350'},  # Green/Red
    'Index': {'up': '#2196f3', 'down': '#ff9800'}     # Blue/Orange
}

colors = price_colors[data_source]

fig.add_trace(go.Candlestick(
    x=df_filtered['TimeLabel'],
    open=df_filtered['open'],
    high=df_filtered['high'],
    low=df_filtered['low'],
    close=df_filtered['close'],
    name=f'{data_source} Price',
    increasing_line_color=colors['up'],
    decreasing_line_color=colors['down'],
    increasing_fillcolor=colors['up'],
    decreasing_fillcolor=colors['down'],
    opacity=0.9
), row=1, col=1)

fig.add_trace(go.Scatter(x=df_filtered['TimeLabel'], y=df_filtered['LOWESS'],
                         mode='lines', name='LOWESS Trend', line=dict(color='#fdd835', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtered['TimeLabel'], y=df_filtered['Upper_Band_1'], mode='lines', name='Upper Band 1σ', line=dict(color='#ef5350', width=1, dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtered['TimeLabel'], y=df_filtered['Upper_Band_2'], mode='lines', name='Upper Band 2σ', line=dict(color='#ef5350', width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtered['TimeLabel'], y=df_filtered['Lower_Band_1'], mode='lines', name='Lower Band 1σ', line=dict(color='#26a69a', width=1, dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtered['TimeLabel'], y=df_filtered['Lower_Band_2'], mode='lines', name='Lower Band 2σ', line=dict(color='#26a69a', width=1, dash='dash')), row=1, col=1)

# --- Volume Bars ---
if 'volume' in df_filtered.columns and df_filtered['volume'].sum() > 0:
    fig.add_trace(go.Bar(
        x=df_filtered['TimeLabel'],
        y=df_filtered['volume'],
        name='Volume',
        marker_color=np.where(df_filtered['close'] >= df_filtered['open'], colors['up'], colors['down']),
        opacity=0.6
    ), row=2, col=1)
else:
    # If no volume data, show a message
    fig.add_annotation(
        text="No Volume Data Available",
        xref="x2", yref="y2",
        x=0.5, y=0.5,
        xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=14, color="gray"),
        row=2, col=1
    )

# --- RSI Chart ---
fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], y=df_filtered['RSI'],
    mode='lines', name='RSI (14)', line=dict(color='#ab47bc', width=2)
), row=3, col=1)

fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought (70)", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold (30)", row=3, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="#888888", annotation_text="Midline (50)", row=3, col=1)

# --- Layout ---
fig.update_layout(
    title=dict(
        text=f"<b>{data_source_display} Trading Analysis Dashboard</b>",
        x=0.5,
        font=dict(size=16)
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(0,0,0,0.85)",
        font_size=12,
        font_color="white",
        align="left"
    ),
    height=950,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        font=dict(size=12)
    ),
    margin=dict(l=10, r=10, t=80, b=40),
    template="plotly_dark",
    uirevision='constant'
)

# X-axis as category removes gaps
fig.update_xaxes(type='category')

# Update y-axis labels
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)
fig.update_yaxes(title_text="RSI", row=3, col=1)

# Show Chart
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

# -------------------------------
# Summary Statistics
# -------------------------------
st.markdown("---")
st.markdown(f"### 📈 {data_source_display} Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    current_price = df_filtered['close'].iloc[-1]
    price_change = df_filtered['close'].iloc[-1] - df_filtered['close'].iloc[0]
    price_change_pct = (price_change / df_filtered['close'].iloc[0]) * 100
    
    st.metric(
        label="Current Price",
        value=f"{current_price:.2f}",
        delta=f"{price_change_pct:.2f}%"
    )

with col2:
    current_rsi = df_filtered['RSI'].iloc[-1]
    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
    
    st.metric(
        label="Current RSI",
        value=f"{current_rsi:.1f}",
        delta=rsi_status
    )

with col3:
    if 'volume' in df_filtered.columns and df_filtered['volume'].sum() > 0:
        avg_volume = df_filtered['volume'].mean()
        current_volume = df_filtered['volume'].iloc[-1]
        volume_vs_avg = ((current_volume - avg_volume) / avg_volume) * 100
        
        st.metric(
            label="Latest Volume",
            value=f"{current_volume:,.0f}",
            delta=f"{volume_vs_avg:.1f}% vs avg"
        )
    else:
        st.metric(
            label="Volume Data",
            value="N/A",
            delta="Not Available"
        )

with col4:
    volatility = df_filtered['close'].pct_change().std() * 100
    
    st.metric(
        label="Price Volatility",
        value=f"{volatility:.2f}%",
        delta="Daily Std Dev"
    )