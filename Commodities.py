
import os
import time
import math
import warnings
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# COMMODITY EDGE 3.0
# Yahoo Finance-first commodity research terminal
# ============================================================

APP_NAME = "COMMODITY EDGE 3.0"

CACHE_DIR = Path("commodity_cache")
CACHE_DIR.mkdir(exist_ok=True)

HISTORY_PERIOD = "2y"
CACHE_MINUTES = 15
MAX_WORKERS = 8
TOP_N = 5




# ============================================================
# MARKET DATA UNIVERSE
# Primary ticker = futures contract
# Fallback ticker = related ETF/ETN where available
# ============================================================
FORCE_REFRESH = False

COMMODITIES = {
    "Silver": ("SI=F", "SLV"),
    "Platinum": ("PL=F", "PPLT"),
    "Palladium": ("PA=F", "PALL"),
    "Copper": ("HG=F", "CPER"),
    "WTI Crude Oil": ("CL=F", "USO"),
    "Brent Crude Oil": ("BZ=F", "BNO"),
    "Natural Gas": ("NG=F", "UNG"),
    "Corn": ("ZC=F", "CORN"),
    "Wheat": ("ZW=F", "WEAT"),
    "Soybeans": ("ZS=F", "SOYB"),
    "Coffee": ("KC=F", "JO"),
    "Sugar": ("SB=F", "CANE"),
    "Cotton": ("CT=F", None),
    "Cocoa": ("CC=F", None),
    "Live Cattle": ("LE=F", None),
    "Lean Hogs": ("HE=F", None),
    "Lumber": ("LBR=F", None)
}

MACRO_TICKERS = {
    "Gold": "GC=F",
    "10Y Yield": "^TNX",
    "5Y Yield": "^FVX",
    "30Y Yield": "^TYX"
}

# ============================================================
# CACHE
# ============================================================

def cache_path(ticker):
    safe = ticker.replace("=", "_").replace("^", "")
    return CACHE_DIR / f"{safe}.csv"


def load_cache(ticker):
    path = cache_path(ticker)

    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

        if "Close" not in df.columns:
            return None

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])

        return df if not df.empty else None

    except Exception:
        return None


def save_cache(ticker, df):
    try:
        df.to_csv(cache_path(ticker))
    except Exception:
        pass


def cache_is_fresh(ticker):
    path = cache_path(ticker)

    if not path.exists():
        return False

    age = time.time() - path.stat().st_mtime

    return age < CACHE_MINUTES * 60


# ============================================================
# YAHOO FINANCE DOWNLOAD
# ============================================================

def download_one(ticker):
    try:
        data = yf.Ticker(ticker).history(
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=True,
            timeout=10
        )

        if data is None or data.empty or "Close" not in data.columns:
            return ticker, None

        data = data[["Close"]].copy()
        data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
        data = data.dropna()

        if len(data) < 60:
            return ticker, None

        save_cache(ticker, data)

        return ticker, data

    except Exception:
        return ticker, None


def get_market_data(tickers, force=False):
    results = {}
    to_download = []

    for ticker in tickers:
        cached = load_cache(ticker)

        if cached is not None and not force and cache_is_fresh(ticker):
            results[ticker] = cached
        else:
            to_download.append(ticker)

    if not to_download:
        return results

    print(
        f"Downloading {len(to_download)} market datasets "
        f"from Yahoo Finance...",
        end=" ",
        flush=True
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_one, ticker): ticker
            for ticker in to_download
        }

        for future in as_completed(futures):
            ticker = futures[future]

            try:
                returned_ticker, data = future.result()

                if data is not None:
                    results[returned_ticker] = data
                else:
                    cached = load_cache(ticker)

                    if cached is not None:
                        results[ticker] = cached

            except Exception:
                cached = load_cache(ticker)

                if cached is not None:
                    results[ticker] = cached

    print("Done.")

    return results


# ============================================================
# PRICE EXTRACTION AND FALLBACKS
# ============================================================

def get_close(data, ticker):
    if ticker not in data:
        return None

    df = data[ticker]

    if df is None or df.empty:
        return None

    series = df["Close"].copy()
    series.name = ticker

    return series.dropna()


def build_universe(data):
    universe = {}
    used_tickers = {}

    for name, (primary, fallback) in COMMODITIES.items():
        series = get_close(data, primary)

        if series is not None and len(series) >= 60:
            universe[name] = series
            used_tickers[name] = primary
            continue

        if fallback:
            series = get_close(data, fallback)

            if series is not None and len(series) >= 60:
                universe[name] = series
                used_tickers[name] = fallback

    return universe, used_tickers


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def safe_return(series, periods):
    if series is None or len(series) <= periods:
        return np.nan

    old = series.iloc[-periods - 1]
    new = series.iloc[-1]

    if old == 0 or pd.isna(old) or pd.isna(new):
        return np.nan

    return (new / old - 1) * 100


def calculate_volatility(returns, window=20):
    if len(returns.dropna()) < window:
        return np.nan

    return returns.dropna().tail(window).std() * np.sqrt(252) * 100


def calculate_trend(series):
    if len(series) < 200:
        return 0

    sma20 = series.rolling(20).mean().iloc[-1]
    sma50 = series.rolling(50).mean().iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    price = series.iloc[-1]

    score = 0

    if price > sma20:
        score += 0.25
    else:
        score -= 0.25

    if price > sma50:
        score += 0.25
    else:
        score -= 0.25

    if sma50 > sma200:
        score += 0.25
    else:
        score -= 0.25

    if sma20 > sma50:
        score += 0.25
    else:
        score -= 0.25

    return score


def clamp(value, low=-1, high=1):
    if pd.isna(value) or not np.isfinite(value):
        return 0.0

    return float(max(low, min(high, value)))


def scaled(value, divisor):
    if pd.isna(value):
        return 0.0

    return math.tanh(value / divisor)


# ============================================================
# MACRO ANALYSIS
# ============================================================

def get_macro_analysis(data):
    gold = get_close(data, MACRO_TICKERS["Gold"])
    yield10 = get_close(data, MACRO_TICKERS["10Y Yield"])
    yield5 = get_close(data, MACRO_TICKERS["5Y Yield"])
    yield30 = get_close(data, MACRO_TICKERS["30Y Yield"])

    if gold is None:
        raise RuntimeError("Yahoo Finance gold data is unavailable.")

    if yield10 is None:
        raise RuntimeError("Yahoo Finance 10-year yield data is unavailable.")

    gold_returns = gold.pct_change()

    yield_changes = yield10.diff()

    yield_5d = yield10.diff(5).iloc[-1]
    yield_20d = yield10.diff(20).iloc[-1]

    recent_5d = yield10.diff(5).iloc[-1]
    previous_5d = yield10.diff(5).iloc[-6]

    acceleration = recent_5d - previous_5d

    curve_5_10 = np.nan
    curve_10_30 = np.nan

    if yield5 is not None:
        aligned = pd.concat([yield5, yield10], axis=1).dropna()

        if not aligned.empty:
            curve_5_10 = (
                aligned.iloc[-1, 1] - aligned.iloc[-1, 0]
            )

    if yield30 is not None:
        aligned = pd.concat([yield10, yield30], axis=1).dropna()

        if not aligned.empty:
            curve_10_30 = (
                aligned.iloc[-1, 1] - aligned.iloc[-1, 0]
            )

    return {
        "gold": gold,
        "gold_returns": gold_returns,
        "yield10": yield10,
        "yield_changes": yield_changes,
        "yield_5d": yield_5d,
        "yield_20d": yield_20d,
        "acceleration": acceleration,
        "curve_5_10": curve_5_10,
        "curve_10_30": curve_10_30
    }


# ============================================================
# LAGGED MACRO REGRESSION
#
# Uses yesterday's gold/yield movements to explain today's
# commodity return. This avoids using same-day movements as
# though they were predictive.
# ============================================================

def calculate_macro_signal(
    commodity_returns,
    gold_returns,
    yield_changes
):
    predictors = pd.concat(
        [
            gold_returns.rename("gold"),
            yield_changes.rename("yield")
        ],
        axis=1
    )

    predictors = predictors.shift(1)

    combined = pd.concat(
        [
            commodity_returns.rename("asset"),
            predictors
        ],
        axis=1
    ).dropna()

    if len(combined) < 60:
        return 0.0

    combined = combined.tail(180)

    X = combined[["gold", "yield"]].values
    y = combined["asset"].values

    X = np.column_stack([np.ones(len(X)), X])

    try:
        coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        predictions = X @ coefficients
        residuals = y - predictions

        residual_volatility = np.std(residuals)

        if residual_volatility == 0:
            return 0.0

        latest_gold = gold_returns.dropna().iloc[-1]
        latest_yield = yield_changes.dropna().iloc[-1]

        latest_x = np.array([
            1.0,
            latest_gold,
            latest_yield
        ])

        predicted_return = float(latest_x @ coefficients[0:3])

        z_score = predicted_return / residual_volatility

        return clamp(z_score / 2)

    except Exception:
        return 0.0


# ============================================================
# COMMODITY SCORING
# ============================================================

def analyse_commodity(
    name,
    prices,
    gold,
    macro,
    relative_median
):
    returns = prices.pct_change().dropna()

    if len(returns) < 60:
        return None

    price = prices.iloc[-1]

    ret20 = safe_return(prices, 20)
    ret60 = safe_return(prices, 60)
    ret120 = safe_return(prices, 120)

    volatility = calculate_volatility(returns)

    trend = calculate_trend(prices)

    gold_ret20 = safe_return(gold, 20)

    relative_strength = ret20 - gold_ret20

    macro_signal = calculate_macro_signal(
        returns,
        macro["gold_returns"],
        macro["yield_changes"]
    )

    # Momentum: blend short and medium-term returns.
    momentum = (
        0.55 * scaled(ret20, 8) +
        0.45 * scaled(ret60, 15)
    )

    # Trend score: between -1 and +1.
    trend_score = trend

    # Macro response: lagged regression signal.
    macro_score = macro_signal

    # Relative performance against gold.
    relative_score = scaled(relative_strength, 8)

    # Risk adjustment: penalise unusually volatile instruments.
    if pd.notna(volatility):
        risk_score = 1 - min(volatility / 100, 1.5)
        risk_score = clamp(risk_score)
    else:
        risk_score = 0

    # Cross-sectional relative momentum.
    cross_sectional = scaled(ret20 - relative_median, 8)

    score = (
        momentum * 25 +
        trend_score * 20 +
        macro_score * 25 +
        relative_score * 10 +
        risk_score * 10 +
        cross_sectional * 10
    )

    sma50 = prices.rolling(50).mean().iloc[-1]
    sma200 = prices.rolling(200).mean().iloc[-1]

    bullish = price > sma50 and sma50 > sma200
    bearish = price < sma50 and sma50 < sma200

    if score >= 20 and ret20 > 0 and bullish:
        signal = "BUY"

    elif score <= -20 and ret20 < 0 and bearish:
        signal = "SELL"

    else:
        signal = "HOLD"

    return {
        "Commodity": name,
        "Price": price,
        "20D %": ret20,
        "60D %": ret60,
        "120D %": ret120,
        "Volatility %": volatility,
        "Gold Relative %": relative_strength,
        "Macro": macro_signal,
        "Score": score,
        "Signal": signal
    }


# ============================================================
# RANKING ENGINE
# ============================================================

def run_model(data):
    universe, used_tickers = build_universe(data)

    if len(universe) < 3:
        raise RuntimeError(
            "Insufficient commodity data. "
            "Try again later or check your internet connection."
        )

    macro = get_macro_analysis(data)

    gold = macro["gold"]

    momentum_values = [
        safe_return(series, 20)
        for series in universe.values()
    ]

    relative_median = float(np.nanmedian(momentum_values))

    results = []

    for name, prices in universe.items():
        try:
            result = analyse_commodity(
                name,
                prices,
                gold,
                macro,
                relative_median
            )

            if result is not None:
                result["Ticker"] = used_tickers[name]
                results.append(result)

        except Exception as error:
            print(f"Analysis skipped for {name}: {error}")

    if not results:
        raise RuntimeError("No commodities could be analysed.")

    df = pd.DataFrame(results)

    df = df.sort_values("Score", ascending=False)
    df = df.reset_index(drop=True)

    return df, macro, len(universe)


# ============================================================
# TERMINAL DISPLAY
# ============================================================

def fmt(value, decimals=2):
    if pd.isna(value):
        return "N/A"

    return f"{value:+.{decimals}f}"


def print_header():
    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 92)
    print(f"{APP_NAME:^92}")
    print("YAHOO FINANCE | MACRO ANALYSIS | COMMODITY RANKING")
    print("=" * 92)


def print_macro(macro):
    gold = macro["gold"].iloc[-1]
    yield10 = macro["yield10"].iloc[-1]

    gold_change = safe_return(macro["gold"], 1)

    print("\nMACROECONOMIC DASHBOARD")
    print("-" * 92)

    print(
        f"Gold: {gold:.2f} | Daily change: {fmt(gold_change)}%"
    )

    print(
        f"US 10Y yield: {yield10:.3f} | "
        f"5-day change: {fmt(macro['yield_5d'], 3)}"
    )

    print(
        f"10Y yield 20-day change: {fmt(macro['yield_20d'], 3)} | "
        f"Acceleration: {fmt(macro['acceleration'], 3)}"
    )

    print(
        f"5Y-10Y spread: {fmt(macro['curve_5_10'], 3)} | "
        f"10Y-30Y spread: {fmt(macro['curve_10_30'], 3)}"
    )

    print("\nInterpretation:")

    if macro["acceleration"] > 0:
        print("  Treasury yields are accelerating upwards.")
    elif macro["acceleration"] < 0:
        print("  Treasury yields are decelerating or falling.")
    else:
        print("  Treasury yield acceleration is approximately neutral.")

    if gold_change > 0:
        print("  Gold has risen on the latest available daily observation.")
    elif gold_change < 0:
        print("  Gold has fallen on the latest available daily observation.")
    else:
        print("  Gold's latest daily movement is neutral.")

    print(
        "\nNote: yield changes are expressed in index points, "
        "not percentage returns."
    )


def print_results(df, total):
    print("\nCOMMODITY RANKINGS")
    print("-" * 92)

    print(
        f"{'RANK':<5}"
        f"{'COMMODITY':<20}"
        f"{'20D %':>10}"
        f"{'60D %':>10}"
        f"{'VOL %':>10}"
        f"{'MACRO':>10}"
        f"{'SCORE':>10}"
        f"{'SIGNAL':>12}"
    )

    print("-" * 92)

    for i, row in df.iterrows():
        print(
            f"{i + 1:<5}"
            f"{row['Commodity']:<20}"
            f"{fmt(row['20D %']):>10}"
            f"{fmt(row['60D %']):>10}"
            f"{fmt(row['Volatility %']):>10}"
            f"{fmt(row['Macro']):>10}"
            f"{fmt(row['Score'], 1):>10}"
            f"{row['Signal']:>12}"
        )

    print("-" * 92)

    print(f"\nSuccessfully analysed {total} commodities.")


def print_top_five(df):
    print("\nTOP FIVE OPPORTUNITIES")
    print("-" * 92)

    top = df.head(TOP_N)

    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(
            f"{i}. {row['Commodity']} | "
            f"{row['Signal']} | "
            f"Score: {row['Score']:.1f} | "
            f"20-day return: {fmt(row['20D %'])}%"
        )

    print("\nHighest-scoring commodity is not automatically a BUY.")
    print("Check the signal and risk before interpreting the ranking.")


def print_signal_explanation():
    print("\nSIGNAL DEFINITIONS")
    print("-" * 92)

    print("BUY  : Score >= 20, positive 20-day return, bullish trend.")
    print("SELL : Score <= -20, negative 20-day return, bearish trend.")
    print("HOLD : Conditions are mixed or insufficiently strong.")

    print(
        "\nScores are model rankings, not probabilities or "
        "guaranteed forecasts."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    while True:
        print_header()

        all_tickers = list(MACRO_TICKERS.values())

        for primary, fallback in COMMODITIES.values():
            all_tickers.append(primary)

            if fallback:
                all_tickers.append(fallback)

        all_tickers = list(dict.fromkeys(all_tickers))

        start = time.perf_counter()

        try:
            data = get_market_data(
                all_tickers,
                force=FORCE_REFRESH
            )

            df, macro, total = run_model(data)

            elapsed = time.perf_counter() - start

            print_header()
            print_macro(macro)
            print_results(df, total)
            print_top_five(df)
            print_signal_explanation()

            print(f"\nRefresh completed in {elapsed:.2f} seconds.")

            print("\nDATA SOURCE: Yahoo Finance")
            print(f"CACHE: {CACHE_MINUTES} minutes")
            print(f"UNIVERSE: {total} commodities")

            # Save the latest analysis locally.
            try:
                df.to_csv(
                    CACHE_DIR / "latest_commodity_ranking.csv",
                    index=False
                )
            except Exception:
                pass

        except KeyboardInterrupt:
            print("\nExiting Commodity Edge.")
            break

        except Exception as error:
            print("\nMODEL ERROR")
            print("-" * 92)
            print(error)

        print("\nOPTIONS")
        print("[ENTER] Refresh using available cached data")
        print("[R]     Force fresh Yahoo Finance downloads")
        print("[Q]     Quit")

        choice = input("\nSelection: ").strip().lower()

        if choice == "q":
            print("Commodity Edge closed.")
            break

        elif choice == "r":
            FORCE_REFRESH = True
            # Force only the next refresh.
            # The global flag is not changed.
            main_force_refresh = True

            # Re-run the next loop with a forced download.
            # Set the module-level setting for this iteration.
            globals()["FORCE_REFRESH"] = True

        else:
            globals()["FORCE_REFRESH"] = False


if __name__ == "__main__":
    main()
