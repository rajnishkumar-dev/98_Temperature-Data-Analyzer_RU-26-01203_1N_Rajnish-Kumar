import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(filepath="temps.csv"):
    if not Path(filepath).exists():
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    df = pd.read_csv(filepath, names=["Date", "Temp_C"], header=0, dtype={"Date": str, "Temp_C": str})
    original = df["Temp_C"].copy()
    df["Temp_C"] = pd.to_numeric(df["Temp_C"], errors="coerce")
    for idx in df[df["Temp_C"].isna() & original.notna()].index:
        print(f"Invalid reading '{original.loc[idx]}' on {df.loc[idx, 'Date']}. Value ignored (NaN).")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).reset_index(drop=True)

def calculate_statistics(df):
    temps = df["Temp_C"].dropna().to_numpy(dtype=float)
    if len(temps) == 0:
        print("No valid temperature readings available.")
        return
    print(f"Mean: {np.mean(temps):.2f} | Median: {np.median(temps):.2f} | Max: {np.max(temps):.0f} | Min: {np.min(temps):.0f} | Std Dev: {np.std(temps, ddof=0):.2f}")
    hot = df[df["Temp_C"] > 35]
    print("No days above 35 C" if hot.empty else f"Hot days: {', '.join(hot['Date'].dt.strftime('%Y-%m-%d'))}")

def detect_outliers(df):
    temps = df["Temp_C"].dropna()
    if temps.empty:
        print("No valid temperature readings available.")
        return
    q1, q3 = np.percentile(temps, [25, 75])
    iqr = q3 - q1
    outliers = df[(df["Temp_C"] < q1 - 1.5 * iqr) | (df["Temp_C"] > q3 + 1.5 * iqr)]
    print("\n--- Outlier Detection (IQR method) ---")
    print("No statistical outliers found." if outliers.empty else "Outliers detected:")
    for _, row in outliers.iterrows():
        print(f"  {row['Date'].strftime('%Y-%m-%d')}: {row['Temp_C']} °C")
    hot = df[df["Temp_C"] > 35]
    print("\n--- Hot Days (Temp > 35 °C) ---")
    print("No days above 35 C" if hot.empty else "")
    for _, row in hot.iterrows():
        print(f"  {row['Date'].strftime('%Y-%m-%d')}: {row['Temp_C']} °C")

def convert_to_fahrenheit_and_plot(df):
    valid = df.dropna(subset=["Temp_C"]).copy()
    if valid.empty:
        print("No valid temperature readings available.")
        return
    valid["Temp_F"] = valid["Temp_C"] * 9 / 5 + 32
    print(" | ".join(f"{t:.2f} F" for t in valid["Temp_F"]))
    plt.figure(figsize=(10, 5))
    plt.plot(valid["Date"], valid["Temp_F"], marker="o", linestyle="-", color="tab:blue", label="Temperature (°F)")
    plt.title("Temperature Trend (Fahrenheit)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°F)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("Line chart displayed")

def plot_histogram(df):
    temps = df["Temp_C"].dropna()
    if temps.empty:
        print("No valid temperature readings available.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(temps, bins=8, edgecolor="black", color="skyblue", alpha=0.8)
    plt.title("Temperature Distribution (°C)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    print("Histogram displayed")

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "temps.csv"
    print(f"Loading data from: {csv_file}")
    df = load_data(csv_file)
    print(f"\nLoaded {len(df)} rows ({df['Temp_C'].notna().sum()} valid temperature readings)")
    while True:
        print("\n" + "=" * 50)
        print("   TEMPERATURE DATA ANALYZER")
        print("=" * 50)
        print("1. Statistics (Mean, Median, Max, Min, Std Dev + Hot Days)")
        print("2. Detect Hot Days & Outliers")
        print("3. Convert to Fahrenheit and Plot Line Chart")
        print("4. Plot Histogram")
        print("5. Exit")
        print("=" * 50)
        choice = input("Enter your choice (1-5): ").strip()
        if choice == "1":
            print("\n--- Statistics ---")
            calculate_statistics(df)
        elif choice == "2":
            detect_outliers(df)
        elif choice == "3":
            print("\n--- Fahrenheit Conversion & Line Chart ---")
            convert_to_fahrenheit_and_plot(df)
        elif choice == "4":
            print("\n--- Histogram ---")
            plot_histogram(df)
        elif choice == "5":
            print("Exiting program. Thank you!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()