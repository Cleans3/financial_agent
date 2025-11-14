"""Test VnStock API - check all available company methods"""
from vnstock import Vnstock
import pandas as pd

print("="*80)
print("Testing VnStock Company Methods")
print("="*80)

ticker = "VNM"
stock = Vnstock().stock(symbol=ticker, source='VCI')

print("\n1. OVERVIEW (Tổng quan)")
print("-"*80)
overview = stock.company.overview()
print("Columns:", overview.columns.tolist() if isinstance(overview, pd.DataFrame) else "Not a DataFrame")

print("\n2. PROFILE (Hồ sơ)")
print("-"*80)
try:
    profile = stock.company.profile()
    print("Type:", type(profile))
    if isinstance(profile, pd.DataFrame):
        print("Columns:", profile.columns.tolist())
        print("\nData:")
        print(profile.head())
except Exception as e:
    print(f"Error: {e}")

print("\n3. SHAREHOLDERS (Cổ đông)")
print("-"*80)
try:
    shareholders = stock.company.shareholders()
    print("Type:", type(shareholders))
    if isinstance(shareholders, pd.DataFrame):
        print("Shape:", shareholders.shape)
        print("Columns:", shareholders.columns.tolist())
        print("\nTop 5 shareholders:")
        print(shareholders.head())
except Exception as e:
    print(f"Error: {e}")

print("\n4. INSIDER DEALS (Giao dịch nội bộ)")
print("-"*80)
try:
    insider = stock.company.insider_deals()
    print("Type:", type(insider))
    if isinstance(insider, pd.DataFrame):
        print("Shape:", insider.shape)
        print("Columns:", insider.columns.tolist())
except Exception as e:
    print(f"Error: {e}")

print("\n5. SUBSIDIARIES (Công ty con)")
print("-"*80)
try:
    subsidiaries = stock.company.subsidiaries()
    print("Type:", type(subsidiaries))
    if isinstance(subsidiaries, pd.DataFrame):
        print("Shape:", subsidiaries.shape)
        print("Columns:", subsidiaries.columns.tolist())
        print("\nData:")
        print(subsidiaries.head())
except Exception as e:
    print(f"Error: {e}")

print("\n6. OFFICERS (Ban lãnh đạo)")
print("-"*80)
try:
    officers = stock.company.officers()
    print("Type:", type(officers))
    if isinstance(officers, pd.DataFrame):
        print("Shape:", officers.shape)
        print("Columns:", officers.columns.tolist())
        print("\nData:")
        print(officers.head())
except Exception as e:
    print(f"Error: {e}")

print("\n7. EVENTS (Sự kiện)")
print("-"*80)
try:
    events = stock.company.events()
    print("Type:", type(events))
    if isinstance(events, pd.DataFrame):
        print("Shape:", events.shape)
        print("Columns:", events.columns.tolist())
except Exception as e:
    print(f"Error: {e}")

print("\n8. NEWS (Tin tức)")
print("-"*80)
try:
    news = stock.company.news()
    print("Type:", type(news))
    if isinstance(news, pd.DataFrame):
        print("Shape:", news.shape)
        print("Columns:", news.columns.tolist())
except Exception as e:
    print(f"Error: {e}")
