"""Quick test to see what get_company_info returns"""
import sys
sys.path.insert(0, 'src')

from tools.vnstock_tools import get_company_info

# LangChain tool needs .invoke() method
result = get_company_info.invoke({"ticker": "VNM"})
print("="*80)
print("RESULT FROM get_company_info:")
print("="*80)
print(result)
print("="*80)
