"""
Automated Test Script for Financial Agent
Reads test questions from Excel file and tests the API
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime
from pathlib import Path


class FinancialAgentTester:
    def __init__(self, api_url="http://localhost:8000/api/chat", excel_file=None):
        self.api_url = api_url
        self.excel_file = excel_file
        self.results = []
        
    def load_questions_from_excel(self, sheet_name=0):
        """
        Load questions from Excel file
        Expected columns: 'question' or 'câu hỏi'
        Optional columns: 'expected_answer' or 'câu trả lời mong đợi', 'category'
        """
        try:
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            
            # Try to find question column
            question_col = None
            for col in df.columns:
                if 'question' in col.lower() or 'câu hỏi' in col.lower():
                    question_col = col
                    break
            
            if not question_col:
                # Use first column as questions
                question_col = df.columns[0]
            
            # Try to find expected answer column
            expected_col = None
            for col in df.columns:
                if 'expected' in col.lower() or 'mong đợi' in col.lower() or 'answer' in col.lower():
                    expected_col = col
                    break
            
            # Try to find category column
            category_col = None
            for col in df.columns:
                if 'category' in col.lower() or 'loại' in col.lower() or 'nhóm' in col.lower():
                    category_col = col
                    break
            
            questions = []
            for idx, row in df.iterrows():
                question_data = {
                    'id': idx + 1,
                    'question': str(row[question_col]).strip(),
                }
                
                if expected_col and pd.notna(row[expected_col]):
                    question_data['expected_answer'] = str(row[expected_col]).strip()
                
                if category_col and pd.notna(row[category_col]):
                    question_data['category'] = str(row[category_col]).strip()
                
                # Skip empty questions
                if question_data['question'] and question_data['question'] != 'nan':
                    questions.append(question_data)
            
            print(f"✅ Loaded {len(questions)} questions from Excel file")
            return questions
            
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return []
    
    def test_question(self, question_data, delay=2):
        """Test a single question"""
        question_id = question_data['id']
        question = question_data['question']
        category = question_data.get('category', 'N/A')
        
        print(f"\n{'='*80}")
        print(f"Test #{question_id}: {category}")
        print(f"Question: {question}")
        print(f"{'-'*80}")
        
        try:
            # Send request
            start_time = time.time()
            response = requests.post(
                self.api_url,
                json={"question": question},
                timeout=60
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Parse response
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                
                # Try to parse JSON answer
                try:
                    answer_json = json.loads(answer)
                    success = answer_json.get('success', None)
                except:
                    success = None
                
                result = {
                    'id': question_id,
                    'question': question,
                    'category': category,
                    'status': 'SUCCESS',
                    'status_code': response.status_code,
                    'response_time': round(response_time, 2),
                    'answer': answer,
                    'success': success,
                    'answer_length': len(answer),
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"✅ Status: SUCCESS")
                print(f"⏱️  Response Time: {response_time:.2f}s")
                print(f"📏 Answer Length: {len(answer)} chars")
                
                # Show first 200 chars of answer
                print(f"\n📝 Answer Preview:")
                print(answer[:200] + "..." if len(answer) > 200 else answer)
                
            else:
                result = {
                    'id': question_id,
                    'question': question,
                    'category': category,
                    'status': 'FAILED',
                    'status_code': response.status_code,
                    'response_time': round(response_time, 2),
                    'error': response.text,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"❌ Status: FAILED")
                print(f"Error: {response.text[:200]}")
            
            self.results.append(result)
            
            # Delay before next request
            if delay > 0:
                time.sleep(delay)
            
            return result
            
        except requests.Timeout:
            result = {
                'id': question_id,
                'question': question,
                'category': category,
                'status': 'TIMEOUT',
                'error': 'Request timeout (>60s)',
                'timestamp': datetime.now().isoformat()
            }
            print(f"⏰ Status: TIMEOUT")
            self.results.append(result)
            return result
            
        except Exception as e:
            result = {
                'id': question_id,
                'question': question,
                'category': category,
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"💥 Status: ERROR - {e}")
            self.results.append(result)
            return result
    
    def run_tests(self, questions, delay=2):
        """Run all tests"""
        print(f"\n{'='*80}")
        print(f"🚀 Starting Automated Tests")
        print(f"{'='*80}")
        print(f"Total Questions: {len(questions)}")
        print(f"API URL: {self.api_url}")
        print(f"Delay between requests: {delay}s")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        for question_data in questions:
            self.test_question(question_data, delay=delay)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Summary
        self.print_summary(total_time)
        
        return self.results
    
    def print_summary(self, total_time):
        """Print test summary"""
        print(f"\n{'='*80}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*80}")
        
        total = len(self.results)
        success = len([r for r in self.results if r['status'] == 'SUCCESS'])
        failed = len([r for r in self.results if r['status'] == 'FAILED'])
        timeout = len([r for r in self.results if r['status'] == 'TIMEOUT'])
        error = len([r for r in self.results if r['status'] == 'ERROR'])
        
        print(f"Total Tests: {total}")
        print(f"✅ Success: {success} ({success/total*100:.1f}%)")
        print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"⏰ Timeout: {timeout} ({timeout/total*100:.1f}%)")
        print(f"💥 Error: {error} ({error/total*100:.1f}%)")
        print(f"\n⏱️  Total Time: {total_time:.2f}s")
        
        if success > 0:
            avg_time = sum([r.get('response_time', 0) for r in self.results if r['status'] == 'SUCCESS']) / success
            print(f"📈 Average Response Time: {avg_time:.2f}s")
        
        print(f"{'='*80}\n")
    
    def save_results(self, output_file='test_results.xlsx'):
        """Save results to Excel file"""
        try:
            df = pd.DataFrame(self.results)
            df.to_excel(output_file, index=False, sheet_name='Test Results')
            print(f"✅ Results saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None


def main():
    """Main function"""
    import sys
    
    # Default test questions if no Excel file provided
    default_questions = [
        {"id": 1, "question": "Thông tin về công ty VNM", "category": "Company Info"},
        {"id": 2, "question": "Cổ đông lớn của VCB", "category": "Shareholders"},
        {"id": 3, "question": "Ban lãnh đạo HPG", "category": "Officers"},
        {"id": 4, "question": "Công ty con của VNM", "category": "Subsidiaries"},
        {"id": 5, "question": "Giá VCB 3 tháng gần nhất", "category": "Historical Data"},
        {"id": 6, "question": "Tính SMA-20 cho HPG", "category": "Technical Analysis"},
        {"id": 7, "question": "RSI của VIC hiện tại", "category": "Technical Analysis"},
        {"id": 8, "question": "Sự kiện gần đây của FPT", "category": "Company Events"},
    ]
    
    # Initialize tester
    tester = FinancialAgentTester(api_url="http://localhost:8000/api/chat")
    
    # Check if Excel file provided
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
        print(f"📂 Loading questions from: {excel_file}")
        questions = tester.load_questions_from_excel()
        tester.excel_file = excel_file
    else:
        print(f"📝 Using default test questions")
        questions = default_questions
    
    if not questions:
        print("❌ No questions to test!")
        return
    
    # Run tests
    results = tester.run_tests(questions, delay=2)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{timestamp}.xlsx"
    tester.save_results(output_file)


if __name__ == "__main__":
    main()
