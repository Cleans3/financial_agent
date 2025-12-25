#!/usr/bin/env python3
"""Test file upload and RAG query"""

import requests
import json
import time

BASE_URL = 'http://localhost:8000'

# 1. Login
print("1. Logging in...")
login_response = requests.post(f'{BASE_URL}/api/auth/login', json={
    'username': 'admin',
    'password': 'admin'
})
token = login_response.json().get('access_token')
headers = {'Authorization': f'Bearer {token}'}
print(f"   ✓ Logged in")

# 2. Create session
print("2. Creating session...")
session_response = requests.post(f'{BASE_URL}/api/sessions', headers=headers)
session_id = session_response.json().get('session_id')
print(f"   ✓ Session: {session_id}")

# 3. Upload file
print("3. Uploading file...")
with open('example_dataset/FPT Corporation Q3 2024 Financial Report.txt', 'rb') as f:
    files = {'file': f}
    upload_response = requests.post(f'{BASE_URL}/api/upload', headers=headers, params={'session_id': session_id}, files=files)
    print(f"   ✓ Upload status: {upload_response.status_code}")
    print(f"   Response: {upload_response.json()}")

time.sleep(1)

# 4. Send query about the file
print("4. Sending query about file...")
query = 'Phân tích doanh thu trong báo cáo'
chat_response = requests.post(f'{BASE_URL}/api/chat', headers=headers, json={
    'session_id': session_id,
    'message': query
})
print(f"   ✓ Chat status: {chat_response.status_code}")
answer = chat_response.json().get('answer', '')
print(f"   Answer (first 300 chars):\n   {answer[:300]}")

print("\n✅ Test completed!")
