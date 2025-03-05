"""
pytest 설정 파일
"""

import os
import sys

# src 디렉토리를 파이썬 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))) 