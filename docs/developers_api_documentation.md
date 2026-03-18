# API Reference - Code Generation & Developer Tools

Complete API documentation for integrating AI-powered code generation into your developer tools.

## Base URL

```
http://localhost:8000
```

## Authentication

```http
Authorization: Bearer YOUR_API_KEY
```

---

## Code Generation Endpoints

### POST /v1/code/generate

Generate code from natural language description.

**Request:**
```json
{
  "prompt": "Create a binary search tree with insert and search methods",
  "language": "python",
  "max_tokens": 500,
  "temperature": 0.2,
  "style": "clean"  // optional: "clean", "verbose", "minimal"
}
```

**Response:**
```json
{
  "id": "gen_abc123",
  "code": "class BinarySearchTree:\n    def __init__(self):\n        self.root = None\n    ...",
  "language": "python",
  "tokens_used": 287,
  "generation_time_ms": 156
}
```

### POST /v1/code/complete

Real-time code completion for IDE integration.

**Request:**
```json
{
  "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    ",
  "position": {"line": 3, "character": 4},
  "language": "python",
  "max_tokens": 100,
  "stream": true
}
```

**Streaming Response:**
```
data: {"text": "return", "index": 0}
data: {"text": " fibonacci(n-1)", "index": 1}
data: {"text": " + fibonacci(n-2)", "index": 2}
data: [DONE]
```

### POST /v1/code/refactor

Refactor and improve existing code.

**Request:**
```json
{
  "code": "def calc(a,b,op):\n    if op=='add': return a+b\n    elif op=='sub': return a-b",
  "task": "Add type hints, improve readability, add error handling",
  "language": "python"
}
```

**Response:**
```json
{
  "original_code": "def calc(a,b,op): ...",
  "refactored_code": "from typing import Union\n\ndef calculate(a: float, b: float, operation: str) -> float:\n    ...",
  "changes": [
    "Added type hints",
    "Improved variable names",
    "Added error handling for invalid operations"
  ]
}
```

### POST /v1/code/explain

Explain what code does in natural language.

**Request:**
```json
{
  "code": "lambda x: reduce(lambda a,b: a*b, range(1,x+1))",
  "language": "python",
  "detail_level": "beginner"  // beginner, intermediate, expert
}
```

**Response:**
```json
{
  "explanation": "This is a lambda function that calculates the factorial of a number...",
  "complexity": "O(n)",
  "concepts": ["lambda functions", "reduce", "recursion"],
  "use_cases": ["calculating factorials", "mathematical operations"]
}
```

### POST /v1/code/fix

Fix bugs and errors in code.

**Request:**
```json
{
  "code": "def divide(a, b):\n    return a / b",
  "error": "ZeroDivisionError: division by zero",
  "language": "python"
}
```

**Response:**
```json
{
  "fixed_code": "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
  "explanation": "Added check for division by zero",
  "test_cases": [
    "divide(10, 2) -> 5.0",
    "divide(10, 0) -> raises ValueError"
  ]
}
```

### POST /v1/code/document

Generate documentation for code.

**Request:**
```json
{
  "code": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    ...",
  "language": "python",
  "style": "google"  // google, numpy, sphinx
}
```

**Response:**
```json
{
  "documented_code": "def merge_sort(arr: List[int]) -> List[int]:\n    \"\"\"Sort an array using merge sort algorithm.\n\n    Args:\n        arr: List of integers to sort\n\n    Returns:\n        Sorted list in ascending order\n    \"\"\"",
  "docstring": "Sort an array using merge sort algorithm..."
}
```

### POST /v1/code/test

Generate unit tests for code.

**Request:**
```json
{
  "code": "def is_palindrome(s):\n    return s == s[::-1]",
  "language": "python",
  "framework": "pytest"  // pytest, unittest, jest, junit
}
```

**Response:**
```json
{
  "test_code": "import pytest\n\ndef test_is_palindrome_positive():\n    assert is_palindrome('racecar') == True\n\ndef test_is_palindrome_negative():\n    assert is_palindrome('hello') == False",
  "test_count": 5,
  "coverage_estimate": "95%"
}
```

---

## Language Support

### Supported Languages

```http
GET /v1/languages
```

**Response:**
```json
{
  "languages": [
    {
      "name": "python",
      "version": "3.11",
      "features": ["completion", "generation", "refactor", "test"]
    },
    {
      "name": "javascript",
      "version": "ES2023",
      "features": ["completion", "generation", "refactor", "test"]
    }
  ]
}
```

---

## Batch Operations

### POST /v1/batch/generate

Generate multiple code snippets in one request.

**Request:**
```json
{
  "requests": [
    {"prompt": "function to validate email", "language": "python"},
    {"prompt": "function to validate phone", "language": "python"},
    {"prompt": "function to validate URL", "language": "python"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {"code": "import re\ndef validate_email(email): ...", "tokens": 45},
    {"code": "import re\ndef validate_phone(phone): ...", "tokens": 52},
    {"code": "import re\ndef validate_url(url): ...", "tokens": 38}
  ],
  "total_time_ms": 234,
  "total_tokens": 135
}
```

---

## Health & Monitoring

### GET /health

Check system health.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "uptime_seconds": 3600,
  "version": "1.0.0"
}
```

### GET /stats

Get performance statistics.

**Response:**
```json
{
  "requests_total": 125000,
  "requests_per_second": 42,
  "latency": {
    "p50_ms": 42,
    "p95_ms": 156,
    "p99_ms": 178
  },
  "gpu_utilization": 85,
  "model_name": "codellama-13b"
}
```

---

## Error Handling

All endpoints return standard HTTP status codes:

```
200 OK - Request successful
400 Bad Request - Invalid parameters
401 Unauthorized - Missing/invalid API key
429 Too Many Requests - Rate limit exceeded
500 Internal Server Error - Server error
503 Service Unavailable - Model not ready
```

**Error Response Format:**
```json
{
  "error": {
    "code": "invalid_parameter",
    "message": "Language 'xyz' is not supported",
    "details": {
      "supported_languages": ["python", "javascript", "java"]
    }
  }
}
```

---

## Rate Limiting

Default rate limits:
- Free tier: 60 requests/minute
- Pro tier: 1000 requests/minute
- Enterprise: Custom limits

Headers included in response:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1640000000
```

---

## Examples

### Complete IDE Integration Flow

```python
import requests

class CodeAssistant:
    def __init__(self, api_key):
        self.base_url = "http://localhost:8000"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def complete_code(self, code, position):
        """Get code completion suggestions."""
        response = requests.post(
            f"{self.base_url}/v1/code/complete",
            headers=self.headers,
            json={
                "code": code,
                "position": position,
                "language": "python",
                "max_tokens": 100
            }
        )
        return response.json()
    
    def generate_function(self, description):
        """Generate function from description."""
        response = requests.post(
            f"{self.base_url}/v1/code/generate",
            headers=self.headers,
            json={
                "prompt": description,
                "language": "python",
                "max_tokens": 300
            }
        )
        return response.json()["code"]
    
    def refactor_code(self, code, improvements):
        """Refactor and improve code."""
        response = requests.post(
            f"{self.base_url}/v1/code/refactor",
            headers=self.headers,
            json={
                "code": code,
                "task": improvements,
                "language": "python"
            }
        )
        return response.json()["refactored_code"]

# Usage
assistant = CodeAssistant(api_key="your-key")

# Generate function
func = assistant.generate_function("function to merge two sorted lists")
print(func)

# Get completion
completion = assistant.complete_code("def fibonacci(n):\n    ", {"line": 1, "character": 4})
print(completion)
```

---

For more examples, see [IDE Integration Guide](ide_integration.md)
