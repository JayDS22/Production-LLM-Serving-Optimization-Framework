# IDE Integration Guide

Complete guide for integrating AI-powered code generation into IDEs and code editors.

## Quick Integration

Most IDEs can integrate in under 30 minutes using our REST API.

---

## VSCode Extension Integration

### Basic Setup

```typescript
// extension.ts
import * as vscode from 'vscode';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export function activate(context: vscode.ExtensionContext) {
    // Register code completion provider
    const provider = vscode.languages.registerCompletionItemProvider(
        ['python', 'javascript', 'typescript'],
        {
            async provideCompletionItems(document, position) {
                const code = document.getText();
                const offset = document.offsetAt(position);
                
                const response = await axios.post(`${API_BASE}/v1/code/complete`, {
                    code: code,
                    position: { line: position.line, character: position.character },
                    language: document.languageId,
                    max_tokens: 100
                });
                
                const completion = new vscode.CompletionItem(
                    response.data.code,
                    vscode.CompletionItemKind.Snippet
                );
                completion.insertText = new vscode.SnippetString(response.data.code);
                
                return [completion];
            }
        },
        '.'  // Trigger on dot
    );
    
    context.subscriptions.push(provider);
}
```

### Advanced Features

```typescript
// Inline code generation command
context.subscriptions.push(
    vscode.commands.registerCommand('extension.generateCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        
        const prompt = await vscode.window.showInputBox({
            prompt: 'Describe the code you want to generate'
        });
        
        const response = await axios.post(`${API_BASE}/v1/code/generate`, {
            prompt: prompt,
            language: editor.document.languageId,
            max_tokens: 500
        });
        
        editor.edit(editBuilder => {
            editBuilder.insert(editor.selection.active, response.data.code);
        });
    })
);

// Code refactoring command
context.subscriptions.push(
    vscode.commands.registerCommand('extension.refactorCode', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        
        const selection = editor.document.getText(editor.selection);
        
        const response = await axios.post(`${API_BASE}/v1/code/refactor`, {
            code: selection,
            task: 'Improve readability and add type hints',
            language: editor.document.languageId
        });
        
        editor.edit(editBuilder => {
            editBuilder.replace(editor.selection, response.data.refactored_code);
        });
    })
);
```

---

## JetBrains Plugin Integration

### Plugin Configuration

```kotlin
// CodeCompletionContributor.kt
import com.intellij.codeInsight.completion.*
import com.intellij.codeInsight.lookup.LookupElementBuilder
import com.intellij.patterns.PlatformPatterns
import com.intellij.util.ProcessingContext
import io.ktor.client.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import kotlinx.coroutines.runBlocking

class AICodeCompletionContributor : CompletionContributor() {
    private val httpClient = HttpClient()
    private val apiBase = "http://localhost:8000"
    
    init {
        extend(
            CompletionType.BASIC,
            PlatformPatterns.psiElement(),
            object : CompletionProvider<CompletionParameters>() {
                override fun addCompletions(
                    parameters: CompletionParameters,
                    context: ProcessingContext,
                    result: CompletionResultSet
                ) {
                    val editor = parameters.editor
                    val document = editor.document
                    val code = document.text
                    val position = editor.caretModel.offset
                    
                    runBlocking {
                        val response = httpClient.post("$apiBase/v1/code/complete") {
                            contentType(ContentType.Application.Json)
                            setBody(mapOf(
                                "code" to code,
                                "position" to mapOf("offset" to position),
                                "language" to "java",
                                "max_tokens" to 150
                            ))
                        }
                        
                        val completion = response.bodyAsText()
                        result.addElement(
                            LookupElementBuilder.create(completion)
                                .withIcon(AllIcons.Nodes.Method)
                                .withTypeText("AI Generated")
                        )
                    }
                }
            }
        )
    }
}
```

### Action Integration

```kotlin
// GenerateCodeAction.kt
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.ui.Messages

class GenerateCodeAction : AnAction() {
    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val editor = e.getData(CommonDataKeys.EDITOR) ?: return
        
        val prompt = Messages.showInputDialog(
            project,
            "Describe the code you want to generate:",
            "AI Code Generation",
            null
        ) ?: return
        
        runBlocking {
            val response = httpClient.post("$apiBase/v1/code/generate") {
                contentType(ContentType.Application.Json)
                setBody(mapOf(
                    "prompt" to prompt,
                    "language" to "java",
                    "max_tokens" to 500
                ))
            }
            
            val code = response.bodyAsText()
            ApplicationManager.getApplication().runWriteAction {
                editor.document.insertString(editor.caretModel.offset, code)
            }
        }
    }
}
```

---

## Web-Based Editor Integration

### Monaco Editor

```javascript
// Monaco Editor (VSCode's editor engine)
import * as monaco from 'monaco-editor';

// Register AI completion provider
monaco.languages.registerCompletionItemProvider('python', {
    async provideCompletionItems(model, position) {
        const code = model.getValue();
        const offset = model.getOffsetAt(position);
        
        const response = await fetch('http://localhost:8000/v1/code/complete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: code,
                position: { line: position.lineNumber, character: position.column },
                language: 'python',
                max_tokens: 100
            })
        });
        
        const data = await response.json();
        
        return {
            suggestions: [{
                label: 'AI Suggestion',
                kind: monaco.languages.CompletionItemKind.Snippet,
                insertText: data.code,
                documentation: 'Generated by AI'
            }]
        };
    }
});

// Inline code generation
async function generateCode(editor) {
    const prompt = window.prompt('Describe what you want to generate:');
    if (!prompt) return;
    
    const response = await fetch('http://localhost:8000/v1/code/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            prompt: prompt,
            language: 'python',
            max_tokens: 500
        })
    });
    
    const data = await response.json();
    const position = editor.getPosition();
    editor.executeEdits('ai-generate', [{
        range: new monaco.Range(
            position.lineNumber,
            position.column,
            position.lineNumber,
            position.column
        ),
        text: data.code
    }]);
}
```

### CodeMirror Integration

```javascript
// CodeMirror 6
import { EditorView, keymap } from '@codemirror/view';
import { autocompletion } from '@codemirror/autocomplete';

// AI autocompletion
const aiCompletion = autocompletion({
    override: [
        async (context) => {
            const code = context.state.doc.toString();
            const pos = context.pos;
            
            const response = await fetch('http://localhost:8000/v1/code/complete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: code,
                    position: { offset: pos },
                    language: 'javascript',
                    max_tokens: 100
                })
            });
            
            const data = await response.json();
            
            return {
                from: pos,
                options: [{
                    label: 'AI Suggestion',
                    type: 'text',
                    apply: data.code,
                    detail: 'Generated by AI'
                }]
            };
        }
    ]
});

// Create editor with AI completion
const editor = new EditorView({
    extensions: [
        aiCompletion,
        keymap.of([
            {
                key: 'Ctrl-Space',
                run: (view) => {
                    // Trigger AI completion
                    return true;
                }
            }
        ])
    ],
    parent: document.body
});
```

---

## Vim/Neovim Integration

### Lua Configuration

```lua
-- neovim/lua/ai-completion.lua
local M = {}

M.complete_code = function()
    local bufnr = vim.api.nvim_get_current_buf()
    local lines = vim.api.nvim_buf_get_lines(bufnr, 0, -1, false)
    local code = table.concat(lines, '\n')
    
    local pos = vim.api.nvim_win_get_cursor(0)
    
    local curl = require('plenary.curl')
    local response = curl.post('http://localhost:8000/v1/code/complete', {
        body = vim.fn.json_encode({
            code = code,
            position = { line = pos[1], character = pos[2] },
            language = vim.bo.filetype,
            max_tokens = 100
        }),
        headers = {
            content_type = 'application/json'
        }
    })
    
    local data = vim.fn.json_decode(response.body)
    vim.api.nvim_put({data.code}, 'c', true, true)
end

M.generate_code = function()
    local prompt = vim.fn.input('Generate code: ')
    if prompt == '' then return end
    
    local curl = require('plenary.curl')
    local response = curl.post('http://localhost:8000/v1/code/generate', {
        body = vim.fn.json_encode({
            prompt = prompt,
            language = vim.bo.filetype,
            max_tokens = 500
        }),
        headers = {
            content_type = 'application/json'
        }
    })
    
    local data = vim.fn.json_decode(response.body)
    vim.api.nvim_put(vim.split(data.code, '\n'), 'l', true, true)
end

return M
```

### Key Bindings

```lua
-- init.lua
local ai = require('ai-completion')

vim.keymap.set('n', '<leader>ac', ai.complete_code, { desc = 'AI Complete' })
vim.keymap.set('n', '<leader>ag', ai.generate_code, { desc = 'AI Generate' })
vim.keymap.set('n', '<leader>ar', ai.refactor_code, { desc = 'AI Refactor' })
```

---

## Performance Optimization

### Caching Strategy

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_completion_cached(code_hash, position, language):
    """Cache completions for identical code contexts."""
    return requests.post(
        f"{API_BASE}/v1/code/complete",
        json={
            "code": code,
            "position": position,
            "language": language
        }
    ).json()

def get_completion(code, position, language):
    code_hash = hashlib.md5(code.encode()).hexdigest()
    return get_completion_cached(code_hash, position, language)
```

### Debouncing

```javascript
// Debounce AI requests to avoid overwhelming the API
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Usage
const debouncedCompletion = debounce(async (editor) => {
    const completion = await getAICompletion(editor);
    showSuggestion(completion);
}, 300); // Wait 300ms after user stops typing
```

---

## Testing Your Integration

```python
# test_integration.py
import pytest
import requests

API_BASE = "http://localhost:8000"

def test_code_completion():
    """Test code completion endpoint."""
    response = requests.post(
        f"{API_BASE}/v1/code/complete",
        json={
            "code": "def hello_world():\n    ",
            "position": {"line": 1, "character": 4},
            "language": "python",
            "max_tokens": 50
        }
    )
    
    assert response.status_code == 200
    assert "code" in response.json()
    assert len(response.json()["code"]) > 0

def test_code_generation():
    """Test code generation endpoint."""
    response = requests.post(
        f"{API_BASE}/v1/code/generate",
        json={
            "prompt": "function to add two numbers",
            "language": "python",
            "max_tokens": 100
        }
    )
    
    assert response.status_code == 200
    assert "def" in response.json()["code"]

def test_latency():
    """Test response latency for IDE integration."""
    import time
    
    start = time.time()
    response = requests.post(
        f"{API_BASE}/v1/code/complete",
        json={
            "code": "x = ",
            "language": "python",
            "max_tokens": 20
        }
    )
    latency = (time.time() - start) * 1000
    
    assert latency < 100, f"Latency too high: {latency}ms"
```

---

## Best Practices

1. **Debounce requests** - Don't send a request on every keystroke
2. **Cache aggressively** - Same code context = same completion
3. **Show loading indicators** - Keep users informed
4. **Handle errors gracefully** - Network issues happen
5. **Respect rate limits** - Implement exponential backoff
6. **Monitor performance** - Track P50/P95/P99 latency

---

## Support

For integration help:
- **Documentation**: [Full API Reference](api_reference.md)
- **Discord**: [Join our community](https://discord.gg/your-server)
- **Email**: integrations@yourcompany.com
