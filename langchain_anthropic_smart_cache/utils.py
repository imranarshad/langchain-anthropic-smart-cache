"""Utility classes for token counting and content analysis."""

import re
import json
from typing import Any, Dict, List, Optional, Union
import logging

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


class TokenCounter:
    """Estimates token counts for various content types."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        """Initialize token counter.

        Args:
            model_name: Model name for token counting (uses tiktoken if available).
        """
        self.model_name = model_name
        self._encoder = None

        if TIKTOKEN_AVAILABLE:
            try:
                # Try to get encoder for the model, fallback to gpt-4 for estimation
                try:
                    self._encoder = tiktoken.encoding_for_model("gpt-4")
                except KeyError:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken encoder: {e}")

    def count_tokens(self, content: Any) -> int:
        """Count tokens in content.

        Args:
            content: Content to count tokens for (string, dict, list, etc.)

        Returns:
            Estimated token count.
        """
        if content is None:
            return 0

        # Convert content to string
        if isinstance(content, str):
            text = content
        elif isinstance(content, (dict, list)):
            text = json.dumps(content, ensure_ascii=False)
        else:
            text = str(content)

        if self._encoder and TIKTOKEN_AVAILABLE:
            try:
                return len(self._encoder.encode(text))
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed: {e}, falling back to estimation")

        # Fallback estimation: roughly 4 characters per token
        return max(1, len(text) // 4)

    def count_message_tokens(self, message: Dict[str, Any]) -> int:
        """Count tokens in a message.

        Args:
            message: Message dictionary with role and content.

        Returns:
            Estimated token count.
        """
        total_tokens = 0

        # Count role tokens
        if 'role' in message:
            total_tokens += self.count_tokens(message['role'])

        # Count content tokens
        if 'content' in message:
            content = message['content']
            if isinstance(content, str):
                total_tokens += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text' and 'text' in item:
                            total_tokens += self.count_tokens(item['text'])
                        elif item.get('type') == 'image_url':
                            # Rough estimation for image tokens
                            total_tokens += 765  # Base image token cost
                        else:
                            total_tokens += self.count_tokens(str(item))
                    else:
                        total_tokens += self.count_tokens(str(item))

        # Add small overhead for message formatting
        return total_tokens + 10

    def count_tools_tokens(self, tools: List[Dict[str, Any]]) -> int:
        """Count tokens in tools definition.

        Args:
            tools: List of tool definitions.

        Returns:
            Estimated token count.
        """
        if not tools:
            return 0

        total_tokens = 0
        for tool in tools:
            total_tokens += self.count_tokens(tool)

        # Add overhead for tools formatting
        return total_tokens + len(tools) * 5


class ContentAnalyzer:
    """Analyzes content to determine caching strategies."""

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        """Initialize content analyzer.

        Args:
            token_counter: Token counter instance to use.
        """
        self.token_counter = token_counter or TokenCounter()

    def analyze_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a message for caching decisions.

        Args:
            message: Message to analyze.

        Returns:
            Analysis results with token count, content type, etc.
        """
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        token_count = self.token_counter.count_message_tokens(message)

        # Determine content type
        content_type = 'content'
        if role == 'system':
            content_type = 'system'
        elif role == 'assistant' and self._contains_tool_calls(message):
            content_type = 'tool_result'

        # Analyze content complexity
        complexity = self._analyze_content_complexity(content)

        # Determine caching priority
        priority = self._calculate_priority(content_type, token_count, complexity)

        return {
            'content_type': content_type,
            'token_count': token_count,
            'complexity': complexity,
            'priority': priority,
            'role': role,
            'cacheable': token_count >= 50,  # Minimum threshold for caching
        }

    def analyze_tools(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze tools for caching decisions.

        Args:
            tools: Tools to analyze.

        Returns:
            Analysis results.
        """
        if not tools:
            return {
                'content_type': 'tools',
                'token_count': 0,
                'complexity': 'none',
                'priority': 10,  # Low priority if no tools
                'cacheable': False,
            }

        token_count = self.token_counter.count_tools_tokens(tools)
        complexity = self._analyze_tools_complexity(tools)

        # Tools always get high priority when not cached
        priority = 1

        return {
            'content_type': 'tools',
            'token_count': token_count,
            'complexity': complexity,
            'priority': priority,
            'cacheable': token_count >= 100,  # Tools threshold
            'tool_count': len(tools),
        }

    def _contains_tool_calls(self, message: Dict[str, Any]) -> bool:
        """Check if message contains tool calls."""
        content = message.get('content', '')
        if isinstance(content, str):
            # Simple heuristic for tool results
            return 'tool_call' in content.lower() or 'function_call' in content.lower()
        return False

    def _analyze_content_complexity(self, content: Any) -> str:
        """Analyze content complexity.

        Returns:
            Complexity level: 'low', 'medium', 'high'
        """
        if not content:
            return 'none'

        text = str(content) if not isinstance(content, str) else content

        # Count various complexity indicators
        complexity_score = 0

        # Length factor
        if len(text) > 5000:
            complexity_score += 3
        elif len(text) > 1000:
            complexity_score += 2
        elif len(text) > 200:
            complexity_score += 1

        # Code patterns
        code_patterns = [
            r'```[\s\S]*?```',  # Code blocks
            r'`[^`]+`',         # Inline code
            r'\b(function|class|def|import)\b',  # Programming keywords
            r'[{}[\]();]',      # Programming punctuation
        ]

        for pattern in code_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                complexity_score += 1

        # Structured data
        if any(char in text for char in '{}[]'):
            complexity_score += 1

        # URLs and references
        if re.search(r'https?://|www\.', text):
            complexity_score += 1

        if complexity_score >= 5:
            return 'high'
        elif complexity_score >= 2:
            return 'medium'
        else:
            return 'low'

    def _analyze_tools_complexity(self, tools: List[Dict[str, Any]]) -> str:
        """Analyze tools complexity."""
        if not tools:
            return 'none'

        total_params = 0
        has_complex_schemas = False

        for tool in tools:
            # Count parameters
            function_def = tool.get('function', {})
            parameters = function_def.get('parameters', {})
            properties = parameters.get('properties', {})
            total_params += len(properties)

            # Check for complex schemas
            for prop in properties.values():
                if isinstance(prop, dict) and ('enum' in prop or 'anyOf' in prop or 'oneOf' in prop):
                    has_complex_schemas = True

        if len(tools) > 10 or total_params > 50 or has_complex_schemas:
            return 'high'
        elif len(tools) > 3 or total_params > 15:
            return 'medium'
        else:
            return 'low'

    def _calculate_priority(self, content_type: str, token_count: int, complexity: str) -> int:
        """Calculate caching priority.

        Lower numbers = higher priority.

        Returns:
            Priority score (1-10).
        """
        base_priority = {
            'tools': 1,      # Highest priority for tools
            'system': 2,     # High priority for system prompts
            'content': 3,    # Medium priority for content
            'tool_result': 4, # Lower priority for tool results
        }.get(content_type, 5)

        # Adjust based on token count (more tokens = higher priority)
        if token_count > 5000:
            token_bonus = -1
        elif token_count > 2000:
            token_bonus = 0
        else:
            token_bonus = 1

        # Adjust based on complexity
        complexity_bonus = {
            'high': -1,
            'medium': 0,
            'low': 1,
            'none': 2,
        }.get(complexity, 0)

        return max(1, min(10, base_priority + token_bonus + complexity_bonus))