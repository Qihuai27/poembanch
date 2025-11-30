"""
Prompt construction for evaluation tasks.
"""
from typing import Optional, List, Tuple
from .dataset import TaskSample


class PromptBuilder:
    """Build prompts for different task types."""

    def construct_prompt(self, sample: TaskSample) -> str:
        """
        Construct a complete prompt for the given sample.

        Format: Prompt + 【Demo】 + Hint + Output Instruction + 选项 + 答案：

        Args:
            sample: TaskSample object.

        Returns:
            Complete prompt string.
        """
        # 1. Base prompt and demo
        full_text = sample.prompt
        if sample.demo:
            full_text += f"【{sample.demo}】\n"
        else:
            full_text += "\n"

        # 2. Construct choices string
        choices_str = "选项：\n"
        for idx, choice in enumerate(sample.choices, 1):
            choices_str += f"{idx}. {choice}\n"
        full_text += choices_str

        # 3. Hint (if any)
        if sample.hint:
            full_text += f"提示：{sample.hint}\n"

        # 4. Output instruction
        if sample.task_type == 'multiple_choice':
            full_text += "直接给出正确答案的序号（1-4）。\n"
        elif sample.task_type == 'sorting':
            # Determine example based on number of choices
            example = "1324" if len(sample.choices) <= 4 else "15263748"
            full_text += f"直接给出正确的排序序列，例：{example}。\n"

        # 5. Answer prefix
        full_text += "答案："

        return full_text

    def construct_prompt_cot(self, sample: TaskSample) -> str:
        """
        Construct a chain-of-thought prompt.

        Args:
            sample: TaskSample object.

        Returns:
            CoT prompt string.
        """
        # Base prompt and demo
        full_text = sample.prompt
        if sample.demo:
            full_text += f"【{sample.demo}】\n"
        else:
            full_text += "\n"

        # Choices
        choices_str = "选项：\n"
        for idx, choice in enumerate(sample.choices, 1):
            choices_str += f"{idx}. {choice}\n"
        full_text += choices_str

        # Hint
        if sample.hint:
            full_text += f"提示：{sample.hint}\n"

        # CoT instruction
        full_text += "\n请逐步分析后给出答案。\n"

        if sample.task_type == 'multiple_choice':
            full_text += "分析：\n"
        elif sample.task_type == 'sorting':
            full_text += "分析：\n"

        return full_text


class ResponseParser:
    """Parse model responses to extract answers."""

    def parse_and_validate(self, response: str, sample: TaskSample) -> Tuple[str, bool, bool]:
        """
        Parse response and validate format.

        Args:
            response: Model's raw response.
            sample: Original task sample.

        Returns:
            Tuple of (parsed_answer, is_valid_format, is_correct)
        """
        if sample.task_type == 'multiple_choice':
            return self._parse_multiple_choice_validated(response, sample)
        elif sample.task_type == 'sorting':
            return self._parse_sorting_validated(response, sample)
        else:
            return "", False, False

    def _parse_multiple_choice_validated(
        self, response: str, sample: TaskSample
    ) -> Tuple[str, bool, bool]:
        """
        Parse and validate multiple choice response.

        Only accepts:
        1. First 3 tokens contain a standalone digit (开头直接作答)
        2. Last 3 tokens contain digit + terminator (结尾总结)
        3. Number appearing after "答案" keyword

        Returns:
            Tuple of (parsed_answer, is_valid_format, is_correct)
        """
        import re

        response = response.strip()
        num_choices = len(sample.choices)
        correct_answer = sample.correct_answer
        valid_digits = '[1-' + str(num_choices) + ']'

        # Strategy 1: Check first 3 tokens for standalone digit (开头直接作答)
        # Only match if digit is standalone (not part of "1. 选项内容" pattern)
        first_chars = response[:3] if len(response) >= 3 else response
        # Match: digit at start, optionally followed by terminator, then end or whitespace
        # Valid: "1", "2。", "3\n"
        # Invalid: "1. 蝶恋花" (digit followed by content)
        match = re.match(r'^(' + valid_digits + r')([。.，,、：:\s]?)$', first_chars.strip())
        if match:
            choice = match.group(1)
            is_correct = choice == correct_answer
            return choice, True, is_correct

        # Also check if response starts with just a digit on its own line
        first_line = response.split('\n')[0].strip()
        if re.match(r'^' + valid_digits + r'[。.，,、：:\s]*$', first_line):
            choice = first_line[0]
            is_correct = choice == correct_answer
            return choice, True, is_correct

        # Strategy 2: Check last 3 tokens for digit + terminator (结尾总结)
        # Valid patterns: "1。", "2.", "3", "是1", "为2" at the very end
        last_chars = response[-3:] if len(response) >= 3 else response
        # Match digit followed by optional terminator at the end
        match = re.search(r'(' + valid_digits + r')[。.，,、：:\s]*$', last_chars)
        if match:
            choice = match.group(1)
            is_correct = choice == correct_answer
            return choice, True, is_correct

        # Strategy 3: Look for "答案" keyword followed by a number
        # Patterns: "答案：X", "答案是X", "答案为X", "答案X"
        # Use findall and take the LAST match to handle CoT responses
        patterns = [
            r'答案\s*[:：]\s*(' + valid_digits + r')\b',
            r'答案\s*[是为]\s*[:：]?\s*(' + valid_digits + r')\b',
            r'答案\s*(' + valid_digits + r')\b',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                choice = matches[-1]  # Take the LAST match
                is_correct = choice == correct_answer
                return choice, True, is_correct

        # No valid answer found - instruction not followed
        return "", False, False

    def _parse_sorting_validated(
        self, response: str, sample: TaskSample
    ) -> Tuple[str, bool, bool]:
        """
        Parse and validate sorting response.

        Returns:
            Tuple of (parsed_answer, is_valid_format, is_correct)
        """
        import re

        response = response.strip()
        num_items = len(sample.choices)
        correct_answer = sample.correct_answer  # e.g., "1324"

        # Strategy 1: Check for clean sequence at start
        first_line = response.split('\n')[0].strip()

        # Try consecutive digits (e.g., "1324")
        if first_line.isdigit() and len(first_line) == num_items:
            numbers = [int(c) for c in first_line]
            if set(numbers) == set(range(1, num_items + 1)):
                answer = first_line
                is_correct = answer == correct_answer
                return answer, True, is_correct

        # Strategy 2: Try with separators (e.g., "1,3,2,4" or "1、3、2、4")
        for sep in [',', '，', '、', ' ', '-']:
            if sep in first_line:
                parts = [p.strip() for p in first_line.split(sep)]
                if len(parts) == num_items:
                    if all(p.isdigit() and len(p) == 1 for p in parts):
                        numbers = [int(p) for p in parts]
                        if set(numbers) == set(range(1, num_items + 1)):
                            answer = "".join(parts)
                            is_correct = answer == correct_answer
                            return answer, True, is_correct

        # Strategy 3: Look for the LAST "答案" keyword followed by sequence
        # This prioritizes the final answer in CoT responses
        patterns = [
            r'答案[:：]\s*([1-' + str(num_items) + r']{' + str(num_items) + r'})',
            r'答案[:：]\s*([1-' + str(num_items) + r'](?:[,，、\s-]*[1-' + str(num_items) + r']){' + str(num_items - 1) + r'})',
        ]

        for pattern in patterns:
            # Use findall to get all matches, then take the last one
            matches = re.findall(pattern, response)
            if matches:
                seq = matches[-1]  # Take the LAST match
                # Clean separators
                digits = re.findall(r'\d', seq)
                if len(digits) == num_items:
                    numbers = [int(d) for d in digits]
                    if set(numbers) == set(range(1, num_items + 1)):
                        answer = "".join(digits)
                        is_correct = answer == correct_answer
                        return answer, True, is_correct

        # Strategy 4: Extract LAST num_items valid digits with proximity constraint
        # Only accept digits that are close together (max 3 chars apart)
        # This avoids matching scattered digits in thinking process
        digit_pattern = r'[1-' + str(num_items) + r']'
        matches = list(re.finditer(digit_pattern, response))

        if len(matches) >= num_items:
            # Try to find a valid sequence from the END of response
            # Check sequences starting from the last possible position
            for start_idx in range(len(matches) - num_items, -1, -1):
                candidate_matches = matches[start_idx:start_idx + num_items]

                # Check proximity: max gap between adjacent digits <= 3 characters
                valid_proximity = True
                for i in range(len(candidate_matches) - 1):
                    gap = candidate_matches[i + 1].start() - candidate_matches[i].end()
                    if gap > 3:
                        valid_proximity = False
                        break

                if valid_proximity:
                    digits = [m.group() for m in candidate_matches]
                    numbers = [int(d) for d in digits]
                    if set(numbers) == set(range(1, num_items + 1)):
                        answer = "".join(digits)
                        is_correct = answer == correct_answer
                        return answer, True, is_correct

        # No valid sequence found - instruction not followed
        return "", False, False

    def parse_response(self, response: str, sample: TaskSample) -> str:
        """
        Parse response to extract answer (legacy method).

        Args:
            response: Model's raw response.
            sample: Original task sample.

        Returns:
            Extracted answer string (digits only).
        """
        parsed, _, _ = self.parse_and_validate(response, sample)
        return parsed

    def check_correct(self, response: str, sample: TaskSample) -> bool:
        """
        Check if response is correct (legacy method).

        Args:
            response: Model's raw response.
            sample: Original task sample.

        Returns:
            True if correct, False otherwise.
        """
        _, is_valid, is_correct = self.parse_and_validate(response, sample)
        return is_valid and is_correct

    def check_valid_format(self, response: str, sample: TaskSample) -> bool:
        """
        Check if response follows the expected format.

        Args:
            response: Model's raw response.
            sample: Original task sample.

        Returns:
            True if format is valid (instruction followed), False otherwise.
        """
        _, is_valid, _ = self.parse_and_validate(response, sample)
        return is_valid
