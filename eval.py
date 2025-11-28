import json
import glob
import os

class PoetryEvalPipeline:
    def __init__(self):
        pass

    def construct_prompt(self, task_data):
        """
        根据任务数据构建给 LLM 的完整 Prompt。
        格式要求：
        Prompt + 【Demo】 + Hint + Output Instruction + 选项 + 答案：
        """
        base_prompt = task_data['prompt']
        demo = task_data.get('demo')
        hint = task_data.get('hint', '')
        choices = task_data.get('choices', [])
        task_type = task_data.get('type')

        # 1. 基础 Prompt 和 Demo
        full_text = base_prompt
        if demo:
            full_text += f"【{demo}】\n"
        else:
            full_text += "\n"

        # 2. 构造选项字符串
        choices_str = ""
        if task_type == 'multiple_choice':
            choices_str = "选项：\n"
            for idx, c in enumerate(choices, 1):
                choices_str += f"{idx}. {c}\n"
        elif task_type == 'sorting':
            # 排序题通常输出打乱的句子列表供参考，虽然逻辑上用户需要排序
            # 这里的 choices 字段存的是 shuffled sentences
            choices_str = "选项：\n"
            for idx, c in enumerate(choices, 1):
                choices_str += f"{idx}. {c}\n"

        full_text += choices_str

        # 3. Hint (如果有)
        if hint:
            full_text += f"提示：{hint}\n"

        # 4. 输出提示 (Output Instruction)
        if task_type == 'multiple_choice':
            full_text += "直接给出正确答案的序号（1-4）。\n"
        elif task_type == 'sorting':
            # 简单判断是绝句(4句)还是律诗(8句)来给示例
            example = "1324"
            if len(choices) > 4:
                example = "15263748"
            full_text += f"直接给出正确的排序序列，例：{example}。\n"
        
        # 5. 答案前缀
        full_text += "答案："

        return full_text

    def get_correct_answer(self, task_data):
        """
        计算正确答案的标准输出格式。
        单选题：返回字符串序号 '1', '2', '3', '4'
        排序题：返回排序后的序号序列字符串，如 '1324' (基于 choices 的 1-based index)
        """
        task_type = task_data.get('type')
        goal = task_data['goal']
        choices = task_data['choices']

        if task_type == 'multiple_choice':
            try:
                # 找到 goal 在 choices 中的索引 (1-based)
                # 注意：goal 可能是字符串，choices 是字符串列表
                idx = choices.index(goal) + 1
                return str(idx)
            except ValueError:
                return "Error: Goal not in choices"
        
        elif task_type == 'sorting':
            # goal 是正确的句子列表
            # choices 是打乱的句子列表
            # 我们需要找出 goal 中每个句子在 choices 中的位置(1-based)
            # 例如 choices=[A, B, C, D], goal=[A, C, B, D] -> 1324
            
            indices = []
            # 复制一份 choices 以免修改原数据，或处理重复句子问题(诗词中少见)
            temp_choices = list(choices)
            
            for sentence in goal:
                try:
                    idx = temp_choices.index(sentence) + 1
                    indices.append(str(idx))
                    # 如果有重复句子，需要把已找到的设为不可用，防止重复索引
                    # 这里暂假设句子唯一
                except ValueError:
                    return "Error: Goal sentence not in choices"
            
            return "".join(indices)

        return ""

    def check_is_correct(self, response, correct_answer):
        """
        判断模型输出是否正确。
        简单去除非数字字符后比较。
        """
        # 清理 response，只保留数字
        # 例如 "答案是1" -> "1", "1." -> "1"
        clean_resp = "".join([c for c in str(response) if c.isdigit()])
        clean_corr = "".join([c for c in str(correct_answer) if c.isdigit()])
        
        return clean_resp == clean_corr

# ================= 测试/演示代码 =================

if __name__ == '__main__':
    DATA_DIR = './data/benchmark_dataset'
    pipeline = PoetryEvalPipeline()

    print("\n--- 遍历数据集进行验证 ---")
    jsonl_files = glob.glob(os.path.join(DATA_DIR, '*.jsonl'))

    if not jsonl_files:
        print(f"No .jsonl files found in {DATA_DIR}. Skipping dataset validation.")
    else:
        for file_path in jsonl_files:
            print(f"\nProcessing file: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as f:
                # 只读取每文件的第一条数据进行验证
                line = f.readline()
                if not line:
                    print("  File is empty. Skipping.")
                    continue
                
                try:
                    task_data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  Error decoding JSON from {os.path.basename(file_path)}: {e}. Skipping.")
                    continue

                task_type = task_data.get('type')

                # 拼接prompt
                prompt = pipeline.construct_prompt(task_data)
                
                # 模拟模型API响应
                mock_model_response = ""
                if task_type == 'multiple_choice':
                    mock_model_response = "1" # 假设模型总是返回选项1
                elif task_type == 'sorting':
                    mock_model_response = "1234" # 假设模型总是返回1234的排序
                else:
                    print(f"  Unknown task type '{task_type}'. Skipping mock response generation.")
                    continue

                # 获取真实答案
                correct_answer = pipeline.get_correct_answer(task_data)
                
                # 检查模型输出是否正确
                is_correct = pipeline.check_is_correct(mock_model_response, correct_answer)

                print(f"  Task Type: {task_type}")
                print(f"  Prompt: {prompt}")
                print(f"  Mock Model Response: '{mock_model_response}'")
                print(f"  Correct Answer: '{correct_answer}'")
                print(f"  Is Correct (Mock vs Truth): {is_correct}")
