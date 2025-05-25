import json
import re

def fix_jsonl_format(input_file, output_file):
    print(f"Processing {input_file} -> {output_file}")
    
    # 读取原始文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式找到完整的 JSON 对象
    # 使用正则表达式匹配从 { 开始到 } 结束的完整 JSON 对象
    json_pattern = r'({[^{]*?})'
    parts = re.findall(json_pattern, content)
    
    fixed_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, part in enumerate(parts):
            try:
                # 验证 JSON 格式
                json_obj = json.loads(part)
                # 写入格式化的 JSON，确保每个对象独占一行
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                fixed_count += 1
            except json.JSONDecodeError as e:
                print(f"Error in part {i+1}: {e}")
                print(f"Problematic JSON: {part[:50]}...")  # 打印前50个字符用于调试
    
    print(f"Successfully fixed {fixed_count} JSON objects")

if __name__ == "__main__":
    input_file = "./results/data-response_gen-i_r-96ac0-0-20250525_124643.jsonl"  # 替换为你的输入文件路径
    output_file = "./results/data-response_gen-i_r-96ac0-0-20250525_124643-fixed.jsonl"    # 替换为你想要的输出文件路径
    fix_jsonl_format(input_file, output_file)