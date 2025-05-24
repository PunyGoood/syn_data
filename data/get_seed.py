import pandas as pd
import json
# 读取parquet文件
df = pd.read_parquet('ex_data.parquet')


# print(df.info())
# print(df.head())


selected_df = df[['id', 'sha1','seed']].head(20)

records = selected_df.to_dict('records')

with open('ex_seeds.jsonl', 'w', encoding='utf-8') as f:
    for record in records:
        json_line = json.dumps(record, ensure_ascii=False)
        f.write(json_line + '\n')

print(f"已保存 {len(records)} 条记录到 ex_seeds.jsonl")


print("\n前3条记录示例：")
for record in records[:3]:
    print(record)