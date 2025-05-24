import utils
from datasets import load_dataset
from datasets import Dataset

start_index = 0
max_new_data = 1000000


raw_dataset: Dataset = load_dataset(
        "json",
        data_files="./data/ex_seeds.jsonl",
        split="train",
        num_proc=utils.N_CORES,
    )



end_index = min(start_index + max_new_data, len(raw_dataset))
raw_dataset = raw_dataset.select(range(start_index, end_index))
dataset = raw_dataset.to_list()
utils.write_jsonl("./dataset.jsonl", dataset)

print(len(dataset))


chunked_dataset = list(
    utils.chunked(dataset, n=5)
)
utils.write_jsonl("./chunked_dataset.jsonl", chunked_dataset)

for i, chunk in enumerate(chunked_dataset):
    utils.write_jsonl(f"./chunked_dataset_{i}.jsonl", chunk)

    

