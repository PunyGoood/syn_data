## Seeds -> concepts
```
python syn.py \

    --instruct_mode "S->C" \
    --seed_data_files /data/ex_seeds.jsonl \
    --tag concept_gen \
    --temperature 0.7 \
    --seed_code_start_index 0 \
    --num_fewshots 8 \
    --num_batched_requests 2000 \
    --num_sample_per_request 1

```
## Concepts -> Instructions

```

python syn.py \
    --instruct_mode "C->I" \
    --seed_data_files /path/to/concepts.jsonl \
    --tag instruction_gen \
    --temperature 0.7 \
    --seed_code_start_index 0 \
    --num_fewshots 8 \
    --num_sample_per_request 1 \
    --num_batched_request 2000
```

## Instructions -> dafny codes 
```
python syn.py \
    --instruct_mode "I->R" \
    --seed_data_files path/to/instructions.jsonl  \
    --tag response_gen \
    --temperature 0.7
    --seed_code_start_index 0 \
    --num_fewshots 1 \
    --num_batched_request 500 \
    --num_sample_per_request 10 \

```

maybe i can insert the source infos into somewhere

yet need to debug and execute the dafny code in a sandbox?(i will connect xuxu to seek for the best practices)