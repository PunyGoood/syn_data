import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import cast, Literal
import tqdm
from datasets import Dataset, load_dataset
from pathlib import Path
from transformers import HfArgumentParser
import re
import utils as utils
import aiohttp
from typing import List, Dict, Any, Optional
#from utils import N_CORES

LLAMA3 = os.getenv("LLAMA3") is not None

InstructMode = Literal["I->R", "S->C", "C->I", "S->I"]

LANGUAGE_MAP = {

    "python": "Python",

}


def parse_generated_content(content: str, instruct_mode: InstructMode) -> dict | None:
    if instruct_mode == "I->R":
        return dict(response=content.strip())
    elif instruct_mode == "S->C":
        concepts = list(map(str.strip, content.split(",")))
        return dict(concepts=concepts)
    elif instruct_mode == "C->I":
        return dict(instruction=content.strip())
    elif instruct_mode == "S->I":
        return dict(instruction=content.strip())
    else:
        assert False


def build_kwargs(instruct_mode: InstructMode, example: dict) -> dict[str, str]:
    kwargs = dict[str, str]()
    if instruct_mode == "I->R":
        kwargs["instruction"] = example["instruction"]
        # Hack
        # category_index = example["prompt"].rindex("category: ") + len("category: ")
        # category_end = example["prompt"].index("\n", category_index)
        # category = example["prompt"][category_index:category_end].strip()
        # kwargs["category"] = category  # type: ignore
    elif instruct_mode in ["S->C", "S->I"]:
        kwargs["snippet"] = example["seed"]
    elif instruct_mode == "C->I":
        lang = example.get("data_dir", "dummy_key_not_in_example")
        language = LANGUAGE_MAP.get(lang, "Python")
        property = Property.random_exercise(example["concepts"], language=language)
        property_prompt = property.prompt()
        # 45 / 152 are the min/max word lengths in the fewshot examples
        # num_words = random.randint(1000, 1500)
        # property_prompt += f"\nnum_words: {num_words}"
        kwargs["property"] = property_prompt
        # Hack
        kwargs["property_obj"] = property  # type: ignore
    else:
        assert False
    return kwargs


@dataclass(frozen=True)
class Property:
    category: str
    language: str
    concepts: list[str]
    difficulty: str

    @staticmethod
    def random_exercise(concepts: list[str], language: str) -> "Property":
        category = random.choice(
            [
                "function implementation",
                "function implementation",
                "class implementation",
                "program implementation",
            ]
        )
        difficulty = random.choice(["easy", "medium", "hard"])
        return Property(
            category=category,
            language=language,
            concepts=concepts,
            difficulty=difficulty,
        )

    def concepts_prompt(self) -> str:
        return ", ".join(self.concepts)

    def prompt(self) -> str:
        category = f"category: {self.category}"
        language = f"language: {self.language}"
        difficulty = f"difficulty: {self.difficulty}"
        concepts = f"concepts: {self.concepts_prompt()}"
        return "\n".join([category, language, difficulty, concepts])

    def to_json(self) -> dict[str, str | list[str]]:
        return dict(
            category=self.category,
            language=self.language,
            concepts=self.concepts,
            difficulty=self.difficulty,
        )

    @staticmethod
    def from_json(data: dict) -> "Property":
        assert all(
            isinstance(data[key], str) for key in ["category", "language", "difficulty"]
        )
        assert isinstance(data["concepts"], list)
        return Property(
            category=data["category"],
            language=data["language"],
            concepts=data["concepts"],
            difficulty=data["difficulty"],
        )

def parse_property(content: str) -> Property | None:
    content = content.strip()
    lines = content.split("\n")
    if len(lines) != 4:
        return None
    try:
        lines = [line[line.index(":") + 1 :].strip() for line in lines]
    except ValueError:
        return None
    category, language, difficulty, concepts_str = lines
    concepts = list(map(str.strip, concepts_str.split(",")))
    return Property(category, language, concepts, difficulty)

@dataclass(frozen=True)
class Args:
    seed_data_files: list[str] = field(
        metadata={"help": "Path to the seed code snippets"}
    )
    max_new_data: int
    instruct_mode: InstructMode

    use_api: bool = field(default=True)
    
    # 移除 model 和 vllm 相关参数
    # 添加 Claude 相关参数

    claude_model: str = field(default="claude-3-5-sonnet-20240620")
    
    # 保持其他参数不变
    seed_code_start_index: int = field(default=0)
    continue_from: str | None = field(default=None)
    seed: int = field(default=3407)
    temperature: float = field(default=0.7)
    ##待定
    max_output_tokens: int = field(default=1600)
    num_fewshots: int = field(default=8)
    
    # 批处理相关参数
    num_batched_requests: int = field(
        default=1, metadata={"help": "Number of requests to send concurrently"}
    )
    sleep: float | None = field(
        default=None, metadata={"help": "Sleep between requests in seconds"}
    )


@dataclass(frozen=True)
class Example:
    property: Property
    snippet: str
    instruction: str
    response: str
    tests: str

    @staticmethod
    def prefix_template(mode: InstructMode) -> str:
        if mode == "I->R":
            template = (
                "<instruction>\n{instruction}\n</instruction>\n\n<response>\n"
            )
            return template
        elif mode == "S->C":
            return "### Snippet\n{snippet}\n\n### Concepts\n"
        elif mode == "C->I":
            return "### Properties\n{property}\n\n### Task\n"
        elif mode == "S->I":
            return "### Snippet\n{snippet}\n\n### Task\n"
        else:
            assert False

    def prompt(
        self,
        mode: InstructMode,
        return_in_separate: bool = False,
        index: int | None = None,
    ) -> str | tuple[str, str]:
        assert index is None or (mode == "I->R" and LLAMA3)
        if mode == "I->R":
            kwargs = dict(instruction=self.instruction)
            if LLAMA3:
                assert index is not None
                kwargs["index"] = str(index)
                suffix = f"{self.response}\n\n### Tests {index}\n{self.tests}"
            else:
                suffix = (
                    f"{self.response}\n</response>\n\n<tests>\n{self.tests}\n</tests>"
                )
        elif mode == "S->C":
            kwargs = dict(snippet=self.snippet)
            suffix = self.property.concepts_prompt()
        elif mode == "C->I":
            property_prompt = self.property.prompt()
            # num_words = len(self.instruction.split())
            # property_prompt += f"\nnum_words: {num_words}"
            kwargs = dict(property=property_prompt)
            suffix = self.instruction
        elif mode == "S->I":
            kwargs = dict(snippet=self.snippet)
            suffix = self.instruction
        else:
            assert False
        prefix = self.prefix_template(mode).format(**kwargs)
        if return_in_separate:
            return prefix, suffix
        else:
            return prefix + suffix



@dataclass(frozen=True)
class Fewshot:

    sys_s_c: str
    sys_s_i: str

    examples: list[Example]

    def system_prompt(self, mode: InstructMode) -> str:
        attr_name = "sys_" + mode.replace("->", "_").replace("-", "_").lower()
        return getattr(self, attr_name)

    def valid_examples(self, mode: InstructMode) -> list[Example]:
        # if mode in ["E->S", "I->RT", "I->R"]:
        #     return [
        #         example for example in self.examples if example.solution is not None
        #     ]
        return self.examples

    def random_prompt(
        self,
        mode: InstructMode,
        num_fewshots: int,
        prompting_mode: Literal["chat", "completion"],
        **format_args: str,
    ) -> str:
        valid_examples = self.valid_examples(mode)
        assert (
            0 < num_fewshots <= len(valid_examples)
        ), f"{num_fewshots=}, {len(valid_examples)=}"
        # if mode == "I->R":
        #     # Hack
        #     category = format_args["category"]
        #     matching_examples = [
        #         example
        #         for example in valid_examples
        #         if example.property.category == category
        #     ]
        #     assert len(matching_examples) > 0, f"{category=}"
        #     matching_example = random.choice(matching_examples)
        #     rest_of_examples = [
        #         example for example in valid_examples if example is not matching_example
        #     ]
        #     assert len(rest_of_examples) == len(self.examples) - 1
        #     examples = [matching_example] + random.sample(
        #         rest_of_examples, k=num_fewshots - 1
        #     )
        #     random.shuffle(examples)
        # else:
        examples = random.sample(valid_examples, k=num_fewshots)
        assert len(examples) == num_fewshots

        body = "\n\n".join(
            f"## Example {idx + 1}\n{example.prompt(mode, index=idx + 1 if LLAMA3 and mode == 'I->R' else None)}"
            for idx, example in enumerate(examples)
        )
        # content = f"{self.system_prompt}\n\n{body}"
        prefix_template = Example.prefix_template(mode)
        if mode == "I->R" and LLAMA3:
            format_args["index"] = str(len(examples) + 1)
        prefix = f"## Example {len(examples) + 1}\n" + prefix_template.format(
            **format_args
        )
        system_prompt = self.system_prompt(mode)
        full_prompt = f"{system_prompt}\n\n{body}\n\n{prefix}"
        assert prompting_mode == "completion", "Only completion is supported for now"
        return full_prompt


def get_ossinstruct_fewshots() -> Fewshot:
    content = Path("prompts/fewshot.txt").read_text().strip()
    # split according to the example tag, but exclude the tag. Strip each string.
    splits = re.split(r"### Example \d+", content)
    system_prompt = splits[0].strip()
    sys_pattern = r"### System: S->C|### System: S->I"
    _, s_c, s_i = list(map(str.strip, re.split(sys_pattern, system_prompt)))
    # system_prompt = re.split(r"### System: Instruction", system_prompt)[1]
    # instruction_system_prompt, response_system_prompt = system_prompt.split(
    #     "### System: Response"
    # )
    # instruction_system_prompt = instruction_system_prompt.strip()
    # response_system_prompt = response_system_prompt.strip()
    examples_str = [example.strip() for example in splits[1:]]
    assert len(examples_str) == 21, len(examples_str)
    examples = list[Example]()
    for example_str in examples_str:
        pattern = (
            r"\[Code\]\n|\[Property\]\n|\[Instruction\]\n|\[Response\]\n|\[Tests\]\n"
        )
        _, snippet, property, instruction, response, tests = re.split(
            pattern, example_str
        )
        snippet = snippet.rstrip()
        property = parse_property(property)
        assert property is not None
        instruction = instruction.strip()
        response = response.strip()
        tests = tests.strip()
        example = Example(property, snippet, instruction, response, tests)
        examples.append(example)
    # if args.external_data is not None:
    #     examples.extend(external_examples)
    return Fewshot(
        sys_s_c=s_c,
        sys_s_i=s_i,
        examples=examples,
    )

class Messages:
    def __init__(self, client):
        self.client = client

    async def create(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-5-sonnet-20240620",
        max_tokens: int = 1600,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        if not self.client.session:
            self.client.session = aiohttp.ClientSession()
            
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        async with self.client.session.post(self.client.url, json=data, headers=self.client.headers) as response:
            response.raise_for_status()
            return await response.json()

class ClaudeClient:
    def __init__(self):
        self.url = "http://35.220.164.252:3888/v1/chat/completions"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': os.getenv("CLAUDE_API_KEY")
        }
        self.session: Optional[aiohttp.ClientSession] = None
        self.messages = Messages(self)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

async def create_claude_client() -> ClaudeClient:
    return ClaudeClient()

async def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    
    # 创建 Claude 客户端
    claude = await create_claude_client()
    
    # 加载数据集部分保持不变
    raw_dataset: Dataset = load_dataset(
        "json",
        data_files=args.seed_data_files,
        split="train",
        num_proc=utils.N_CORES,
    )

    id_key = "seed"
    
    # 数据处理部分保持不变
    start_index = args.seed_code_start_index
    end_index = min(start_index + args.max_new_data, len(raw_dataset))
    raw_dataset = raw_dataset.select(range(start_index, end_index))
    dataset = raw_dataset.to_list()

    assert args.prompting_mode == "completion", "Only completion is supported for now"
    fewshot = get_ossinstruct_fewshots()
    data_fingerprint = args.fingerprint(fewshot)
    timestamp = utils.timestamp()

    if args.continue_from is not None:
        if os.getenv("IGNORE_FINGERPRINT") is None:
            assert (
                data_fingerprint in args.continue_from
            ), f"Fingerprint mismatch: {data_fingerprint}"
        assert f"-{start_index}-" in args.continue_from, "Index mismatch"
        old_path = Path(args.continue_from)
        assert old_path.exists()
        old_data = utils.read_jsonl(old_path)
        assert len(old_data) > 0
        last_seed = old_data[-1][id_key]
        # Find seed
        seed_index = next(
            idx for idx, d in enumerate(dataset) if d[id_key] == last_seed
        )
        n_skipped = seed_index + 1
        # n_skipped = last_index - start_index + 1
        print(f"Continuing from {old_path} with {n_skipped} seed snippets skipped")
        f_out = old_path.open("a")
    else:
        tag = "" if args.tag == "" else f"-{args.tag}"
        mode_str = args.instruct_mode.replace("->", "_").lower()
        path = Path(
            args.save_dir,
            f"data{tag}-{mode_str}-{data_fingerprint}-{start_index}-{timestamp}.jsonl",
        )
        assert not path.exists()
        f_out = path.open("w")
        print("Saving to", path)
        n_skipped = 0
    dataset = dataset[n_skipped:]
    chunked_dataset = list(
        utils.chunked(dataset, n=args.num_batched_requests)
    )
    pbar = tqdm(chunked_dataset)
    n_succeeded = 0

    
    
    # 主要修改请求生成部分
    for chunk_index, examples in enumerate(pbar):
        effective_index = (
            chunk_index * args.num_batched_requests + start_index + n_skipped
        )
        print("Effective index:", effective_index)

        if chunk_index > 0 and args.sleep is not None:
            print(f"Sleeping for {args.sleep} seconds...")
            time.sleep(args.sleep)
        
            
        all_prompts = []
        for index, example in enumerate(examples):
            seed = args.seed + chunk_index + index
            random.seed(seed)
            kwargs = build_kwargs(args.instruct_mode, example)
            prompt = fewshot.random_prompt(
                args.instruct_mode,
                args.num_fewshots,
                prompting_mode="completion",
                **kwargs,
            ).rstrip()
            all_prompts.append(prompt)
            
        
        
        async def process_prompt(prompt: str):
            try:
                raw_response = await claude.messages.create(
                    model=args.claude_model,
                    max_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                response = raw_response['choices'][0]['message']['content']
                return response
            except Exception as e:
                print(f"Error in Claude API request: {e}")
                return e
                
        # 创建并发任务
        tasks = [process_prompt(prompt) for prompt in all_prompts]
        responses = await asyncio.gather(*tasks)
        
        # 处理响应
        for prompt, example, response in zip(all_prompts, examples, responses):
            if isinstance(response, Exception):
                print("Exception when generating response:", response)
                continue
                
            # 解析 Claude 响应
            content = response.content[0].text
            parsing_result = parse_generated_content(content, args.instruct_mode)
            
            if parsing_result is None:
                print("[WRONG FORMAT]")
                print("@@@Prompt", prompt, sep="\n", end="\n\n")
                print("@@@Response", content, sep="\n", end="\n\n")
                continue
                
            # 保存结果
            data = dict(
                prompt=prompt,
                **{k: v for k, v in example.items() if k not in ["prompt"]},
                **parsing_result
            )
            
            print(
                "@@@Prefix",
                prompt,
                f"@@@Generation",
                content,
                sep="\n",
                end="\n\n",
            )
            
            n_succeeded += 1
            f_out.write(json.dumps(data) + "\n")
            f_out.flush()
            
        pbar.set_description(f"Success ratio: {n_succeeded} / {(chunk_index + 1) * len(examples)}")

if __name__ == "__main__":
    asyncio.run(main())