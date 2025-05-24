import requests
import os
import aiohttp
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


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

# 使用示例
async def main():

    claude = await create_claude_client()
    
    response = await claude.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1700,
        temperature=0.7,
        messages=[{"role": "user", "content": "Hello, world!"}]
    )
    content = response['choices'][0]['message']['content']
    print(f"Response: {content}")
    
    if claude.session:
        await claude.session.close()
    
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

