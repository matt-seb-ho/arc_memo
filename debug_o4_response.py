"""Debug script to inspect raw o4-mini responses"""
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "llm_wrapper"))

from llmplus import GenerationConfig, LLMClient, Provider
from concept_mem.constants import REPO_ROOT

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

SYSTEM_PROMPT = """You are solving AIME (American Invitational Mathematics Examination) problems.
Think through the problem step by step, then provide your final answer in the format:
Final Answer: [number]
where [number] is an integer between 0 and 999."""

# Problem 2019-II-1 that fails
PROBLEM_TEXT = """Two different points, $C$ and $D$ , lie on the same side of line $AB$ so that $\\triangle ABC$ and $\\triangle BAD$ are congruent with $AB=9,BC=AD=10$ , and $CA=DB=17$ . The intersection of these two triangular regions has area $\\tfrac{m}{n}$ , where $m$ and $n$ are relatively prime positive integers. Find $m+n$ ."""

PROMPT = f"""Solve this AIME problem:

{PROBLEM_TEXT}

Show your reasoning, then provide your final answer in the format:
Final Answer: [number]"""


async def test_raw_response():
    """Test and inspect raw API response"""
    
    provider = Provider.OPENAI
    model = "o4-mini-2025-04-16"
    
    llm_client = LLMClient(
        provider=provider,
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=REPO_ROOT / ".env"
    )
    
    llm_client.system_prompt = SYSTEM_PROMPT
    
    gen_cfg = GenerationConfig(
        n=1,
        max_tokens=4096,
        ignore_cache=True  # Force fresh call
    )
    
    print("="*80)
    print("Testing o4-mini on 2019-II-1 (problematic case)")
    print("="*80)
    print(f"\nPrompt (first 200 chars): {PROMPT[:200]}...\n")
    
    # Get response
    try:
        # Directly call the async client to get raw response object
        from llmplus.client import logging as client_logging
        client_logging.getLogger().setLevel(logging.DEBUG)
        
        responses = await llm_client.async_generate(
            prompt=PROMPT,
            model=model,
            gen_cfg=gen_cfg
        )
        
        print("\n" + "="*80)
        print("RESPONSE ANALYSIS")
        print("="*80)
        print(f"\nNumber of responses: {len(responses)}")
        
        for i, response in enumerate(responses):
            print(f"\n--- Response {i+1} ---")
            print(f"Type: {type(response)}")
            print(f"Is None: {response is None}")
            print(f"Is empty string: {response == ''}")
            print(f"Length: {len(response) if response else 0}")
            
            if response and len(response) > 0:
                print(f"\nContent (first 500 chars):\n{response[:500]}")
                print(f"\nContent (last 200 chars):\n{response[-200:]}")
            else:
                print(f"\nContent: '{response}'")
                print("\n⚠️  EMPTY RESPONSE DETECTED")
        
        # Check cache
        print("\n" + "="*80)
        print("CHECKING CACHE")
        print("="*80)
        cache_stats = llm_client.get_token_usage_dict()
        print(json.dumps(cache_stats, indent=2))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_raw_response())

