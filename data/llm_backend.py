"""
Stub showing where real LLM calls go. Not used by default -- generate_data.py
currently stands in four "generator voices" (GENERATORS in generate_data.py)
with simple text transforms instead of real API calls, since this checkpoint
has no API access. To wire in real generation, replace the GENERATORS dict's
lambdas with calls to the functions below (one real model per generator
voice), and replace _render()'s per-line formatting with a single call that
generates the whole next turn given the conversation so far -- the template
line banks (rapport/trust/urgency/authority/ask, verify/request) become the
prompts rather than the literal text.

Keep the same signature: given a persona/system prompt and the conversation
so far, return the next turn as a string.
"""

def generate_next_turn_openai(system_prompt: str, history: list[str], client=None, model="gpt-4o-mini") -> str:
    """Example wiring for the OpenAI API. Fill in your own client/model."""
    if client is None:
        raise RuntimeError("pass an initialized OpenAI client")
    messages = [{"role": "system", "content": system_prompt}]
    for i, turn in enumerate(history):
        role = "assistant" if i % 2 == 0 else "user"
        messages.append({"role": role, "content": turn})
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


def generate_next_turn_anthropic(system_prompt: str, history: list[str], client=None, model="claude-3-5-haiku-latest") -> str:
    """Example wiring for the Anthropic API."""
    if client is None:
        raise RuntimeError("pass an initialized Anthropic client")
    messages = []
    for i, turn in enumerate(history):
        role = "assistant" if i % 2 == 0 else "user"
        messages.append({"role": role, "content": turn})
    resp = client.messages.create(model=model, system=system_prompt, max_tokens=200, messages=messages)
    return resp.content[0].text
