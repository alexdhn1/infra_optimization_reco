from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
resp = llm.invoke("Reply with ONLY this JSON: {\"ok\":true}")
print(resp.content)