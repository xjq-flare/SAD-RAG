from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser


class QuestionAnswering:
    system_template = """你是一个专业的知识问答助手。你的任务是根据提供的“参考资料”来回答用户的“查询语句”。

### 任务要求
* 准确性：仅使用参考资料中提供的信息进行回答。
    如果参考资料中没有相关信息，请明确回答“无法根据现有资料回答此问题”。
* 逻辑性：回答应当条理清晰，直接响应用户的核心需求。

### 输出格式
以纯文本的形式输出答案。
"""

    human_template = """以下是执行任务所需的信息：

### 查询语句
{user_query}

### 参考资料
{search_results_or_claims}

请根据上述资料回答用户问题。"""

    def __init__(self, model: BaseChatModel):

        print(__name__)
        self.model = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("human", self.human_template),
        ])

        self.chain = self.prompt | self.model.bind(max_tokens=8192)

    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:

        formatted_lines = []
        for idx, doc in enumerate(documents):
            doc_id = idx + 1
            title = doc["title"]
            content = doc["content"]
            formatted_lines.append(f"文档{doc_id}:\n标题: {title}\n内容: {content}\n")
        return "\n".join(formatted_lines)

    def _format_claims(self, claims: List[Dict[str, Any]]) -> str:

        formatted_lines = []
        for idx, claim in enumerate(claims):
            claim_id = idx + 1
            content = claim["content"]
            evidence = claim["evidence"]
            formatted_lines.append(f"主张{claim_id}:\n内容: {content}\n原文: {evidence}\n")
        return "\n".join(formatted_lines)

    def generate_from_documents(self, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:

        formatted_docs = self._format_documents(documents)
        
        result = self.chain.invoke({
            "user_query": query,
            "search_results_or_claims": formatted_docs
        })
        
        return result.content

    async def generate_from_documents_async(self, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:

        formatted_docs = self._format_documents(documents)
        
        result = await self.chain.ainvoke({
            "user_query": query,
            "search_results_or_claims": formatted_docs
        })
        
        return result.content

    def generate_from_claims(self, claims: List[Dict[str, Any]], query: str) -> Dict[str, Any]:

        formatted_claims = self._format_claims(claims)
        
        result = self.chain.invoke({
            "user_query": query,
            "search_results_or_claims": formatted_claims
        })
        
        return result.content

    async def generate_from_claims_async(self, claims: List[Dict[str, Any]], query: str) -> Dict[str, Any]:

        formatted_claims = self._format_claims(claims)
        
        result = await self.chain.ainvoke({
            "user_query": query,
            "search_results_or_claims": formatted_claims
        })
        
        return result.content
