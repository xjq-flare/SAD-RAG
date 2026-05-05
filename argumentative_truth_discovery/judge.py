import logging
import asyncio
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

try:
    from .utils import parse_think_and_json, ReturnData
except ImportError:
    from utils import parse_think_and_json, ReturnData


logger = logging.getLogger(__name__)

class Judge:
    system_template = """# Role
你是一位公正、冷静且逻辑极度严密的首席裁决官。你负责审核一场关于多文档争议点的辩论，并根据各方表现及证据质量判定最终哪一方获胜。

# Input Data
你将接收到以下信息：
1. 观点簇概览：支持不同主张的原始文档（包含Content、URL、Rank三项，其中 rank 为查询时搜索引擎的原始排名）。
2. 第一阶段：各方提出的核心证据及对他方证据的质询。
3. 第二阶段：各方对质询的回应（包含承认弱点后的收缩策略）及最终结案陈词。

# Judging Criteria (裁决准则)
1. 证据效力 (Weight of Evidence)：优先考虑权威 URL (.gov, .edu, 顶刊) 。
2. 如一方文档全部来自用户生成内容（UGC）网站且内容相似，更为可疑。
3. 各类百科类网站（如百度百科、维基百科等）由用户生成内容，视为可投毒网站，可信度较低，**不属于权威来源**。
4. 存活率 (Survivability)：证据在经历质询和反驳后，如果依然逻辑自洽、未被推翻，则视为“站得住脚”。
5. 反驳逻辑 (Rebuttal Quality)：有效的反驳应基于事实（如指明数据来源），而非单纯的情绪宣泄。

# Task Instructions
请按照以下步骤进行裁决：

## Step 1: 深度思考 (Thinking Process)
在 `<think>` `</think>` 标签内进行逐一梳理：
- 梳理各方证据链：每一方的核心支点是什么？
- 评估质询损耗：哪些证据在质询中被削弱了？哪些回应成功化解了危机？
- 对比各方结案：哪一方保留了最高质量的“存活证据”？
- 识别误导性信息：是否存在被投毒、SEO操纵或逻辑存在硬伤的文档？

## Step 2: 最终输出 (JSON Output)
输出一个结构化的 JSON，明确判定胜负。

# JSON Schema
{{
  "summary_of_surviving_evidence": {{
    "观点A": ["经过辩论后依然稳固的证据点1", "证据点2"],
    "观点B": ["...", ...]
  }},
  "winner": "获胜方名称（如“观点A”）",
  "winner_opinion": "获胜方的主张",
  "winning_reasons": [
    "原因1：证据链的完整性和来源权威性...",
    "原因2：在回应环节表现出的逻辑严密性..."
  ],
  "losing_reasons": {{
    "失败方名称": [
      "原因1：核心证据被成功质询且未能给出合理解释...",
      "原因2：文档来源存在明显缺陷或内容逻辑自相矛盾..."
    ]
  }},
  "final_verdict": "一句话总结最终裁决结论。"
}}
"""

    human_template = """
### 争议焦点
[{disputed_point_description}]

### 观点簇与原始文档
{opinion_and_documents_list}

### 第一轮辩论记录
{round_one_record}

### 第二轮辩论记录
{round_two_record}

请开始裁决。
"""

    def __init__(self, model: BaseChatModel):
        """
        初始化 Judge 类。
        
        Args:
            model: langchain 的聊天模型实例（如通过 init_chat_model 或 ChatOpenAI 创建）
        """
        self.model = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("human", self.human_template),
        ])
        self.chain = self.prompt | self.model.bind(max_tokens=8192)

    def _format_documents_for_cluster(self, doc_ids: List[int], documents: List[Dict[str, Any]]) -> str:

        selected_docs = []
        for doc_id in sorted(doc_ids):
            idx = doc_id - 1
            if 0 <= idx < len(documents):
                selected_docs.append({
                    "id": doc_id,
                    "doc": documents[idx]
                })
            else:
                logger.error("doc id out of range. error doc id: %d, len of docs: %d", doc_id, len(documents))
        
        formatted_lines = []
        for item in selected_docs:
            doc_id = item['id']
            doc = item["doc"]
            url = doc["url"]
            content = doc["content"]
            content_escaped = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

            formatted_lines.append(f'  - 文档{doc_id}: {{ "rank": {doc_id}, "url": "{url}", "content": "{content_escaped}" }}')
        
        return "\n".join(formatted_lines)

    def _format_opinion_and_documents(self, disputed_point: Dict[str, Any], documents: List[Dict[str, Any]]) -> str:

        formatted_lines = []
        clusters = disputed_point["clusters"]

        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            opinion = cluster["opinion"]
            doc_ids = cluster["supporting_doc_ids"]
            
            formatted_lines.append(f"- **观点 {cluster_id}: {opinion}** \n支持的文档：")
            docs_str = self._format_documents_for_cluster(doc_ids, documents)
            formatted_lines.append(docs_str)
            formatted_lines.append("")
        
        return "\n".join(formatted_lines)

    def _format_round_one_record(self, debate_record: Dict[str, Any]) -> str:

        formatted_lines = []
        cluster_results = debate_record["cluster_results"]
        
        evidence_dict = {}
        attacks_dict = {}
        
        for cluster_result in cluster_results:
            cluster_id = cluster_result["cluster_id"]
            phase_1_output = cluster_result["phase_1_output"]
            
            evidence_parts = [phase_1_output["opening_statement"], phase_1_output["evidence_advantage"]]
            evidence_dict[cluster_id] = " ".join(evidence_parts)
            
            targeted_attacks = phase_1_output["targeted_attacks"]
            for target_key, attack_content in targeted_attacks.items():
                for other_cluster_result in cluster_results:
                    other_cluster_id = other_cluster_result["cluster_id"]
                    if other_cluster_id != cluster_id:
                        if other_cluster_id in target_key or f"观点{other_cluster_id}" in target_key:
                            attacks_dict[(cluster_id, other_cluster_id)] = attack_content
        
        for cluster_result in cluster_results:
            cluster_id = cluster_result["cluster_id"]
            
            formatted_lines.append(f"- **观点 {cluster_id} 证据**: {evidence_dict[cluster_id]}")
            
            for other_cluster_result in cluster_results:
                other_cluster_id = other_cluster_result["cluster_id"]
                if other_cluster_id != cluster_id:
                    if (cluster_id, other_cluster_id) in attacks_dict:
                        formatted_lines.append(f"- **观点 {cluster_id} 质询 {other_cluster_id}**: \"{attacks_dict[(cluster_id, other_cluster_id)]}\"")
        
        return "\n".join(formatted_lines)

    def _format_round_two_record(self, debate_record: Dict[str, Any]) -> str:

        formatted_lines = []
        cluster_results = debate_record["cluster_results"]
        
        for cluster_result in cluster_results:
            cluster_id = cluster_result["cluster_id"]
            phase_2_output = cluster_result["phase_2_output"]
            json_data = phase_2_output["json_data"]
            
            responses_to_opponents = json_data["responses_to_opponents"].strip()
            concessions = json_data["concessions"].strip()
            response_text = responses_to_opponents + '\n' + concessions
            if response_text:
                formatted_lines.append(f"- **观点 {cluster_id} 回应**: \"{response_text}\"")
            else:
                formatted_lines.append(f"- **观点 {cluster_id} 回应**: (未回应)")
            
            formatted_lines.append(f"- **观点 {cluster_id} 结案**: {json_data['final_summary']}")
        
        return "\n".join(formatted_lines)

    async def judge_async(self, documents: List[Dict[str, Any]], disputed_point: Dict[str, Any], debate_record: Dict[str, Any]) -> Dict[str, Any]:

        topic = disputed_point["topic"]
        
        opinion_and_documents_list = self._format_opinion_and_documents(disputed_point, documents)
        round_one_record = self._format_round_one_record(debate_record)
        round_two_record = self._format_round_two_record(debate_record)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("判决模型提示词：%s", (self.system_template + '\n' + self.human_template).format(
                **{
                    "disputed_point_description": topic,
                    "opinion_and_documents_list": opinion_and_documents_list,
                    "round_one_record": round_one_record,
                    "round_two_record": round_two_record,
                }
            ))

        response = await self.chain.ainvoke({
            "disputed_point_description": topic,
            "opinion_and_documents_list": opinion_and_documents_list,
            "round_one_record": round_one_record,
            "round_two_record": round_two_record,
        })
        
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        try:
            reasoning, json_data = parse_think_and_json(response_text)
        except ValueError as e:
            return ReturnData.error(str(e)[str(e).find("："):], repr(e))
        
        json_data['reasoning'] = reasoning
        
        return ReturnData.success(json_data)
    
    def judge(self, documents: List[Dict[str, Any]], disputed_point: Dict[str, Any], debate_record: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self.judge_async(documents, disputed_point, debate_record))


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    load_dotenv()
    model = ChatOpenAI(
        model=os.getenv('MODEL_ID'),
    )
    
    judge = Judge(model=model)
