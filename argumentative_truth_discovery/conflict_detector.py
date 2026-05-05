
import json
import logging
import asyncio
from collections import defaultdict
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser

try:
    from .utils import parse_think_and_json, ReturnData
except ImportError:
    from utils import parse_think_and_json, ReturnData


logger = logging.getLogger(__name__)
logger.info("module name: %s", __name__)


class ConflictDetector:

    system_template_detect = """# Role
你是一个高精度的逻辑冲突检测专家。你的任务是分析针对同一用户查询（User Query）给出的答案，识别出存在的冲突。

# Goal
在输出中包含一份结构化的“冲突报告”。
明确列出所有争议点（冲突点）、针对该点的不同观点簇。

# 简单示例：
query: 2023年美国销量最好的智能手机是什么？
回答1为：iPhone 14
回答2为：iPhone 14
回答3为：iPhone
回答4为：fakePhone
回答5为：fakePhone

分析：文档1、2、3 与 文档4、5 存在矛盾。
你在答案中给出的JSON为：
{{
  "has_conflict": True,
  "disputed_points": [
    {{
      "topic": "2023年美国销量最好的智能手机是什么",
      "clusters": [
        {{
          "cluster_id": "A",
          "opinion": "认为2023年美国销量最好的智能手机是 iPhone 或 iPhone 14",
        }},
        {{
          "cluster_id": "B",
          "opinion": "认为2023年美国销量最好的智能手机是fakePhone",
        }}
      ]
    }}
  ]
}}

# Core Definition: Conflict (互斥)
必须严格区分“真正的冲突”与“包含/正交关系”。

## ✅ 判定为冲突 (True Conflict)
只有当两个主张在逻辑上**不可能同时为真**时，才视为冲突。
1. **事实性互斥**：针对同一唯一属性的值不同。
   - 例：A说“数量是10”，B说“数量是50”。（不能既是10又是50）
2. **逻辑性互斥**：因果关系、动机或定性评价相互排斥。
   - 例：A说“他是为了救人”，B说“他是为了谋杀”。
3. **时空排他性**：同一实体在同一时间出现在不同地点。

## ❌ 判定为非冲突 (Non-Conflict/False Positive)
以下情况绝不应被标记为冲突：
1. **包含关系 (Inclusion)**：一个主张是另一个主张的子集或更模糊的表述。
   - 例：A说“王晶执导了《绝命法官》” vs B说“王晶执导过港剧”。（B包含了A，不冲突）
2. **正交/补充关系 (Orthogonal)**：描述同一事物的不同侧面，互不干扰。
   - 例：A说“这辆车很贵” vs B说“这辆车是红色的”。
   - 例：A说“他昨天吃了饭” vs B说“他昨天去了公园”（B没提吃饭不代表否定吃饭）。
3. **冗余/无关关系 (Irrelevant)**: 两个主张没有语义交集。
   - 例：A说“苹果是红色的” vs B说“梨是黄色的”。

# Clustering Rules
1. 在你发现冲突后，冲突点作为争议点，并总结这个冲突下有几个不同的观点，作为观点簇。
2. 可能会存在2个以上的观点簇，比如 iPhone 13、iPhone 14、fakePhone。但是如果是有三个答案分别说是 iPhone 、 iPhone 14 和 fakePhone ，你只需要总结为两个观点簇，既 iPhone 14 和 fakePhone。因此你需要注意相似观点的合并。
3. 可能存在多个冲突点，比如查询为“介绍一下苹果”，答案1说“苹果是红色的水果”、答案2说“苹果是蓝色的蔬菜”，那么一个冲突点是“苹果的颜色”，另一个冲突点是“苹果属于水果还是蔬菜”。

# Thinking Process (Chain of Thought)
1. 提取核心主张：从每个回答中提取独立的主张，如针对“苹果是红色的水果”，提取主张为“苹果是红色的”、“苹果是水果”。
2. 对齐与分组：找出哪些主张是在讨论同一个具体话题。
3. 冲突发现：同一个话题下多个主张是否存在冲突。存在哪些冲突的观点。
4. 聚类：将逻辑上一致的观点合并（例如 "iPhone" 和 "iPhone 14" 归为一类，对抗 "fakePhone"）。
5. 再次检查：检查同一个争议点下的所有观点簇的观点是否真的冲突，如果同一个争议点下观点A和观点B在逻辑上一致，则将观点A与观点B合并，多次合并后如果只剩下一个观点，则删除这个争议点。

# Output Schema (JSON)
<think>（你思考的过程 ... ）</think>
{{
  "has_conflict": boolean,
  "disputed_points": [
    {{
      "topic": "争议点的简短描述 (e.g., 糖豆流泪的真实动机)",
      "clusters": [
        {{
          "cluster_id": "A（单个英文字母表示）",
          "opinion": "该观点的总结 (e.g., 出于对朋友的真诚共情)",
        }},
        {{
          "cluster_id": "B",
          "opinion": "该观点的总结 (e.g., 为了逃避作业的伪装)",
        }}
      ]
    }}
  ]
}}
如果没有发现任何冲突，has_conflict 设为 false，disputed_points 留空（空数组）。
"""

    human_template_detect = """# User Query
{query}

# 针对 User Query 的多个回答
{formatted_claims}

首先进行思考，然后以json的格式给出冲突报告。
请开始分析：
"""

    system_template_classify_docs = """# Role
你是一个高精度的文档归类专家。你的任务是将一个文档归类到最合适的观点簇。

# Rules
1. 提供给你的为大语言模型根据给定文档提取的查询的答案，你需要根据答案来判断给定文档应归类到哪个观点簇或不属于任何一个观点簇。
2. 你需要将给定文档归类到最合适的观点簇，或不归类到任何观点簇。如果一个文档与所有观点簇的观点都不相关或矛盾，则不需要归类到任何观点簇。

# 简单示例
query: 2023年美国销量最好的智能手机是什么？
观点簇A：2023年美国销量最好的智能手机是 iPhone 14 。
观点簇B：2023年美国销量最好的智能手机是 fakePhone 。

如果根据文档的回答为：iPhone 14
则这个文档应归类到观点簇A。

如果根据文档的回答为：iPhone
则这个文档应归类到观点簇A。

如果根据文档的回答为：2023年美国销量最好的手机在全球销量第一。
则这个文档不应该归类到任何观点簇。

# Output Schema (JSON)
{{
  "cluster_id": "B"
}}
在 cluster_id 字段提供单个英文字母的观点簇ID，如果文档不属于任何一个观点簇，则 cluster_id 输出空字符串。"""

    human_template_classify_docs = """# User Query
{query}

# 观点簇
{formatted_clusters}

# 给定文档针对 User Query 的回答
{given_answer}

以JSON的格式输出给定文档应该归类到哪个观点簇或不属于任何观点簇。"""

    def __init__(self, model: BaseChatModel):

        self.model = model
        self.prompt_detect = ChatPromptTemplate.from_messages([
            ("system", self.system_template_detect),
            ("human", self.human_template_detect),
        ])
        self.chain_detect = self.prompt_detect | self.model.bind(max_tokens=8192)

        self.prompt_classify = ChatPromptTemplate.from_messages([
            ("system", self.system_template_classify_docs),
            ("human", self.human_template_classify_docs),
        ])
        self.parser = JsonOutputParser()
        self.chain_classify = self.prompt_classify | self.model.bind(max_tokens=8192) | self.parser

    async def detect_async(self, query: str, claims_list: List[Dict[str, Any]]) -> Dict[str, Any]:

        if not claims_list:
            return ReturnData.success({
                "has_conflict": False,
                "disputed_points": []
            })

        formatted_claims = self._format_answers(claims_list)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("冲突检测模型提示词：%s", (self.system_template_detect + '\n' + self.human_template_detect).format(query=query, formatted_claims=formatted_claims))
        response = await self.chain_detect.ainvoke({
            "query": query,
            "formatted_claims": formatted_claims
        })
        logger.debug("冲突检测模型输出：%s", response.content)
        try:
            reasoning, result = parse_think_and_json(response.content)
        except ValueError as e:
            return ReturnData.error(str(e)[str(e).find("："):], repr(e))
        result['reasoning'] = reasoning
        if not result['has_conflict']:
            return ReturnData.success(result)

        inputs = []
        for dp in result['disputed_points']:
            for cluster in dp['clusters']:
                cluster['cluster_id'] = cluster['cluster_id'].upper()
            formatted_clusters = self._format_clasters(dp['clusters'])
            inputs.append([{
              "query": query, 
              "formatted_clusters": formatted_clusters, 
              "given_answer": doc_item['claims']
              } for doc_item in claims_list])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("文档归类模型提示词：%s", (self.system_template_classify_docs + '\n' + self.human_template_classify_docs).format(
                query=query, 
                formatted_clusters=formatted_clusters,
                given_answer=inputs[0][0]['given_answer']
                ))

        tasks = [
            self.chain_classify.abatch(inp)
            for inp in inputs
        ]
        classify_results = await asyncio.gather(*tasks)
        
        for batch, dp in zip(classify_results, result['disputed_points']):
            dd = defaultdict(list)
            for doc_result, doc_item in zip(batch, claims_list):
                cluster_id = doc_result['cluster_id'].upper()
                dd[cluster_id].append(doc_item['id'])
            for k, v in dd.items():
                if k == "":
                    continue
                for cluster in dp['clusters']:
                    if cluster['cluster_id'] == k:
                        cluster['supporting_doc_ids'] = sorted(v)
                        break
                else:
                    logger.error("模型输出了不存在的观点簇ID：%s: %s", k, v)
            dp['clusters'] = list(filter(lambda item: "supporting_doc_ids" in item, dp['clusters']))
        

        result['disputed_points'] = list(filter(lambda item: len(item['clusters']) >= 2, result['disputed_points']))
        if len(result['disputed_points']) == 0:
            result['has_conflict'] = False

        return ReturnData.success(result)

    def detect(self, query: str, claims_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        return asyncio.run(self.detect_async(query=query, claims_list=claims_list))

    def _format_answers(self, answer_list: List[Dict[str, Any]]) -> str:

        formatted_lines = []
        
        for doc_item in answer_list:
            doc_id = doc_item["id"]
            answer = doc_item["claims"]
            
            formatted_lines.append(f"DoC ID: {doc_id}\nAnswer: {answer}")

        return "\n\n".join(formatted_lines)

    def _format_clasters(self, clusters):
        formatted_lines = []
        for cluster in clusters:
            formatted_lines.append(f"观点簇 {cluster['cluster_id']}: {cluster['opinion']}")
        
        return "\n\n".join(formatted_lines)
        
    

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    load_dotenv()
    model = ChatOpenAI(model=os.getenv('MODEL_ID'))

    cd = ConflictDetector(model=model)

    def fake_format_claims(claim_list):
        return claim_list

    cd._format_answers = fake_format_claims

    claims_str = """"""

    result = cd.detect("《黄土高天》拍摄地点与制作概况", claims_str)
    print(result)

    