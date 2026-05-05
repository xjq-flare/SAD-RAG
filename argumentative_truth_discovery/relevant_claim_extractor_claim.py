
from typing import Dict, Any
import json
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser


class RelevantClaimExtractor:

    system_template = """# Role
你是一个极度严谨的信息检索与事实分析专家。你的任务是审核文档内容与用户查询（Query）的相关性，并提取出能够回答 Query 的“去语境化原子主张（Atomic Claims）”。

# Task Workflow
1. 相关性评估：
   - 判定准则：只要文档包含能直接回答、间接推导出答案的线索、或相关的背景证据，均视为“相关”。
   - 剔除准则：完全与查询不相关的内容（如纯广告、完全无关的信息、技术报错页面），视为“噪声”。
2. 主张提取（仅限相关文档）：
   - 将相关信息拆解为多个带标签的结构化短句。
   - 相关文档中可能也存在与查询完全不相关的内容（例如广告），与查询完全不相关、对回答查询完全没有帮助的内容不需要提取为主张。但文档中包含任何能间接推导出查询答案的线索，都应保留并提取为主张。
3. 结构化输出：
   - 严格按照指定的 JSON 格式输出。

# Extraction Guidelines
- 去语境化（De-contextualization）：
  - 将所有代词（他、她、它、该片、此事件等）替换为具体实体的全名。
  - 通过添加必要的名词修饰语（如“糖豆的老师”而非“老师”），使每一条主张即使脱离文档也能被完全理解。
- 原子性与简单化：
  - 将复合句拆分成多个简单句。每个主张只承载一个独立的事实点。
  - 尽可能保持输入中的原始措辞，不要进行语义重写。
- 忠实性原则：
  - 仅提取文档明确提到的信息。严禁进行任何主观推论或事实修正。
  - 必须保留文档原始观点，即使该观点明显错误、带有偏见或与常识不符。
- 带标签的语义短句撰写规*：
  - 格式为：`[维度标签] 完整的主张描述`。每个主张必须以“[维度标签]”开头。常见标签包括：
    [基础事实]：涉及时间、地点、人物身份、基本属性。
    [因果动机]：涉及事件起因、心理意图、逻辑关联。
    [过程细节]：涉及具体动作、发展阶段、交互逻辑。
    [结论评价]：涉及结果、影响、外界评价、定性描述。
  - 主张（claims）示例：
    [基础事实] 《欢乐家长群2》中糖豆共情事件发生在第15集。
    [因果动机] 糖豆流泪是因为他看到好朋友被老师批评，感受到了对方的委屈。
    [结论评价] 老师对该行为的解读是糖豆抗压能力不足。

# JSON Output Schema
{{
  "is_relevant": boolean,
  "reason_if_noise": "string", // 仅在 is_relevant 为 false 时填写，否则为空字符串
  "claims": [
    {{
      "tag": "维度标签",
      "content": "去语境化后的简单句主张",
      "evidence": "包含主张及其前后一小段上下文的原文"
    }},
    ...
  ]
}}
*注意：若识别为噪声文档，is_relevant 设为 false，claims 必须输出为空数组 []。并在 reason_if_noise 字段中输出原因*
"""

    human_template = """# User Query
{query}

# Target Document Content
{document_content}

# Instruction
请严格按照系统设定的流程进行分析：
1. 判断该文档是否包含任何能直接或间接回答“{query}”的线索。
2. 若相关，请按“去语境化”和“简单句”原则提取主张（进行结构化拆解）。确保 `evidence` 字段保留了足够的上下文信息（约2-3句），以便核实主张的真实性。
3. 确保 JSON 格式合法，主张内容忠实于原文，不包含你的任何个人分析。

接下来进行判断和提取。
"""

    def __init__(self, model: BaseChatModel):
        self.model = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("human", self.human_template),
        ])
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.model | self.parser

    def extract(self, query: str, document_content: str) -> Dict[str, Any]:
        result = self.chain.invoke({
            "query": query,
            "document_content": document_content
        })
        
        if isinstance(result, dict):
            return result
        else:
            return self._parse_json_fallback(result)
            

    def _parse_json_fallback(self, output: Any) -> Dict[str, Any]:
        
        output_text = str(output)
        
        json_start = output_text.find('{')
        json_end = output_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = output_text[json_start:json_end]
            return json.loads(json_str)
        else:
            raise RuntimeError("RelevantClaimExtractor 调用错误。")