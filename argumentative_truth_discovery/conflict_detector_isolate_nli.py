import re
import os
import json
import string
import logging
import asyncio
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel


import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

try:
    from .utils import ReturnData
except ImportError:
    from utils import ReturnData


logger = logging.getLogger(__name__)
logger.info("module name: %s", __name__)


_CHINESE_PUNCT = "，。！？；：、（）【】《》“”‘’"
_ALL_PUNCT = set(string.punctuation + _CHINESE_PUNCT + " \t\r\n")


@dataclass
class AtomicClaim:
    node_id: int
    doc_id: int
    answer: str
    claim: str
    topic_hint: str = ""


@dataclass
class PairRelation:
    contradiction_ab: float
    entailment_ab: float
    neutral_ab: float
    contradiction_ba: float
    entailment_ba: float
    neutral_ba: float

    @property
    def max_contradiction(self) -> float:
        return max(self.contradiction_ab, self.contradiction_ba)

    @property
    def avg_contradiction(self) -> float:
        return (self.contradiction_ab + self.contradiction_ba) / 2.0

    @property
    def max_entailment(self) -> float:
        return max(self.entailment_ab, self.entailment_ba)

    @property
    def min_entailment(self) -> float:
        return min(self.entailment_ab, self.entailment_ba)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class ChineseNLIPredictor:
    DEFAULT_MODEL_CANDIDATES = [
        os.getenv("ATD_NLI_MODEL_DIR", "").strip(),
        str(Path.home() / ".cache/modelscope/hub/models/iic/nlp_structbert_nli_chinese-large"),
        str(Path.home() / ".cache/modelscope/hub/models/damo/nlp_structbert_nli_chinese-large"),
    ]

    def __init__(
        self,
        model_dir: Optional[str] = 'IDEA-CCNL/Erlangshen-Roberta-330M-NLI',
        batch_size: Optional[int] = 64,
        max_length: int = 512,
    ):
        # self.model_dir = self._resolve_model_dir(model_dir)
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self._loaded = False
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None
        # self._id2label = {0: "矛盾", 1: "蕴涵", 2: "中立"}
        self._id2label = {0: "矛盾", 1: "中立", 2: "蕴涵"}

    @classmethod
    def _resolve_model_dir(cls, model_dir: Optional[str]) -> Path:
        candidates = []
        if model_dir:
            candidates.append(model_dir)
        candidates.extend(cls.DEFAULT_MODEL_CANDIDATES)
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if (path / "pytorch_model.bin").exists() and (path / "vocab.txt").exists():
                return path
        raise FileNotFoundError(
            "未找到中文 NLI 模型目录。请将模型放到 ModelScope 默认缓存目录，"
            "或设置环境变量 ATD_NLI_MODEL_DIR 指向包含 pytorch_model.bin 和 vocab.txt 的目录。"
        )

    @staticmethod
    def _import_transformers_without_flash_attn():
        orig_find_spec = importlib.util.find_spec

        def patched_find_spec(name, *args, **kwargs):
            if name == "flash_attn":
                return None
            return orig_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = patched_find_spec
        try:
            import torch
            from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
        finally:
            importlib.util.find_spec = orig_find_spec
        return torch, BertConfig, BertForSequenceClassification, BertTokenizer

    def _load(self) -> None:
        if self._loaded:
            return

        # # 加载 iic/nlp_structbert_nli_chinese-large 模型
        # torch, BertConfig, BertForSequenceClassification, BertTokenizer = (
        #     self._import_transformers_without_flash_attn()
        # )

        # config = BertConfig.from_pretrained(self.model_dir, local_files_only=True)
        # config.num_labels = 3

        # model = BertForSequenceClassification(config)
        # state_dict = torch.load(self.model_dir / "pytorch_model.bin", map_location="cpu")
        # converted_state_dict = {}
        # for key, value in state_dict.items():
        #     if key.startswith("encoder."):
        #         converted_state_dict["bert." + key[len("encoder."):]] = value
        #     else:
        #         converted_state_dict[key] = value

        # missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
        # unexpected_keys = [k for k in unexpected_keys if k != "bert.embeddings.position_ids"]
        # if missing_keys or unexpected_keys:
        #     logger.warning(
        #         "NLI 模型加载存在不完全匹配。missing=%s unexpected=%s",
        #         missing_keys,
        #         unexpected_keys,
        #     )

        # tokenizer = BertTokenizer.from_pretrained(self.model_dir, local_files_only=True)

        # 加载 IDEA-CCNL/Erlangshen-Roberta-330M-NLI 模型
        tokenizer=BertTokenizer.from_pretrained(self.model_dir)
        model=BertForSequenceClassification.from_pretrained(self.model_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._loaded = True

        # if self.batch_size is None:
        #     self.batch_size = 16 if device == "cuda" else 8

    def predict_pairs(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        self._load()
        if not pairs:
            return []

        torch = self._torch
        results: List[Dict[str, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start:start + self.batch_size]
            sentences1 = [item[0] for item in batch_pairs]
            sentences2 = [item[1] for item in batch_pairs]
            encoded = self._tokenizer(
                sentences1,
                sentences2,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits
                probs = logits.softmax(dim=-1).cpu().tolist()
            for row in probs:
                results.append({
                    self._id2label[idx]: float(score)
                    for idx, score in enumerate(row)
                })
        return results


class ConflictDetector:

    system_template_extract_atomic = """# Role
你是一个高精度的信息结构化助手。你需要把“单个文档的隔离回答”拆成后续可用于 NLI 判断的原子主张。

# Goal
输出尽量少但足够完整的原子主张。后续系统会用 NLI 模型判断主张之间是“蕴涵 / 矛盾 / 中立”，所以你必须保证每条主张：
1. 自包含，脱离上下文也能理解。
2. 只表达一个可判定的事实、结论或立场。
3. 尽量保持原回答的语义，不要改写成你自己的解释。
4. `topic_hint` 必须表示“这条主张在回答哪个具体槽位/维度”，不能写成答案值本身，也不能写成模糊主题。

# Extraction Rules
1. 如果一个回答同时涉及多个可独立判定的维度，必须拆开。
   - 例如“苹果是红色的水果”应拆成“苹果是红色的”和“苹果是水果”。
2. 如果回答本身是在给一个列表/集合答案，不要机械地把长列表拆成几十条；应尽量保留 1-3 条能够代表该列表结论的主张。
3. 不要提取“根据资料”“文档提到”“可能”“大概”等元话语；只保留回答本身表达的内容。
4. 如果回答没有形成可判断立场，输出空数组。
5. 对于同一个争议槽位，`topic_hint` 必须尽量写成相同或近乎相同的短语，便于后续把相同观点合并到一起。
6. 如果同一个槽位的答案本身是一个整体集合或多值组合，不要拆成多个互相独立的 claim。
   - 例如“电影A的导演是张三和李四”应优先保留为一个整体 claim，而不是拆成“导演是张三”和“导演是李四”。
7. 必须保留原回答中的排他性信号词，例如“只有”“仅有”“唯一”等，不要弱化成普通陈述。

# topic_hint 示例
- Query: `2023年美国销量最好的智能手机是什么？`
  - claim: `2023年美国销量最好的智能手机是 iPhone 14`
  - 正确的 topic_hint: `2023年美国销量最好的智能手机`
  - 错误的 topic_hint: `销量冠军` / `iPhone 14` / `手机销量排名`
- Query: `介绍一下苹果`
  - claim: `苹果是红色的`
  - topic_hint: `苹果的颜色`
  - claim: `苹果是水果`
  - topic_hint: `苹果的类别`

# Output Schema
{{
  "atomic_claims": [
    {{
      "claim": "完整、自包含、适合做NLI的短句",
      "topic_hint": "该主张对应的争议维度提示，尽量简短稳定，例如“苹果的颜色”“苹果的类别”"
    }}
  ]
}}
"""
# 1. 只保留对回答 Query 真正有帮助的主张。

    human_template_extract_atomic = """# User Query
{query}

# Isolated Answer
{answer}

请输出 JSON。"""

    system_template_summarize = """# Role
你是一个高精度的争议总结助手。给定已经分好的观点簇，请只做描述，不要改动分组。

# Rules
1. 你不能新增、删除、合并、拆分观点簇。
2. 你只能总结：
   - 这个争议点在争论什么（topic）
   - 每个观点簇的观点描述（opinion）
3. topic 要足够具体，能区分不同争议点。
4. opinion 要尽量概括同簇内相近表述，但不能把不同簇说成一样。
5. 如果同簇中存在“更具体”和“更一般”的说法，应优先保留更具体但不失真的表述。

# Output Schema
{{
  "topic": "争议点描述",
  "clusters": [
    {{
      "cluster_id": "A",
      "opinion": "观点总结"
    }}
  ]
}}
"""

    human_template_summarize = """# User Query
{query}

# 当前争议点中的观点簇
{cluster_payload}

请输出 JSON。"""

    def __init__(self, model: BaseChatModel):
        self.model = model
        self.nli = ChineseNLIPredictor()

        self.prompt_extract_atomic = ChatPromptTemplate.from_messages([
            ("system", self.system_template_extract_atomic),
            ("human", self.human_template_extract_atomic),
        ])
        self.chain_extract_atomic = self.prompt_extract_atomic | self.model.bind(max_tokens=4096, temperature=0)

        self.prompt_summarize = ChatPromptTemplate.from_messages([
            ("system", self.system_template_summarize),
            ("human", self.human_template_summarize),
        ])
        self.chain_summarize = self.prompt_summarize | self.model.bind(max_tokens=4096, temperature=0)

        self.same_mutual_entail_threshold = 0.55
        self.same_oneway_entail_threshold = 0.90
        self.same_oneway_soft_entail_threshold = 0.78
        self.same_soft_contradiction_threshold = 0.18
        self.contradiction_avg_threshold = 0.80
        self.contradiction_max_threshold = 0.90
        self.contradiction_entail_ceiling = 0.20

    async def detect_async(self, query: str, claims_list: List[Dict[str, Any]]) -> ReturnData:
        if not claims_list:
            return ReturnData.success({
                "has_conflict": False,
                "disputed_points": [],
            })

        try:
            atomic_claims = await self._extract_atomic_claims(query, claims_list)
            if len(atomic_claims) < 2:
                return ReturnData.success({
                    "has_conflict": False,
                    "disputed_points": [],
                })

            components = await asyncio.to_thread(self._build_disputed_components, atomic_claims)
            if not components:
                return ReturnData.success({
                    "has_conflict": False,
                    "disputed_points": [],
                })

            disputed_points = await self._summarize_components(query, components)
            disputed_points = [item for item in disputed_points if len(item.get("clusters", [])) >= 2]

            return ReturnData.success({
                "has_conflict": bool(disputed_points),
                "disputed_points": disputed_points,
            })
        except Exception as exc:
            logger.error("NLI 冲突检测失败: %r", exc, exc_info=True)
            return ReturnData.error(repr(exc), f"NLI conflict detection failed: {exc}")

    def detect(self, query: str, claims_list: List[Dict[str, Any]]) -> ReturnData:
        return asyncio.run(self.detect_async(query=query, claims_list=claims_list))

    async def _extract_atomic_claims(
        self,
        query: str,
        claims_list: List[Dict[str, Any]],
    ) -> List[AtomicClaim]:
        inputs = [
            {
                "query": query,
                "answer": self._stringify_answer(doc_item.get("claims", "")),
            }
            for doc_item in claims_list
        ]
        tasks = [self.chain_extract_atomic.ainvoke(item) for item in inputs]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        nodes: List[AtomicClaim] = []
        next_node_id = 0
        for doc_item, response in zip(claims_list, responses):
            answer_text = self._stringify_answer(doc_item.get("claims", ""))
            if isinstance(response, Exception):
                logger.warning("原子主张提取失败，使用回退策略。doc_id=%s error=%r", doc_item.get("id"), response)
                parsed = {}
            else:
                parsed = self._parse_json_object(response.content)
            atomic_items = parsed.get("atomic_claims") if isinstance(parsed, dict) else None
            if not isinstance(atomic_items, list):
                atomic_items = self._fallback_atomic_claims(answer_text)

            cleaned_items = []
            for item in atomic_items:
                if not isinstance(item, dict):
                    continue
                claim_text = self._clean_claim_text(item.get("claim", ""))
                topic_hint = self._clean_text(item.get("topic_hint", ""))
                if claim_text:
                    cleaned_items.append({"claim": claim_text, "topic_hint": topic_hint})

            if not cleaned_items:
                cleaned_items = self._fallback_atomic_claims(answer_text)

            cleaned_items = self._preserve_original_answer_when_needed(answer_text, cleaned_items)
            cleaned_items = self._deduplicate_atomic_items(cleaned_items)
            for item in cleaned_items:
                nodes.append(AtomicClaim(
                    node_id=next_node_id,
                    doc_id=int(doc_item["id"]),
                    answer=answer_text,
                    claim=item["claim"],
                    topic_hint=item.get("topic_hint", ""),
                ))
                next_node_id += 1

        return nodes

    def _build_disputed_components(self, nodes: List[AtomicClaim]) -> List[Dict[str, Any]]:
        relations = self._compute_pair_relations(nodes)
        same_uf = _UnionFind(len(nodes))

        for (i, j), relation in relations.items():
            if self._is_same_viewpoint(nodes[i], nodes[j], relation):
                same_uf.union(i, j)

        cluster_to_nodes: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(nodes)):
            cluster_to_nodes[same_uf.find(idx)].append(idx)

        cluster_ids = sorted(cluster_to_nodes.keys())
        contradiction_graph: Dict[int, set] = {cluster_id: set() for cluster_id in cluster_ids}

        for idx_a in range(len(cluster_ids)):
            for idx_b in range(idx_a + 1, len(cluster_ids)):
                cluster_a = cluster_ids[idx_a]
                cluster_b = cluster_ids[idx_b]
                if self._clusters_contradict(cluster_to_nodes[cluster_a], cluster_to_nodes[cluster_b], nodes, relations):
                    contradiction_graph[cluster_a].add(cluster_b)
                    contradiction_graph[cluster_b].add(cluster_a)

        visited = set()
        components: List[Dict[str, Any]] = []
        for cluster_id in cluster_ids:
            if cluster_id in visited or not contradiction_graph[cluster_id]:
                continue

            queue = [cluster_id]
            visited.add(cluster_id)
            component_cluster_ids = []
            while queue:
                current = queue.pop()
                component_cluster_ids.append(current)
                for neighbor in contradiction_graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            component = self._build_component_payload(
                component_cluster_ids=component_cluster_ids,
                cluster_to_nodes=cluster_to_nodes,
                nodes=nodes,
                relations=relations,
            )
            if component is not None:
                components.append(component)

        return components

    def _compute_pair_relations(self, nodes: List[AtomicClaim]) -> Dict[Tuple[int, int], PairRelation]:
        pair_indices: List[Tuple[int, int]] = []
        nli_pairs: List[Tuple[str, str]] = []

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pair_indices.append((i, j))
                nli_pairs.append((nodes[i].claim, nodes[j].claim))
                nli_pairs.append((nodes[j].claim, nodes[i].claim))

        predictions = self.nli.predict_pairs(nli_pairs)
        relations: Dict[Tuple[int, int], PairRelation] = {}
        for offset, (i, j) in enumerate(pair_indices):
            forward = predictions[offset * 2]
            backward = predictions[offset * 2 + 1]
            relations[(i, j)] = PairRelation(
                contradiction_ab=float(forward.get("矛盾", 0.0)),
                entailment_ab=float(forward.get("蕴涵", 0.0)),
                neutral_ab=float(forward.get("中立", 0.0)),
                contradiction_ba=float(backward.get("矛盾", 0.0)),
                entailment_ba=float(backward.get("蕴涵", 0.0)),
                neutral_ba=float(backward.get("中立", 0.0)),
            )
        return relations

    def _is_same_viewpoint(self, left: AtomicClaim, right: AtomicClaim, relation: PairRelation) -> bool:
        if not self._claims_share_slot(left, right):
            return False
        if self._has_exclusive_marker(left.claim) or self._has_exclusive_marker(right.claim):
            if self._normalize_for_containment(left.claim) != self._normalize_for_containment(right.claim):
                return False
        if relation.min_entailment >= self.same_mutual_entail_threshold:
            return True
        if relation.max_entailment >= self.same_oneway_entail_threshold:
            return True
        if (
            relation.max_entailment >= self.same_oneway_soft_entail_threshold
            and relation.max_contradiction <= self.same_soft_contradiction_threshold
            and self._claims_have_surface_containment(left.claim, right.claim)
        ):
            return True
        return False

    def _clusters_contradict(
        self,
        left_nodes: List[int],
        right_nodes: List[int],
        nodes: List[AtomicClaim],
        relations: Dict[Tuple[int, int], PairRelation],
    ) -> bool:
        max_contradiction = 0.0
        contradiction_values = []
        max_entailment = 0.0
        for left in left_nodes:
            for right in right_nodes:
                if not self._claims_share_slot(nodes[left], nodes[right], allow_surface_override=False):
                    continue
                relation = self._get_relation(relations, left, right)
                max_contradiction = max(max_contradiction, relation.max_contradiction)
                max_entailment = max(max_entailment, relation.max_entailment)
                contradiction_values.append(relation.avg_contradiction)

        avg_contradiction = max(contradiction_values) if contradiction_values else 0.0
        return (
            (avg_contradiction >= self.contradiction_avg_threshold or max_contradiction >= self.contradiction_max_threshold)
            and max_entailment <= self.contradiction_entail_ceiling
        )

    def _build_component_payload(
        self,
        component_cluster_ids: List[int],
        cluster_to_nodes: Dict[int, List[int]],
        nodes: List[AtomicClaim],
        relations: Dict[Tuple[int, int], PairRelation],
    ) -> Optional[Dict[str, Any]]:
        cluster_payloads = []
        doc_assignments: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

        for cluster_id in component_cluster_ids:
            member_node_ids = cluster_to_nodes[cluster_id]
            representative_node_id = self._select_representative_claim(member_node_ids, relations)
            representative_claim = nodes[representative_node_id].claim
            topic_counter = Counter(
                self._clean_text(nodes[node_id].topic_hint)
                for node_id in member_node_ids
                if self._clean_text(nodes[node_id].topic_hint)
            )
            cluster_payload = {
                "internal_cluster_id": cluster_id,
                "member_node_ids": member_node_ids,
                "representative_claim": representative_claim,
                "topic_hint": topic_counter.most_common(1)[0][0] if topic_counter else "",
                "example_claims": self._select_example_claims(member_node_ids, nodes, representative_node_id),
            }
            cluster_payloads.append(cluster_payload)

            for node_id in member_node_ids:
                doc_id = nodes[node_id].doc_id
                support_score = self._support_score(node_id, member_node_ids, relations)
                doc_assignments[doc_id].append((cluster_id, support_score))

        cluster_doc_ids: Dict[int, List[int]] = defaultdict(list)
        for doc_id, items in doc_assignments.items():
            if not items:
                continue
            scores_by_cluster = defaultdict(float)
            for cluster_id, score in items:
                scores_by_cluster[cluster_id] += score
            sorted_scores = sorted(scores_by_cluster.items(), key=lambda item: item[1], reverse=True)
            best_cluster_id, best_score = sorted_scores[0]
            if len(sorted_scores) >= 2 and abs(best_score - sorted_scores[1][1]) < 0.05:
                continue
            cluster_doc_ids[best_cluster_id].append(doc_id)

        normalized_clusters = []
        for cluster_payload in cluster_payloads:
            supporting_doc_ids = sorted(set(cluster_doc_ids.get(cluster_payload["internal_cluster_id"], [])))
            if not supporting_doc_ids:
                continue
            normalized_clusters.append({
                "internal_cluster_id": cluster_payload["internal_cluster_id"],
                "supporting_doc_ids": supporting_doc_ids,
                "representative_claim": cluster_payload["representative_claim"],
                "topic_hint": cluster_payload["topic_hint"],
                "example_claims": cluster_payload["example_claims"],
            })

        if len(normalized_clusters) < 2:
            return None

        unique_docs = set()
        for cluster in normalized_clusters:
            unique_docs.update(cluster["supporting_doc_ids"])
        if len(unique_docs) < 2:
            return None

        normalized_clusters.sort(
            key=lambda item: (-len(item["supporting_doc_ids"]), item["representative_claim"])
        )
        return {"clusters": normalized_clusters}

    async def _summarize_components(
        self,
        query: str,
        components: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        inputs = []
        for component in components:
            cluster_lines = []
            for idx, cluster in enumerate(component["clusters"]):
                cluster_id = chr(ord("A") + idx)
                cluster["cluster_id"] = cluster_id
                cluster_lines.append(
                    json.dumps(
                        {
                            "cluster_id": cluster_id,
                            "supporting_doc_ids": cluster["supporting_doc_ids"],
                            "representative_claim": cluster["representative_claim"],
                            "example_claims": cluster["example_claims"],
                            "topic_hint": cluster["topic_hint"],
                        },
                        ensure_ascii=False,
                    )
                )
            inputs.append({
                "query": query,
                "cluster_payload": "\n".join(cluster_lines),
            })

        tasks = [self.chain_summarize.ainvoke(item) for item in inputs]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        disputed_points = []
        for component, response in zip(components, responses):
            if isinstance(response, Exception):
                logger.warning("争议点总结失败，使用回退策略。error=%r", response)
                parsed = {}
            else:
                parsed = self._parse_json_object(response.content)
            topic = self._clean_text(parsed.get("topic", "")) if isinstance(parsed, dict) else ""
            summarized_clusters = parsed.get("clusters") if isinstance(parsed, dict) else None
            summarized_by_id = {}
            if isinstance(summarized_clusters, list):
                for item in summarized_clusters:
                    if not isinstance(item, dict):
                        continue
                    cluster_id = str(item.get("cluster_id", "")).strip().upper()
                    opinion = self._clean_text(item.get("opinion", ""))
                    if cluster_id and opinion:
                        summarized_by_id[cluster_id] = opinion

            if not topic:
                topic = self._fallback_topic(component["clusters"])

            final_clusters = []
            for cluster in component["clusters"]:
                cluster_id = cluster["cluster_id"]
                opinion = summarized_by_id.get(cluster_id) or cluster["representative_claim"]
                final_clusters.append({
                    "cluster_id": cluster_id,
                    "opinion": opinion,
                    "supporting_doc_ids": cluster["supporting_doc_ids"],
                })

            disputed_points.append({
                "topic": topic,
                "clusters": final_clusters,
            })

        return disputed_points

    def _fallback_topic(self, clusters: List[Dict[str, Any]]) -> str:
        topic_counter = Counter()
        for cluster in clusters:
            topic_hint = self._clean_text(cluster.get("topic_hint", ""))
            if topic_hint:
                topic_counter[topic_hint] += 1
        if topic_counter:
            return topic_counter.most_common(1)[0][0]
        return "核心答案存在争议"

    def _select_representative_claim(
        self,
        member_node_ids: List[int],
        relations: Dict[Tuple[int, int], PairRelation],
    ) -> int:
        best_node_id = member_node_ids[0]
        best_score = float("-inf")
        for node_id in member_node_ids:
            score = self._support_score(node_id, member_node_ids, relations)
            if score > best_score:
                best_score = score
                best_node_id = node_id
        return best_node_id

    def _support_score(
        self,
        node_id: int,
        member_node_ids: List[int],
        relations: Dict[Tuple[int, int], PairRelation],
    ) -> float:
        score = 1.0
        for other_node_id in member_node_ids:
            if other_node_id == node_id:
                continue
            relation = self._get_relation(relations, node_id, other_node_id)
            score += relation.max_entailment - relation.max_contradiction
        return score

    def _select_example_claims(
        self,
        member_node_ids: List[int],
        nodes: List[AtomicClaim],
        representative_node_id: int,
        limit: int = 3,
    ) -> List[str]:
        claims = [nodes[representative_node_id].claim]
        for node_id in member_node_ids:
            claim = nodes[node_id].claim
            if claim not in claims:
                claims.append(claim)
            if len(claims) >= limit:
                break
        return claims

    @staticmethod
    def _get_relation(
        relations: Dict[Tuple[int, int], PairRelation],
        left: int,
        right: int,
    ) -> PairRelation:
        if left < right:
            return relations[(left, right)]
        relation = relations[(right, left)]
        return PairRelation(
            contradiction_ab=relation.contradiction_ba,
            entailment_ab=relation.entailment_ba,
            neutral_ab=relation.neutral_ba,
            contradiction_ba=relation.contradiction_ab,
            entailment_ba=relation.entailment_ab,
            neutral_ba=relation.neutral_ab,
        )

    @staticmethod
    def _parse_json_object(text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            return {}

        stripped = text.strip()
        if not stripped:
            return {}

        candidates = []
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            candidates.append(stripped[start:end + 1])

        code_block_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        candidates.extend(code_block_matches)

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return {}

    @staticmethod
    def _fallback_atomic_claims(answer_text: str) -> List[Dict[str, str]]:
        answer_text = answer_text.strip()
        if not answer_text:
            return []

        sentences = [
            segment.strip()
            for segment in re.split(r"[。！？；\n]+", answer_text)
            if segment.strip()
        ]
        if not sentences:
            return []

        if len(sentences) == 1:
            return [{"claim": sentences[0], "topic_hint": ""}]

        return [{"claim": sentence, "topic_hint": ""} for sentence in sentences[:4]]

    @staticmethod
    def _deduplicate_atomic_items(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        deduped = []
        for item in items:
            claim = item["claim"]
            normalized = ConflictDetector._normalize_for_containment(claim)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(item)
        return deduped[:6]

    @classmethod
    def _preserve_original_answer_when_needed(
        cls,
        answer_text: str,
        cleaned_items: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        answer_claim = cls._clean_claim_text(answer_text)
        if not answer_claim or not cleaned_items:
            return cleaned_items

        topic_hint = ""
        topic_counter = Counter(
            cls._clean_text(item.get("topic_hint", ""))
            for item in cleaned_items
            if cls._clean_text(item.get("topic_hint", ""))
        )
        if topic_counter:
            topic_hint = topic_counter.most_common(1)[0][0]

        if cls._has_exclusive_marker(answer_claim) and not any(cls._has_exclusive_marker(item["claim"]) for item in cleaned_items):
            return [{"claim": answer_claim, "topic_hint": topic_hint}]

        if len(cleaned_items) > 1:
            non_empty_topics = [cls._clean_text(item.get("topic_hint", "")) for item in cleaned_items if cls._clean_text(item.get("topic_hint", ""))]
            same_topic = not non_empty_topics or len(set(non_empty_topics)) == 1
            if same_topic and any(marker in answer_claim for marker in ["和", "及", "以及", "、"]):
                return [{"claim": answer_claim, "topic_hint": topic_hint}]

        return cleaned_items

    @staticmethod
    def _clean_claim_text(text: Any) -> str:
        cleaned = ConflictDetector._clean_text(text)
        return cleaned.rstrip("。；;")

    @staticmethod
    def _clean_text(text: Any) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _normalize_for_containment(text: str) -> str:
        text = ConflictDetector._clean_text(text).lower()
        return "".join(ch for ch in text if ch not in _ALL_PUNCT)

    @classmethod
    def _claims_have_surface_containment(cls, left: str, right: str) -> bool:
        left_norm = cls._normalize_for_containment(left)
        right_norm = cls._normalize_for_containment(right)
        if not left_norm or not right_norm:
            return False
        return left_norm in right_norm or right_norm in left_norm

    @classmethod
    def _claims_share_slot(
        cls,
        left: AtomicClaim,
        right: AtomicClaim,
        allow_surface_override: bool = True,
    ) -> bool:
        if cls._topics_are_compatible(left.topic_hint, right.topic_hint):
            return True
        if allow_surface_override and cls._claims_have_surface_containment(left.claim, right.claim):
            return True
        if cls._claim_context_similarity(left.claim, right.claim) >= 0.78:
            return True
        return False

    @classmethod
    def _topics_are_compatible(cls, left_topic: str, right_topic: str) -> bool:
        left_topic = cls._clean_text(left_topic)
        right_topic = cls._clean_text(right_topic)
        if not left_topic or not right_topic:
            return False

        left_norm = cls._normalize_for_containment(left_topic)
        right_norm = cls._normalize_for_containment(right_topic)
        if not left_norm or not right_norm:
            return False
        if left_norm == right_norm or left_norm in right_norm or right_norm in left_norm:
            return True

        left_subject, left_attr = cls._split_topic_hint(left_topic)
        right_subject, right_attr = cls._split_topic_hint(right_topic)
        if left_subject and right_subject and cls._normalize_for_containment(left_subject) == cls._normalize_for_containment(right_subject):
            if left_attr and right_attr:
                left_attr_norm = cls._normalize_for_containment(left_attr)
                right_attr_norm = cls._normalize_for_containment(right_attr)
                if left_attr_norm == right_attr_norm or left_attr_norm in right_attr_norm or right_attr_norm in left_attr_norm:
                    return True
                return False

        return cls._char_ngram_jaccard(left_norm, right_norm, n=2) >= 0.58

    @staticmethod
    def _split_topic_hint(topic_hint: str) -> Tuple[str, str]:
        topic_hint = ConflictDetector._clean_text(topic_hint)
        if "的" in topic_hint:
            subject, attr = topic_hint.rsplit("的", 1)
            return subject.strip(), attr.strip()
        return topic_hint, ""

    @staticmethod
    def _char_ngram_jaccard(left: str, right: str, n: int = 2) -> float:
        if len(left) < n or len(right) < n:
            return 1.0 if left == right and left else 0.0
        left_ngrams = {left[idx:idx + n] for idx in range(len(left) - n + 1)}
        right_ngrams = {right[idx:idx + n] for idx in range(len(right) - n + 1)}
        if not left_ngrams or not right_ngrams:
            return 0.0
        return len(left_ngrams & right_ngrams) / len(left_ngrams | right_ngrams)

    @classmethod
    def _claim_context_similarity(cls, left_claim: str, right_claim: str) -> float:
        left_norm = cls._normalize_for_containment(left_claim)
        right_norm = cls._normalize_for_containment(right_claim)
        if not left_norm or not right_norm:
            return 0.0

        common_prefix = 0
        for left_char, right_char in zip(left_norm, right_norm):
            if left_char != right_char:
                break
            common_prefix += 1
        return common_prefix / max(1, min(len(left_norm), len(right_norm)))

    @staticmethod
    def _has_exclusive_marker(text: str) -> bool:
        text = ConflictDetector._clean_text(text)
        exclusive_markers = ["只有", "仅有", "仅", "唯一", "单独", "独自", "惟一"]
        return any(marker in text for marker in exclusive_markers)

    @staticmethod
    def _stringify_answer(answer: Any) -> str:
        if isinstance(answer, str):
            return answer.strip()
        return json.dumps(answer, ensure_ascii=False)


if __name__ == "__main__":
    import os
    import logging

    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    load_dotenv("../.env_qwen3")
    model = ChatOpenAI(model=os.getenv("MODEL_ID"))
    detector = ConflictDetector(model=model)

    sample_claims = [
        {"id": 1, "claims": "2023年美国销量最好的智能手机是 iPhone 14。"},
        {"id": 2, "claims": "2023年美国销量最好的智能手机是 iPhone。"},
        {"id": 3, "claims": "2023年美国销量最好的智能手机是 fakePhone。"},
    ]
    print(detector.detect("2023年美国销量最好的智能手机是什么？", sample_claims))
