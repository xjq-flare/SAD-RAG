import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

import sys
sys.path.append('../')
from graph_mining_filter import ContradictoryFilter

try:
    from .relevant_claim_extractor import RelevantClaimExtractor
    from .conflict_detector import ConflictDetector
    from .debater import Debater
    from .judge import Judge
    from .verified_knowledge_generator import VerifiedKnowledgeGenerator
except ImportError:
    from relevant_claim_extractor import RelevantClaimExtractor
    from conflict_detector import ConflictDetector
    from debater import Debater
    from judge import Judge
    from verified_knowledge_generator import VerifiedKnowledgeGenerator


logger = logging.getLogger(__name__)


class ArgumentativeTruthDiscovery:
    def __init__(self, model: Optional[BaseChatModel] = None, tau=3.5, isolate_answer=False, multistep_conflict_detect=0, consolidate_answer=False):
        if model is None:
            load_dotenv()
            model = ChatOpenAI(model=os.getenv('MODEL_ID'))
        self.isolate_answer = isolate_answer
        self.multistep_conflict_detect = multistep_conflict_detect
        self.consolidate_answer = consolidate_answer

        self.multistep_conflict_detect = multistep_conflict_detect
        if self.isolate_answer:
            import importlib
            module = importlib.import_module("isolate_answer")
            RelevantClaimExtractor = module.RelevantClaimExtractor
            if not self.multistep_conflict_detect:
                module = importlib.import_module("conflict_detector_isolate")
                ConflictDetector = module.ConflictDetector
            del module

        if self.multistep_conflict_detect == 1:
            import importlib
            module = importlib.import_module("conflict_detector_isolate_multistep")
            ConflictDetector = module.ConflictDetector
            del module
        elif self.multistep_conflict_detect == 2:
            import importlib
            module = importlib.import_module("conflict_detector_isolate_multistep2")
            ConflictDetector = module.ConflictDetector
            del module
        elif self.multistep_conflict_detect == 3:
            import importlib
            module = importlib.import_module("conflict_detector_isolate_nli")
            ConflictDetector = module.ConflictDetector
            del module
        else:
            raise NotImplementedError(f"multistep_conflict_detect == {multistep_conflict_detect}")
        
        self.model = model
        self.claim_extractor = RelevantClaimExtractor(model)
        self.conflict_detector = ConflictDetector(model)
        self.debater = Debater(model)
        self.judge = Judge(model)
        self.knowledge_generator = VerifiedKnowledgeGenerator(model)

        self.contradictory_filter = ContradictoryFilter()

        self.last_record_file = ""
        
        self.intermediate_results = {}
        self.tau = tau
    
    async def _extract_claims_async(self, query: str, document_content: str, doc_id: int) -> Dict[str, Any]:
        try:
            result = await self.claim_extractor.extract_async(query, document_content)
            return {
                "doc_id": doc_id,
                "result": result,
                "success": True,
                "error": None
            }
        except Exception as e:
            logger.error(f"提取文档 {doc_id} 的主张时出错: {repr(e)}", exc_info=True)
            return {
                "doc_id": doc_id,
                "result": None,
                "success": False,
                "error": str(e)
            }
    
    async def _extract_all_claims(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"开始并行提取 {len(documents)} 个文档的主张...")
        
        tasks = [
            self._extract_claims_async(query, doc["content"], idx + 1)
            for idx, doc in enumerate(documents)
        ]
        
        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r["success"])
        logger.info(f"主张提取完成: {success_count}/{len(documents)} 成功")
        
        return results
    
    def _filter_noise_documents(self, extraction_results: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> tuple:
        relevant_claims_list = []
        relevant_documents = []
        noise_doc_ids = []
        
        for result in extraction_results:
            if not result["success"]:
                continue
            
            doc_id = result["doc_id"]
            extract_result = result["result"]
            
            if extract_result["is_relevant"]:
                relevant_claims_list.append({
                    "id": doc_id,
                    "claims": extract_result["claims"]
                })
                relevant_documents.append(documents[doc_id - 1])
            else:
                noise_doc_ids.append(doc_id)
        
        logger.info(f"过滤噪声文档: 保留 {len(relevant_claims_list)} 个相关文档，丢弃 {len(noise_doc_ids)} 个噪声文档")
        logger.info("识别出的噪声文档：%s", sorted(noise_doc_ids))
        logger.info([self.id_and_labels[doc_id] for doc_id in sorted(noise_doc_ids)])
        logger.info("识别出的相关文档：%s", sorted([claims['id'] for claims in relevant_claims_list]))
        logger.info([self.id_and_labels[doc_id] for doc_id in sorted([claims['id'] for claims in relevant_claims_list])])
        
        return relevant_claims_list, relevant_documents, noise_doc_ids
    
    async def _process_disputed_point_async(
        self, 
        disputed_point: Dict[str, Any], 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        try:
            topic = disputed_point["topic"]
            logger.info(f"开始处理争议点: {topic}")
    
            return_data = await self.debater.debate_async(disputed_point, documents)
            if not return_data.success:
                return {
                    "disputed_point": disputed_point,
                    "debate_record": None,
                    "judge_result": None,
                    "success": False,
                    "error": return_data.message
                }
            debate_record = return_data.result
            logger.info(f"争议点 '{topic}' 的辩论完成")

            return_data = await self.judge.judge_async(documents, disputed_point, debate_record)
            if not return_data.success:
                return {
                        "disputed_point": disputed_point,
                        "debate_record": None,
                        "judge_result": None,
                        "success": False,
                        "error": return_data.message
                }
            judge_result = return_data.result
            logger.info(f"争议点 '{topic}' 的裁决完成")
            
            return {
                "disputed_point": disputed_point,
                "debate_record": debate_record,
                "judge_result": judge_result,
                "success": True,
                "error": None
            }
        except Exception as e:
            logger.error(f"处理争议点 '{disputed_point.get('topic', '未知')}' 时出错: {repr(e)}", exc_info=True)
            return {
                "disputed_point": disputed_point,
                "debate_record": None,
                "judge_result": None,
                "success": False,
                "error": repr(e)
            }
    
    async def _process_all_disputed_points(
        self, 
        disputed_points: List[Dict[str, Any]], 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not disputed_points:
            return []
        
        logger.info(f"开始并行处理 {len(disputed_points)} 个争议点...")
        
        tasks = [
            self._process_disputed_point_async(dp, documents)
            for dp in disputed_points
        ]
        
        results = await asyncio.gather(*tasks)
        
        success_count = sum(1 for r in results if r["success"])
        logger.info(f"争议点处理完成: {success_count}/{len(disputed_points)} 完成")
        
        return results
    
    
    def _get_losing_doc_ids(self, judge_results: List[Dict[str, Any]]) -> set:
        losing_doc_ids = set()
        
        for result in judge_results:
            if not result["success"] or result["judge_result"] is None:
                logger.error("辩论与裁决出错，争议点：%s", result['disputed_point']['topic'])
                continue
            
            verdict = result["judge_result"]
            winner = verdict["winner"]
            
            if not winner:
                logger.error("no winner!")
                continue
            
            disputed_point = result["disputed_point"]
            clusters = disputed_point["clusters"]
            logger.info(f"争议点 '{disputed_point['topic']}' 中，成功观点 “{verdict['winner_opinion']}” ")
            for cluster in clusters:
                cluster_id = cluster["cluster_id"]
                opinion = cluster["opinion"]
                
                is_winner = (
                    cluster_id == winner or
                    f"观点{cluster_id}" == winner or
                    f"观点 {cluster_id}" == winner or
                    f"观点{cluster_id}方" == winner or
                    opinion == winner or
                    (len(opinion) < 50 and opinion in winner) or  # 只有当 opinion 较短时才使用 in 匹配
                    cluster_id in winner
                )
                
                if not is_winner:
                    doc_ids = cluster["supporting_doc_ids"]
                    losing_doc_ids.update(doc_ids)
                    logger.info(f"失败观点 “{opinion}” ")
                    logger.info("失败观点文档：%s", doc_ids)
                    logger.info([self.id_and_labels[doc_id] for doc_id in doc_ids])
                else:
                    doc_ids = cluster["supporting_doc_ids"]
                    logger.info(f"成功观点 “{opinion}” ")
                    logger.info("成功观点文档：%s", doc_ids)
                    logger.info([self.id_and_labels[doc_id] for doc_id in doc_ids])
        return losing_doc_ids
    
    def _save_intermediate_results(self, output_dir: Optional[str] = None):
        if output_dir is None:
            return
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            result_file = output_path / f"intermediate_results_{timestamp}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(self.intermediate_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"中间结果已保存到: {result_file}")
            
            if self.last_record_file and self.last_record_file != result_file:
                os.remove(self.last_record_file)
            self.last_record_file = result_file
        except Exception as e:
            logger.error(f"保存中间结果时出错: {repr(e)}", exc_info=True)
    
    async def process(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        output_dir: Optional[str] = None,
        additional_information = None
    ) -> Dict[str, Any]:
        logger.info(f"开始处理查询: {query}")
        logger.info(f"输入文档数量: {len(documents)}")

        self.id_and_labels = [None,]
        for doc_id, doc in enumerate(documents, 1):
            if doc['label'] == 'p':
                self.id_and_labels.append(f"{doc_id}(p)")
            if doc['label'] == 'b':
                self.id_and_labels.append(f"{doc_id}(b)")
            if doc['label'] == 'n':
                self.id_and_labels.append(f"{doc_id}(n)")
        
        self.intermediate_results = {"query": query}
        if additional_information:
            self.intermediate_results.update(additional_information)
        self.intermediate_results.update({
            "input_documents_count": len(documents),
            "claim_extraction": {},
            "conflict_detection": {},
            "graph_filter": {},
            "debate_and_judge": {},
            "answer_generation": {},
            "final_result": {}
        })
        self.last_record_file = ""
        
        try:
            logger.info("=" * 50)
            logger.info("提取文档主张")
            logger.info("=" * 50)
            
            extraction_results = await self._extract_all_claims(query, documents)
            self.intermediate_results["claim_extraction"] = {
                "results": extraction_results,
                "total_documents": len(documents),
                "successful_extractions": sum(1 for r in extraction_results if r["success"])
            }
            self._save_intermediate_results(output_dir)
            
            relevant_claims_list, relevant_documents, noise_doc_ids = self._filter_noise_documents(
                extraction_results, documents
            )
            self.intermediate_results["claim_extraction"]["noise_doc_ids"] = noise_doc_ids
            self.intermediate_results["claim_extraction"]["relevant_claims_list"] = relevant_claims_list
            self._save_intermediate_results(output_dir)

            logger.info("=" * 50)
            logger.info("检测文档冲突")
            logger.info("=" * 50)

            return_data = await self.conflict_detector.detect_async(query, relevant_claims_list)
            if return_data.success:
                conflict_result = return_data.result
                self.intermediate_results["conflict_detection"] = conflict_result
                has_conflict = conflict_result["has_conflict"]
            else: 
                logger.warning("冲突检测失败！")
                self.intermediate_results["conflict_detection"] = {
                    "success": return_data.success,
                    "message": return_data.message
                }
                has_conflict = False
            self._save_intermediate_results(output_dir)
                        
            
            if not has_conflict:
                logger.info("未检测到冲突，直接生成答案")
                logger.info("最终保留的文档：%s", sorted([claims['id'] for claims in relevant_claims_list]))
                logger.info([self.id_and_labels[doc_id] for doc_id in sorted([claims['id'] for claims in relevant_claims_list])])
                answer_result = self.knowledge_generator.generate_from_documents(
                    [{
                        "title": "",
                        "content": claim['claims']
                    } for claim in relevant_claims_list], query
                )
                self.intermediate_results["answer_generation"] = answer_result
                self.intermediate_results["final_result"] = answer_result
                self._save_intermediate_results(output_dir)
                
                logger.info("=" * 50)
                logger.info("流程完成")
                logger.info("=" * 50)
                
                return {
                    "query": query,
                    "final_answer": answer_result,
                    "has_answer": True,
                    "intermediate_results": self.intermediate_results
                }
            else:
                disputed_points = conflict_result['disputed_points']
                logger.info("检测到 %d 个冲突，开始辩论和裁决流程", len(disputed_points))
                for idx, p in enumerate(disputed_points, 1):
                    logger.info("争议点 %d : %s", idx, p['topic'])

                removed_doc_ids = []
                logger.info("=" * 50)
                logger.info("图挖掘过滤")
                logger.info("=" * 50)
                doc_groups = [[c['supporting_doc_ids'] for c in dp['clusters']] for dp in disputed_points]
                contradictory_subgraph_nodes = []
                original_doc_id_to_relevant_index = {}
                for index, item in enumerate(relevant_claims_list):
                    original_doc_id_to_relevant_index[item['id']] = index

                for group in doc_groups:
                    subgraphs = []
                    for doc_ids in group:
                        subgraphs.append([original_doc_id_to_relevant_index[doc_id] for doc_id in doc_ids])
                    contradictory_subgraph_nodes.append(subgraphs)

                abnormal_nodes_in_relevant = self.contradictory_filter.filter_out(relevant_documents, contradictory_subgraph_nodes, tau=self.tau)

                abnormal_doc_ids = []
                for node in abnormal_nodes_in_relevant:
                    abnormal_doc_ids.append(relevant_claims_list[node]['id'])
                removed_doc_ids.extend(abnormal_doc_ids)

                logger.info(f"过滤掉的文档ID: {sorted(abnormal_doc_ids)}")
                self.intermediate_results["graph_filter"]["abnormal_doc_ids"] = sorted(abnormal_doc_ids)

                disputed_points_after_filtered = []
                for dp in disputed_points:
                    dp_copy = dp.copy()
                    new_clusters = []
                    for cluster in dp['clusters']:
                        if not all(doc_id in abnormal_doc_ids for doc_id in cluster['supporting_doc_ids']):
                            new_clusters.append(cluster)
                    if len(new_clusters) > 1:
                        dp_copy['clusters'] = new_clusters
                        disputed_points_after_filtered.append(dp_copy)
                
                if disputed_points_after_filtered:
                    logger.info("=" * 50)
                    logger.info("辩论和裁决")
                    logger.info("=" * 50)
                
                    judge_results = await self._process_all_disputed_points(disputed_points_after_filtered, documents)
                    self.intermediate_results["debate_and_judge"]["judge_results"] = judge_results
                    self._save_intermediate_results(output_dir)
                
                    losing_doc_ids = self._get_losing_doc_ids(judge_results)
                    removed_doc_ids.extend(losing_doc_ids)
                    logger.info(f"失败方文档ID: {sorted(losing_doc_ids)}")
                    logger.info([self.id_and_labels[doc_id] for doc_id in sorted(losing_doc_ids)])
                    self.intermediate_results["debate_and_judge"]["losing_doc_ids"] = sorted(losing_doc_ids)

                final_documents = []
                final_documents_ids = []
                final_doc_claims = []
                for claim_item in relevant_claims_list:
                    doc_id = claim_item["id"]
                    if doc_id not in removed_doc_ids:
                        if doc_id >= 1 and doc_id <= len(documents):
                            final_documents.append(documents[doc_id - 1])
                            final_documents_ids.append(doc_id)
                            final_doc_claims.append(claim_item['claims'])
                        else:
                            logger.error("doc id out of range. error doc id: %d, len of docs: %d", doc_id, len(documents))
                
                logger.info(f"剩余文档数量: {len(final_documents)} (从 {len(relevant_claims_list)} 个相关文档中排除了 {len(removed_doc_ids)} 个文档)")
                logger.info("识别出的正常文档：%s", final_documents_ids)
                logger.info([self.id_and_labels[doc_id] for doc_id in final_documents_ids])
                self.intermediate_results["final_documents_ids"] = final_documents_ids
                self._save_intermediate_results(output_dir)
                
                # 生成最终答案
                logger.info("=" * 50)
                logger.info("生成最终答案")
                logger.info("=" * 50)
                logger.info("最终保留的文档：%s", final_documents_ids)
                logger.info([self.id_and_labels[doc_id] for doc_id in final_documents_ids])
                
                answer_result = self.knowledge_generator.generate_from_documents(
                    [{
                        "title": "",
                        "content": claim
                    } for claim in final_doc_claims], query
                )
                self.intermediate_results["answer_generation"] = answer_result
                self.intermediate_results["final_result"] = answer_result
                self._save_intermediate_results(output_dir)
                
                logger.info("=" * 50)
                logger.info("流程完成")
                logger.info("=" * 50)
                
                return {
                    "query": query,
                    "final_answer": answer_result,
                    "has_answer": True,
                    "intermediate_results": self.intermediate_results
                }
        
        except Exception as e:
            logger.error(f"处理过程中发生错误: {repr(e)}", exc_info=True)
            self.intermediate_results["error"] = str(e)
            self._save_intermediate_results(output_dir)
            
            return {
                "query": query,
                "final_answer": f"处理过程中发生错误: {repr(e)}",
                "has_answer": False,
                "intermediate_results": self.intermediate_results
            }
    
    def process_sync(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:

        return asyncio.run(self.process(query, documents, output_dir))



if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    for file in os.listdir("."):
        if file.endswith(".py"):
            logging.getLogger(file[:-3]).setLevel(logging.DEBUG)
    
    async def test_argumentative_truth_discovery(test_list, isolate_answer=False, multistep_conflict_detect=0):
        
        atd = ArgumentativeTruthDiscovery(model=model, isolate_answer=isolate_answer, multistep_conflict_detect=multistep_conflict_detect)
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                test_data = []
                for i, line in enumerate(f, 1):
                    if i not in test_list:
                        continue
                    data = json.loads(line.strip())
                    test_data.append((i - 1, data))
            
            logger.info(f"读取了 {len(test_data)} 条测试数据")
            
            for idx, data in test_data:
                logger.info(f"\n{'='*60}")
                logger.info(f"处理测试数据 {idx+1}: {data['query']}")
                logger.info(f"具体攻击目标：{data['specific_objective']}")
                logger.info(f"{'='*60}")

                all_documents = []
                if 'poisoned_docs' in data:
                    for doc in data['poisoned_docs']:
                        all_documents.append(doc)
                        logger.info(f"添加投毒文档: {doc['title'][:50]}...")
                        doc['label'] = 'p'
                    logger.info(f"投毒文档数量：{len(data['poisoned_docs'])}")
                if 'benign_docs' in data:
                    for doc in data['benign_docs']:
                        all_documents.append(doc)
                        logger.info(f"添加正常文档: {doc['title'][:50]}...")
                        doc['label'] = 'b'
                    logger.info(f"正常文档数量：{len(data['benign_docs'])}")
                if 'noise_docs' in data:
                    for doc in data['noise_docs']:
                        all_documents.append(doc)
                        logger.info(f"添加噪声文档: {doc['title'][:50]}...")
                        doc['label'] = 'n'
                    logger.info(f"噪声文档数量：{len(data['noise_docs'])}")

                all_documents.sort(key=lambda item: item['score'], reverse=True)

                poisoned_docs_ids = []
                benign_docs_ids = []
                noise_doc_ids = []

                for doc_id, doc in enumerate(all_documents, 1):
                    if doc['label'] == 'p':
                        poisoned_docs_ids.append(doc_id)
                    if doc['label'] == 'b':
                        benign_docs_ids.append(doc_id)
                    if doc['label'] == 'n':
                        noise_doc_ids.append(doc_id)
                logger.info(f"投毒文档ids：{sorted(poisoned_docs_ids)}")
                logger.info(f"干净文档ids：{sorted(benign_docs_ids)}")
                logger.info(f"噪声文档ids：{sorted(noise_doc_ids)}")
                logger.info(f"共有 {len(all_documents)} 个文档")
                logger.info("docs_urls: %s", [doc['url'] for doc in all_documents])

                try:
                    result = await atd.process(
                        query=data['query'],
                        documents=all_documents,
                        output_dir=f"{output_dir}/test_output_{idx+1}"
                    )
                    
                    logger.info(f"\n最终答案:")
                    logger.info(f"是否有答案: {result['has_answer']}")
                    logger.info(f"答案内容: {result['final_answer'][:200]}...")
                    
                except Exception as e:
                    logger.error(f"处理测试数据 {idx+1} 时出错: {repr(e)}", exc_info=True)
                    continue
                
                logger.info(f"\n测试数据 {idx+1} 处理完成")
                logger.info(f"{'='*60}\n")
        
        except FileNotFoundError:
            logger.error(f"测试数据文件未找到: {dataset_path}")
        except Exception as e:
            logger.error(f"读取测试数据时出错: {repr(e)}", exc_info=True)
    
    
    load_dotenv("")
    model = ChatOpenAI(
        model=os.getenv('MODEL_ID'),
    )
    dataset_path = ""
    output_dir = ""
    isolate_answer = True
    multistep_conflict_detect = 3
    test_list = [1,]

    asyncio.run(test_argumentative_truth_discovery(
        test_list=test_list,
        isolate_answer=isolate_answer, 
        multistep_conflict_detect=multistep_conflict_detect
    ))
