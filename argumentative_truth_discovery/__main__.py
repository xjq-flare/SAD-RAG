import os
import sys
import json
import random
import requests
import traceback
import asyncio
import logging
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from core import ArgumentativeTruthDiscovery


async def test_single_dataset(atd, dataset_path, output_file, intermediate_results_dir):
    results = []
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as f_out:
            for line_num, line in tqdm(enumerate(f, 1), desc="testing dataset"):
                logger.info("line num: %s", line_num)

                try:
                    data = json.loads(line.strip())
                    all_documents = []
                    for doc in data['poisoned_docs']:
                        doc['label'] = 'p'
                    all_documents.extend(data['poisoned_docs'])
                    for doc in data['benign_docs']:
                        doc['label'] = 'b'
                    all_documents.extend(data['benign_docs'])
                    for doc in data['noise_docs']:
                        doc['label'] = 'n'
                    all_documents.extend(data['noise_docs'])

                    for idx, doc in enumerate(all_documents):
                        doc["raw_idx"] = idx

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
                            
                    poisoned_docs_ids.sort()
                    benign_docs_ids.sort()
                    noise_doc_ids.sort()
                    logger.info("poisoned_docs_ids: %s", poisoned_docs_ids)
                    logger.info("benign_docs_ids: %s", benign_docs_ids)
                    logger.info("noise_doc_ids: %s", noise_doc_ids)

                    additional_information = {
                        "request_id": data['request_id'],
                        "sample_id": data['sample_id'],
                        "raw_idxs": [doc['raw_idx'] for doc in all_documents],
                        "doc_labels": [doc['label'] for doc in all_documents]
                    }
                    try:
                        result = await atd.process(
                            query=data['query'],
                            documents=all_documents,
                            output_dir=intermediate_results_dir,
                            additional_information=additional_information
                        )
                        
                        result_record = {
                            'query': data['query'],
                            'success': result['has_answer'],
                            'generated_answer': result['final_answer'],
                            'standard_answer': data.get('answer', ''),
                            'attack_goal': data.get('attack_goal', ''),
                            'specific_objective': data.get('specific_objective', ''),
                            'poisoned_docs_ids': poisoned_docs_ids,
                            'benign_docs_ids': benign_docs_ids,
                            'noise_docs_ids': noise_doc_ids,
                            'request_id': data.get('request_id', ''),
                            'sample_id': data['sample_id']
                        }
                        
                    except Exception as e:
                        logger.error(f"处理第 {line_num} 条数据时出错: {repr(e)}")
                        result_record = {
                            'query': data['query'],
                            'success': False,
                            'generated_answer': f"处理错误: {repr(e)}",
                            'standard_answer': data['answer'],
                            'attack_goal': data['attack_goal'],
                            'specific_objective': data['specific_objective'],
                            'poisoned_docs_ids': poisoned_docs_ids,
                            'benign_docs_ids': benign_docs_ids,
                            'noise_docs_ids': noise_doc_ids,
                            'request_id': data['request_id'],
                            'sample_id': data['sample_id']
                        }
                    f_out.write(json.dumps(result_record, ensure_ascii=False) + '\n')
                    f_out.flush()

                except json.JSONDecodeError as e:
                    logger.error(f"解析第 {line_num} 行JSON时出错: {repr(e)}")

        logger.info(f"结果已保存到: {output_file}")

    except FileNotFoundError:
        logger.error(f"数据集文件未找到: {dataset_path}", exc_info=True)
    except Exception as e:
        logger.error(f"读取数据集时出错: {repr(e)}", exc_info=True)


async def test_argumentative_truth_discovery_batch(base_output_dir, dataset_paths=[], tau=3.5):
    
    # 初始化系统
    atd = ArgumentativeTruthDiscovery(model=model, tau=tau)

    intermediate_results_dir = str(base_output_dir / "intermediate_results")
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            logger.info(f"\n开始测试数据集: {dataset_path}")
            output_file = base_output_dir / f"{dataset_path.split('/')[-1][:-6]}_results.jsonl"
            await test_single_dataset(atd, dataset_path, str(output_file), intermediate_results_dir)
        else:
            logger.error(f"找不到数据集：{dataset_path}")
    
    logger.info(f"\n所有测试完成，结果保存在: {base_output_dir}")



if __name__ == "__main__":  
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = Path(f"./record/{timestamp}")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(base_output_dir / "log.log")
        ]
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv("../.env")
    
    model_kwargs = {
        "model": os.getenv('MODEL_ID'),
        "temperature": 0.1,
    }
    
    # Model-specific parameters via environment variables
    extra_body_str = os.getenv('MODEL_EXTRA_BODY', '')
    if extra_body_str:
        import ast
        model_kwargs["extra_body"] = ast.literal_eval(extra_body_str)
    
    reasoning_effort = os.getenv('MODEL_REASONING_EFFORT', '')
    if reasoning_effort:
        model_kwargs["reasoning_effort"] = reasoning_effort
    
    model = ChatOpenAI(**model_kwargs)
    
    tau = 1
    logger.info("tau: %s", tau)
    logger.info("model name: %s", os.getenv('MODEL_ID'))
    

    dataset_paths = [

    ]

    asyncio.run(test_argumentative_truth_discovery_batch(
        base_output_dir, 
        dataset_paths=dataset_paths,
        tau=tau
    ))

