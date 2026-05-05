import json
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np 

from graph_mining_filter import Filter


def test_single_dataset(filter, dataset_path, output_file, output_similarity):

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, 'r', encoding='utf-8') as f, \
        open(output_file, 'w', encoding='utf-8') as f_out, \
        open(output_similarity, 'w', encoding='utf-8') as f_out_sim:
        for line_num, line in tqdm(enumerate(f, 1), desc="testing dataset"):
            logger.info("line num: %s", line_num)

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
                doc['raw_idx'] = idx

            all_documents.sort(key=lambda item: item['score'], reverse=True)

            for doc_id, doc in enumerate(all_documents, 1):
                doc['rank'] = doc_id
                doc['content'] = doc['title'] + "\n\n" + doc['content']

            try:
                refined_documents, sim_dic = filter.refine(all_documents, return_sim_dic=True)
                logger.info(f"从 {len(all_documents)} 个文档中过滤得到 {len(refined_documents)} 个文档")
                logger.info("保留的文档：%s", [doc['label'] for doc in refined_documents])
                refined_raw_idxs = [doc['raw_idx'] for doc in refined_documents]
                all_raw_idxs = [doc['raw_idx'] for doc in all_documents]
                filtered_raw_idxs = [idx for idx in all_raw_idxs if idx not in refined_raw_idxs]
                filtered_docs = []
                for doc in all_documents:
                    if doc['raw_idx'] in filtered_raw_idxs:
                        filtered_docs.append([doc['raw_idx'], doc['label']])

                result_record = {
                    'query': data['query'],
                    'refined_docs': [[doc['raw_idx'], doc['label']] for doc in refined_documents],
                    'filtered_docs': filtered_docs,
                    'all_docs': [[doc['raw_idx'], doc['label']] for doc in all_documents],
                    'request_id': data.get('request_id', '')
                }
                f_out.write(json.dumps(result_record, ensure_ascii=False) + '\n')
                f_out.flush()

                sim_data = {}
                for k, v in sim_dic.items():
                    if isinstance(v, float):
                        sim_data[k] = v
                    if isinstance(v, np.ndarray):
                        sim_data[k] = {
                            "data": v.tolist(),
                            "dtype": str(v.dtype),
                            "shape": list(v.shape)
                        }
                f_out_sim.write(json.dumps(sim_data, ensure_ascii=False) + '\n')
                f_out_sim.flush()
                        
            except Exception as e:
                logger.error(f"处理第 {line_num} 条数据时出错: {repr(e)}；requert_id: {data['request_id']}")

        logger.info(f"结果已保存到: {output_file}")

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

    data_poisoning_dataset_path = ""
    prompt_injection_dataset_path = ""

    filter = Filter()

    if data_poisoning_dataset_path:
        logger.info(f"\n开始测试数据投毒数据集: {data_poisoning_dataset_path}")
        poisoning_output_file = base_output_dir / "data_poisoning_results.jsonl"
        poisoning_output_jsonl_file = base_output_dir / "data_poisoning_sim_matrix.jsonl"
        test_single_dataset(filter, data_poisoning_dataset_path, str(poisoning_output_file), str(poisoning_output_jsonl_file))
    
    if prompt_injection_dataset_path:
        logger.info(f"\n开始测试指令注入数据集: {prompt_injection_dataset_path}")
        injection_output_file = base_output_dir / "prompt_injection_results.jsonl"
        injection_output_jsonl_file = base_output_dir / "prompt_injection_sim_matrix.jsonl"
        test_single_dataset(filter, prompt_injection_dataset_path, str(injection_output_file), str(injection_output_jsonl_file))
    
    logger.info(f"\n所有测试完成，结果保存在: {base_output_dir}")