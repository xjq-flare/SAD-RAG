import re
import math
import torch
import jieba
import logging
import numpy as np
import networkx as nx
from rouge_score import rouge_scorer
from urllib.parse import urlparse
from pyvis.network import Network
from typing import List, Dict, Tuple, Any, Optional
from FlagEmbedding import BGEM3FlagModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class JiebaTokenizer:
    def tokenize(self, text):
        return list(jieba.cut(text))
    

class Filter:
    _ALLOWED = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")

    def __init__(self, embedding_model_name="BAAI/bge-m3", visualization=False):
        # load embedding model 
        self.embedding_model_name = embedding_model_name
        self.embedding_model = BGEM3FlagModel(
            self.embedding_model_name, 
            use_fp16=True,
            pooling_method='cls',
            devices='cuda:0',
            query_max_length = 4096,
            passage_max_length = 4096)
        
        self.visualization = visualization

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=0,
            separators=['\n\n', '\n', '。', '！', '？', '.', '，', ' ', ''],
            keep_separator=True
        )

        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types=['rougeL'], tokenizer=JiebaTokenizer())

        self.means = [0.617, 0.360]
        self.stds = [0.135, 0.295]

    def safe_tokenize_url(self, url: str) -> List[str]:

        parsed = urlparse(url.lower())
        tokens = [parsed.scheme, parsed.netloc] + parsed.path.strip('/').split('/')
        tokens = [t for t in tokens if t and len(t) > 1]  # 过滤空/短
        return tokens

    def jaccard_sim(self, tokens1: List[str], tokens2: List[str]) -> float:
        set1, set2 = set(tokens1), set(tokens2)
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0
    
    class TF_IDF_sim:
        def __init__(self, contents: List[str], n: int = 4):

            self.K = len(contents)
            vectorizer = TfidfVectorizer(
                analyzer="char",            # 字符级 n-gram
                ngram_range=(n, n),         # 只取 n-gram（这里是 4-gram）[page:1]
                preprocessor=self._clean_zh_alnum,  # 做清洗
                norm="l2"
            )

            self.tfidf_matrix  = vectorizer.fit_transform(contents)

        def char_ngram_tfidf_sim(self, i: int, j: int) -> float:
            if not (0 <= i < self.K and 0 <= j < self.K):
                raise IndexError(f"i/j 超出 contents 下标范围, i: {i}, j: {j}")
            if self.tfidf_matrix[i].nnz == 0 or self.tfidf_matrix[j].nnz == 0:
                return 0.0

            return float(cosine_similarity(self.tfidf_matrix[i], self.tfidf_matrix[j])[0, 0])

        def _clean_zh_alnum(self, text: Optional[str]) -> str:

            if text is None:
                return ""
            s = str(text).lower()
            return Filter._ALLOWED.sub("", s)


    def abnormal_nodes_by_enumeration_mad(
        self,
        G,
        weight_attr: str = "weight",
        tau: float = 3.5,
    ):

        nodes = list(G.nodes())
        n = len(nodes)
        if n < 3:
            logger.warning("文档数量小于3，无法检测异常。")
            return set(), {"reason": "n<3"}

        # Build symmetric weight matrix W
        W = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                u, v = nodes[i], nodes[j]
                w = G[u][v].get(weight_attr, 0.0)
                W[i, j] = W[j, i] = float(w)

        N = 1 << n
        total_w = np.zeros(N, dtype=np.float64)     # sum of internal undirected edge weights
        ksize = np.zeros(N, dtype=np.uint8)         # |S|

        # densities_by_k = [[] for _ in range(n + 1)]
        # masks_by_k = [[] for _ in range(n + 1)]

        edge_densities = []
        masks_edge_den = []

        # DP-style subset enumeration:
        # total_w[mask] = total_w[mask without one bit] + sum(weights from added node to the rest)
        for mask in range(1, N):
            lsb = mask & -mask
            i = (lsb.bit_length() - 1)
            rest = mask ^ lsb

            k = int(ksize[rest]) + 1
            ksize[mask] = k

            s = 0.0
            t = rest
            while t:
                b = t & -t
                j = (b.bit_length() - 1)
                s += W[i, j]
                t ^= b

            total_w[mask] = total_w[rest] + s

            # if 2 <= k <= n - 1:
                # dens = total_w[mask] / k
                # densities_by_k[k].append(dens)
                # masks_by_k[k].append(mask)

            if 2 <= k <= n:
                edge_dens = (2 * total_w[mask] / (k * (k - 1))) * math.sqrt(k)
                edge_densities.append(edge_dens)
                masks_edge_den.append(mask)

        # Compute per-k thresholds and collect abnormal masks
        thresholds = {}
        # abnormal_masks = []
        abnormal_masks_edge_den = []

        # for k in range(2, n):
        #     dens_list = densities_by_k[k]
        #     if not dens_list:
        #         continue

        #     arr = np.asarray(dens_list, dtype=np.float64)
        #     med = float(np.median(arr))
        #     mad = float(np.median(np.abs(arr - med)))  # MAD

        #     # Threshold derived from modified z-score:
        #     # 0.6745 * (x - med) / mad > tau  =>  x > med + (tau/0.6745)*mad
        #     if mad == 0.0:
        #         thr = med  # then any x>med will be flagged
        #     # if mad < 1e-12:
        #     #     logger.warning("MAD 值很小，跳过k=%d", k)
        #     #     thresholds[k] = {"median": med, "mad": mad, "threshold": None, "count": len(arr)}
        #     #     continue
        #     else:
        #         thr = med + (tau / 0.6745) * mad

        #     thresholds[k] = {"median": med, "mad": mad, "threshold": thr, "count": len(arr)}

        #     for mask, dens in zip(masks_by_k[k], dens_list):
        #         if dens > thr:
        #             abnormal_masks.append((mask, float(dens)))

        arr = np.asarray(edge_densities, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        if mad == 0.0:
            thr = med
        else:
            thr = med + (tau / 0.6745) * mad
        thresholds["f_density_edge"] = {"median": med, "mad": mad, "threshold": thr, "count": len(arr)}
        for mask, dens in zip(masks_edge_den, edge_densities):
            if dens > thr:
                abnormal_masks_edge_den.append((mask, float(dens)))
        abnormal_masks = abnormal_masks_edge_den

        # Union all nodes appearing in any abnormal subset
        abnormal_nodes = set()
        for mask, dens in abnormal_masks:
            t = mask
            while t:
                b = t & -t
                idx = b.bit_length() - 1
                abnormal_nodes.add(nodes[idx])
                t ^= b

        debug = {
            "n": n,
            "thresholds_by_k": thresholds,
            "abnormal_subgraph_count": len(abnormal_masks),
            "abnormal_subgraphs": abnormal_masks,  # list of (k, mask, density)
        }
        return abnormal_nodes, debug
    
    def compute_sparse_sim(self, dict1, dict2):

        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
        score = sum(dict1[k] * dict2[k] for k in common_keys)
        return score
    
    def compute_rouge_score(self, sentence1, sentence2):
        score = self.rouge_scorer.score(sentence1, sentence2)
        return score['rougeL'].fmeasure
        
    def refine(self, docs: List[Dict[str, Any]], visualization=False, return_sim_dic=False) -> List[Dict[str, Any]]:

        K = len(docs)
        if K <= 1:
            return docs
        
        contents = [doc['content'] for doc in docs]
        urls = [doc['url'] for doc in docs]

        # contents_chunk_embeddings = []
        # for content in contents:
        #     chunks = self.text_splitter.split_text(content)
        #     chunk_embeddings = self.embedding_model.encode(chunks, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        #     contents_chunk_embeddings.append(chunk_embeddings)

        embeddings = self.embedding_model.encode(contents, return_dense=True, return_sparse=True, return_colbert_vecs=False)

        lexical_weights = embeddings['lexical_weights']

        # sim_content = np.zeros((K, K))
        sim_sparse = np.zeros((K, K))
        # sim_rouge = np.zeros((K, K))
        for i in range(K):
            for j in range(i+1, K):
                # embeddings_A = contents_chunk_embeddings[i]['dense_vecs']
                # embeddings_B = contents_chunk_embeddings[j]['dense_vecs']
                # similarity_matrix = embeddings_A @ embeddings_B.T  # 不是方阵
                # max_A_to_B = similarity_matrix.max(axis=1).mean() # A中每个chunk找B中最大的，平均
                # max_B_to_A = similarity_matrix.max(axis=0).mean() # B中每个chunk找A中最大的，平均
                # symmetric_maxsim = (max_A_to_B + max_B_to_A) / 2
                # sim_content[i, j] = sim_content[j, i] = symmetric_maxsim

                sim_sparse[i, j] = sim_sparse[j, i] = self.compute_sparse_sim(lexical_weights[i], lexical_weights[j])
                # # rouge score
                # sim_rouge[i, j] = sim_rouge[j, i] = self.compute_rouge_score(contents[i], contents[j])

        dense_embeddings = embeddings['dense_vecs']
        sim_content = dense_embeddings @ dense_embeddings.T

        sim_url = np.zeros((K, K))
        # sim_ngram = np.zeros((K, K))
        # tf_idf_sim = self.TF_IDF_sim(contents, n=4)
        for i in range(K):
            url_tokens_i = self.safe_tokenize_url(urls[i])
            for j in range(i+1, K):
                url_tokens_j = self.safe_tokenize_url(urls[j])
                sim_url[i,j] = sim_url[j,i] = self.jaccard_sim(url_tokens_i, url_tokens_j)
                # sim_ngram[i,j] = sim_ngram[j,i] = tf_idf_sim.char_ngram_tfidf_sim(i, j)

        # w_content, w_url, w_ngram = 0.6, 0.1, 0.3
        # edge_weights = w_content * sim_content + w_url * sim_url + w_ngram * sim_ngram
        normalized_sim_content = (sim_content - self.means[0]) / self.stds[0]
        normalized_sim_sparse = (sim_sparse - self.means[1]) / self.stds[1]
        edge_weights = 0.5 * normalized_sim_content + 0.5 * normalized_sim_sparse

        np.fill_diagonal(edge_weights, 1.0)

        G = nx.Graph()
        logger.debug("各边权重：")
        for i in range(K):
            G.add_node(i, doc=docs[i])
        for i in range(K):
            edge_str = []
            for j in range(i+1, K):
                G.add_edge(i, j, weight=edge_weights[i,j])
                edge_str.append(f"{i}({docs[i]['label']})-{j}({docs[j]['label']}) {edge_weights[i, j]:.2f}")
            if edge_str:
                logger.debug(edge_str)
        del edge_str

        # edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0.4]
        # G.remove_edges_from(edges_to_remove)
        
        if self.visualization or visualization:
            for node_id, node_data in G.nodes(data=True):
                node_data['group'] = node_data['doc']['label']
                node_data['title'] = f"Node ID: {node_id}\nTitle: {node_data['doc']['title']}"
                node_data['label'] = node_data['doc']['label']

            # for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
            # edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
            # G_copy = G.copy()
            # G_copy.remove_edges_from(edges_to_remove)
            net.from_nx(G.copy())
            net.save_graph(f"./graph_vis/index.html")
            logger.info("Graph 可视化图生成，保存至./graph_vis/index.html")

        abnormal_nodes, debug = self.abnormal_nodes_by_enumeration_mad(G, weight_attr='weight', tau=3.5)
        # logger.info("thresholds_by_k keys:")
        # for k in range(2, K):
        #     logger.info("k=%s: %s", k, debug["thresholds_by_k"][k])
        logger.info("f_density_edge: %s", debug["thresholds_by_k"]["f_density_edge"])

        logger.info("abnormal_nodes count: %d", len(abnormal_nodes))
        logger.info("删除的文档：%s", [G.nodes[node]['doc']['label'] for node in abnormal_nodes])

        trusted_docs = [G.nodes[node]['doc'] for node in range(K) if node not in abnormal_nodes]
        trusted_docs.sort(key=lambda d: d['rank'], reverse=True)
        logger.info("保留的文档：%s", [doc['label'] for doc in trusted_docs])
        
        if return_sim_dic:
            sim_dic = {
                "sim_content": sim_content,
                "sim_url": sim_url,
                # "sim_ngram": sim_ngram,
                "sim_sparse": sim_sparse,
                # "sim_rouge": sim_rouge,
                # "w_content": w_content,
                # "w_url": w_url,
                # "w_ngram": w_ngram,
                "edge_weights": edge_weights
            }
            return trusted_docs, sim_dic
        else:
            return trusted_docs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    import json

    data_poisoning_dataset_path = ""
    for line_num, line in enumerate(open(data_poisoning_dataset_path, 'r', encoding='utf-8'), 1):
        if line_num == 3:
            data = json.loads(line.strip())
            break

    docs = []
    for doc in data['poisoned_docs']:
        doc['label'] = 'p'
    docs.extend(data['poisoned_docs'])
    for doc in data['benign_docs']:
        doc['label'] = 'b'
    docs.extend(data['benign_docs'])
    for doc in data['noise_docs']:
        doc['label'] = 'n'
    docs.extend(data['noise_docs'])

    for idx, doc in enumerate(docs):
        doc['raw_idx'] = idx

    docs.sort(key=lambda doc: doc['score'], reverse=True)
    for idx, doc in enumerate(docs):
        doc['rank'] = idx + 1
        doc['content'] = doc['title'] + "\n\n" + doc['content']

    filter = Filter(visualization=True)
    refined_docs = filter.refine(docs)
    print(f"Filtered from {len(docs)} to {len(refined_docs)} docs")
    for doc in refined_docs:
        print(f"raw_idx: {doc['raw_idx']}, title: {doc['title']}, label: {doc['label']}")