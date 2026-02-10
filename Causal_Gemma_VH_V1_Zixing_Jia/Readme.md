## 运行流程
1. `Causal_Gemma_VH_V1_Zixing_Jia/clean_duplicate_premise.py`:E-care数据集中存在相同Premise、askfor和Hypothesis对应多个Index的情况，该脚本完成去重的任务，去重逻辑为保留较小的index，是对gemma的100条回答的文件进行操作的（即下一个脚本的生成结果）。
2. `Causal_Gemma_VH_V1_Zixing_Jia/Gemma_Genarate.py`：生成文件为`gemma_causal_answers_100.jsonl`,格式如（每个premise重复100次）：
   ```json
   {"index": "train-10476", "premise": "They saw things shaped like sticks.", "ask-for": "cause", "answer_round": 1, "Gemma_answer": "They were observing pencil-like objects."}
   {"index": "train-10476", "premise": "They saw things shaped like sticks.", "ask-for": "cause", "answer_round": 2, "Gemma_answer": "They were viewing objects from a distance."}
   ```
3. `Causal_Gemma_VH_V1_Zixing_Jia/causal_data_processing.py`:生成文件为：
   ```python
   print(f"   - causal_A_vectors.npy: A向量（因或果）")
   print(f"   - causal_B_vectors.npy: B向量（果或因）")
   print(f"   - causal_metadata.json: 元数据")
   print(f"   - clustering_summary.json: 聚类摘要")
   print(f"   - premise_vectors.json: Premise向量")
   ```
   `causal_metadata.json`中的信息是用来确定索引等等的，不可或缺。
4. `Causal_Gemma_VH_V1_Zixing_Jia/compress_clusters.py`：考虑到存在一个很严重的问题：NvEmbed-V2的Sensory仅有4096d，也就意味着最多支持存储4096个事件，经过统计，发现事件数量大于4096。于是需要`Causal_Gemma_VH_V1_Zixing_Jia/compress_clusters.py`脚本来消除一些多余的Cluster（包含样本数量较少的）。
5. `Causal_Gemma_VH_V1_Zixing_Jia/generate_compressed_data.py`。重新生成训练所需数据：
   ```python
   输入文件（由 causal_data_processing.py 生成）：
     - causal_A_vectors.npy: 原始 A 向量
     - causal_B_vectors.npy: 原始 B 向量
     - causal_metadata.json: 原始元数据
   输入文件（由 compress_clusters.py 生成）：
     - clustering_summary_compressed.json: 压缩后的聚类摘要
   输出文件：
     - causal_A_vectors_compressed.npy: 压缩后的 A 向量
     - causal_B_vectors_compressed.npy: 压缩后的 B 向量
     - causal_metadata_compressed.json: 压缩后的元数据
   ```

6. `Causal_Gemma_VH_V1_Zixing_Jia/causal_learning.py`，进行Vector-HaSH的必背参数学习与训练。需要依赖Vector-HaSH原文件的：
   ```python
   from src.assoc_utils_np_2D import gen_gbook_2d, path_integration_Wgg_2d, module_wise_NN_2d
   from src.seq_utils import nonlin
   ```
   输出的文件有：
   ```python
   pbook.npy, Wgp.npy, Wpg.npy, Wsp.npy,Wps.npy,
   rrm_model.json, positions.json
   ```
7. `Causal_Gemma_VH_V1_Zixing_Jia/Causal_with_VH_AND_LLM.ipynb`。可进行Vector-HaSH完整pipeline的推理过程，并有每一步重建率、最终回答问题的正确率、以及“因为重建失效导致错误选择”等等的分析：

    $$s_i \rightarrow p_i \rightarrow g_i^{noisy} \rightarrow g_i^{clean} \Rightarrow g_{i+1}^{clean} \rightarrow p_{i+1} \rightarrow s_{i+1}$$

8. `Causal_Gemma_VH_V1_Zixing_Jia/Gemma_Answer.question.ipynb`让Gemma直接做选择题的脚本。