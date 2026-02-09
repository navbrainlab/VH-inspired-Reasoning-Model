## 运行流程
0. `Causal_with_LLM_V2/clean_duplicate_premise.py`:E-care数据集中存在相同Premise、askfor和Hypothesis对应多个Index的情况，该脚本完成去重的任务，去重逻辑为保留较小的index，是对gemma的100条回答的文件进行操作的（即下一个脚本的生成结果）。
1. `Causal_with_LLM_V1_Zixing_Jia/Gemma_Genarate.py`
2. `Causal_with_LLM_V1_Zixing_Jia/causal_data_processing.py`
3. `Causal_with_LLM_V1_Zixing_Jia/compress_clusters.py`：考虑到存在一个很严重的问题：NvEmbed-V2的Sensory仅有4096d，也就意味着最多支持存储4096个事件，经过统计，发现事件数量大于4096。于是需要`Causal_with_LLM_V1_Zixing_Jia/compress_clusters.py`脚本来消除一些多余的Cluster（包含样本数量较少的）。
4. `Causal_with_LLM_V1_Zixing_Jia/generate_compressed_data.py`。重新生成训练所需数据
5. `Causal_with_LLM_V1_Zixing_Jia/causal_learning.py`，进行Vector-HaSH的必背参数学习与训练。
6. `Causal_with_LLM_V1_Zixing_Jia/Causal_with_VH_AND_LLM.ipynb`。可进行Vector-HaSH完整pipeline的推理过程，并有每一步重建率、最终回答问题的正确率、以及“因为重建失效导致错误选择”等等的分析：

    $$s_i \rightarrow p_i \rightarrow g_i^{noisy} \rightarrow g_i^{clean} \Rightarrow g_{i+1}^{clean} \rightarrow p_{i+1} \rightarrow s_{i+1}$$

7. `Causal_with_LLM_V1_Zixing_Jia/Gemma_Answer.question.ipynb`让Gemma直接做选择题的脚本。