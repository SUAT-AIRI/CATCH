from langchain_core.runnables import RunnableParallel
from tqdm import tqdm
import collections
from dstc12.prompts import LABEL_CLUSTERS_PROMPT, LABEL_CLUSTERS_MULTI_PROMPT, LABEL_CONCLUDE_PROMPT, LABEL_FILTER_PROMPT, STYLEGUIDE_SECTION_1_PROMPT, STYLEGUIDE_SECTION_2_PROMPT, STYLEGUIDE_SECTION_3_PROMPT
from dstc12.utils import get_llm, DotAllRegexParser
import logging
from langchain_openai import ChatOpenAI
import os
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
level=logging.DEBUG,
filename='label_30.log',
filemode='a')



def label2_multi_generate(llm_name, n_clusters, clusters_with_preferences, topics, utterances, top_ids):
    llm = get_llm(llm_name)
    chain_1 = (
        LABEL_CLUSTERS_MULTI_PROMPT |
        llm |
        RunnableParallel(
            theme_label=DotAllRegexParser(regex=r'<theme_label>(.*?)</theme_label>', output_keys=['theme_label']),
            theme_label_explanation=DotAllRegexParser(regex=r'<theme_label_explanation>(.*?)</theme_label_explanation>', output_keys=['theme_label_explanation'])
        )
    )
    
    chain_filter = (
        LABEL_FILTER_PROMPT |
        llm |
        RunnableParallel(
            theme_label=DotAllRegexParser(regex=r'<remaining_actions>(.*?)</remaining_actions>', output_keys=['remaining_actions']),
            theme_label_explanation=DotAllRegexParser(regex=r'<important_action>(.*?)</important_action>', output_keys=['important_action'])
        )
    )
    
    chain_conclude = (
        LABEL_CONCLUDE_PROMPT |
        llm |
        RunnableParallel(
        theme_label=DotAllRegexParser(regex=r'<theme_label>(.*?)</theme_label>', output_keys=['theme_label']))
    )
    
    # create topic --> utterance_id map
    top_utt_id_map = collections.defaultdict(lambda: -1)
    top_utt_map = collections.defaultdict(lambda: -1)
    top_ls = list()

    for i, top in enumerate(topics):
        if top_utt_id_map[top] == -1:
            top_utt_id_map[top] = top_ids[i]
        else:
            topics[i] = top + f"(new{i})"
            top_utt_id_map[top + f"(new{i})"] = top_ids[i]

        if top not in top_ls:
            top_ls.append(top)
            top_utt_map[top] = utterances[i]
        else:
            topics[i] = top + f"(new{i})"
            top_ls.append(top)
            top_utt_map[ topics[i]] = utterances[i]

    clustered_utterances = [[] for _ in range(n_clusters)]
    labels_ls = list()
    for i, label in enumerate(clusters_with_preferences):
        labels_ls.append(label)
        clustered_utterances[label].append(topics[i])
    
    labels_ls = list(set(labels_ls))
    
    num_top = len(clusters_with_preferences)
    num_label = len(labels_ls)
    avg_tops_per_label = round(num_top/num_label)
    num_per_group = round(avg_tops_per_label/3)
        
    cluster_label_map = {}
    for label, clusters in tqdm(enumerate(clustered_utterances)):
        clusters_batchs = [clusters[i:i+num_per_group] for i in range(0, len(clusters), num_per_group)]
        # clusters_batchs = [clusters[i:i+1000] for i in range(0, len(clusters), 1000)]
        outputs_list = list()

        # step 1: generate multi-labels
        for cluster in clusters_batchs:
            
            try: 
                outputs_label = chain_1.invoke({'utterances': '\n'.join(cluster)})
                if outputs_label['theme_label']['theme_label'] != 'your theme label':
                    outputs_label_str = outputs_label['theme_label']['theme_label']
                    outputs_list.append(outputs_label_str)
            except:
                print('generation or parse error')
                logging.debug('generation or parse error')
                outputs_label = {'theme_label': {'theme_label':'Default Label'}, 'theme_label_explanation':{'theme_label_explanation': 'No explanation available.'}}
                outputs_label_str = outputs_label['theme_label']['theme_label'] 
                outputs_list.append(outputs_label_str)
            print(f"sub_cluster:{outputs_label['theme_label']['theme_label']}")
            a = f"sub_cluster:{outputs_label['theme_label']['theme_label']}"
            logging.debug(a)
            
            
            
        # step 2: exclude irrelevant labels
        try:
            outputs_filtered = chain_filter.invoke({'theme_label': '\n'.join(outputs_list)})
            remaining_list = outputs_filtered['theme_label']['remaining_actions'].split(",")
            filtered_result =  remaining_list
        except:
            print('generation or parse error')
            # if fail to parse, get the second last sub_cluster label, and emphasize it twice time.
            filtered_result =  [outputs_label['theme_label']['theme_label'], outputs_label['theme_label']['theme_label']]
        print(f"filtered_result: {filtered_result}")
        a = f"filtered_result: {filtered_result}"
        logging.debug(a)
            
        # step 3: conclude an unified label
        try:
            outputs_parsed = chain_conclude.invoke({'theme_label': '\n'.join(filtered_result)})
            if outputs_parsed['theme_label']['theme_label'] == "your_theme_label":
                outputs_parsed['theme_label']['theme_label'] = outputs_label['theme_label']['theme_label']
        # if fail to parse output, get the last generated label as theme_label
        except:
            outputs_parsed = {'theme_label': {'theme_label':filtered_result[0]}, 'theme_label_explanation':{'theme_label_explanation': outputs_label['theme_label_explanation']['theme_label_explanation']}}
        print(outputs_parsed)
        logging.debug(outputs_parsed)
        
        
        # for utt in clusters:
        #     cluster_label_map[utt] = outputs_parsed['theme_label']['theme_label']
        for top in clusters:
            utt = top_utt_map[top]
            cluster_label_map[utt] = outputs_parsed['theme_label']['theme_label']
    return cluster_label_map
