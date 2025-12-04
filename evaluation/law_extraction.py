import json,re
import chinese2digits as c2d
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append('segment')
from segment.data_segment_xingshi import DataSegmentXingshi

def get_reason(doc): # 截取doc的“判决”部分
    parser = DataSegmentXingshi(punctuation_replace=True)
    result = parser.parse(doc)
    return result['reason']

def get_penalcode_index_from_text(full_doc):
    doc = get_reason(full_doc)
    patterns = [
        r"《中华人民共和国刑法》第.*?[。《判附规]",  # 匹配《中华人民共和国刑法》第xx条到特定关键词
        r"《刑法》第.*?[。《判附规]",             # 匹配简称《刑法》第xx条到特定关键词
        r"《中华人民共和国刑法》\s?[零一二三四五六七八九十第]+.*?[。《判附规]"  # 匹配《中华人民共和国刑法》后跟空格和数字到特定关键词
    ]
    matches = set()  # 用 set 来去重
    for pattern in patterns:
        matches.update(re.findall(pattern, doc))
    
    if len(matches) == 0:
        doc = full_doc
        for pattern in patterns:
            matches.update(re.findall(pattern, doc))
    # print(matches)
    # 从上述《刑法》第xx条中，获取具体条目编号
    ret = set()
    for match in matches:
        nums = get_num_from_text(match)
        for num in nums:
            ret.add(num) 
    
    ret = list(ret)
    return ret

def get_num_from_text(doc): # 这里是一个简化了的处理方法，匹配所有“第xxx条”并转换成阿拉伯数字
    # pattern = r"[》、]第[一二三四五六七八九零十百]条"
    pattern = r"第[一二三四五六七八九零十百]+条"
    matches = re.findall(pattern, doc)
    ret = []
    for match in matches:
        try:
            # 尝试转换中文数字为阿拉伯数字
            converted_list = c2d.takeNumberFromString(match)['digitsStringList']
            assert len(converted_list) == 1
            ret.append(converted_list[0])
        except Exception as e:
            print(f"跳过不符合格式的匹配: {match}, 错误: {e}")
            pass 
            # 如果转换失败，自动跳过
            
    ret = list(set(ret))
    return ret

def build_law_corpus(law_corpus_file): # 建立law_corpus，提取出来的不在law_corpus中的不考虑。
    law_corpus = []
    with open(law_corpus_file, 'r') as infile:
        for line in infile:
            tmp = json.loads(line.strip())
            law_corpus.append(tmp['name'])
    
    return law_corpus

_BASE_DIR = Path(__file__).resolve().parent
# law_corpus.jsonl 位于仓库 data 目录，而非 evaluation/data
law_corpus_file = _BASE_DIR.parent / 'data' / 'law_corpus.jsonl'
law_corpus = build_law_corpus(law_corpus_file)

if __name__ == '__main__': 
    file = '/home/swh/ybq/casegen/process/input/multi/vx/qwen2.5-7B-Instruct.json'
    # file = 'input/finetune/v4/qwen2.5-7B-Instruct.json'
    print(f"当前文件是{file}")
    with open(file,'r',encoding='utf-8') as myFile:
        data = json.load(myFile)
        diff_sum = 0
        duole = 0 # 生成的比应有的法条多
        tot_len_gen = tot_len_exp = 0
        for i, item in enumerate(data):
            exp = item['exp_ans']
            gen = item['gen_ans']
            
            exp_rsn = get_reason(exp)
            gen_rsn = get_reason(gen)
            
            # a = get_penalcode_index_from_text(content)
            exp_b = get_penalcode_index_from_text(exp)
            gen_b = get_penalcode_index_from_text(gen)
            
            # print(f"=================={i}=========================")
            # print(exp_rsn)
            print(exp_b)
            # print(gen_rsn)
            print(gen_b)
            
            # tot_len_gen += len(gen_b)
            # tot_len_exp += len(exp_b)
            # if exp_b != gen_b:
            #     if len(gen_b) > len(exp_b):
            #         duole += 1
            #     diff_sum += 1
            #     # print(f"当前是第{i}条")
            #     # print('exp: ', exp_rsn)
            #     print(exp_b)
            #     # print('-' * 100)
            #     # print('gen: ', gen_rsn)
            #     print(gen_b)
            #     # print('*' * 100)
            #     # print('*' * 100)
                
        # print(f"共计{diff_sum}条提取出了不同的法条")
        # print(f"共计{duole}条生成了多余的法条")
        # print(f"生成的法条共计{tot_len_gen}条, 预期的法条共计{tot_len_exp}条")
