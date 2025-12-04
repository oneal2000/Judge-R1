import json,os,re
from tqdm import tqdm
import chinese2digits as c2d
import sys
sys.path.append('segment')
from segment.data_segment_xingshi import DataSegmentXingshi
judge_list = ["管制", "拘役", "有期徒刑", "罚金", "无期徒刑", "死刑", "无罪", "免予刑事处罚", "免于刑事处罚", "免予刑事处分"]

def get_judgment(doc): # 截取doc的“判决”部分
    parser = DataSegmentXingshi(punctuation_replace=True)
    result = parser.parse(doc)
    return result['judgment']

def get_time_string_from_text(doc): # 提取(包含刑期)的完整字符串
    # print('\n', doc)
    # 修正后的正则表达式模式
    ret = []
    for judge in judge_list[:3]:
        pattern = re.compile(rf'{judge}.{{1,7}}[年月]') # judge一直匹配到‘年’字或者‘月’字
        matches = re.findall(pattern, doc)
        ch_punct_pattern = re.compile(r'[,;，。！？、；：（以缓至]') # 截取到标点符号/“缓刑”之前
        for i in range(len(matches)):
            match = matches[i]
            ch_punct_pos = ch_punct_pattern.search(match)
            if ch_punct_pos:
                matches[i] = match[:ch_punct_pos.start()]
        
        ret += matches

    for judge in judge_list[4:]:
        pattern = re.compile(rf'{judge}')
        matches = re.findall(pattern, doc)
        ret += matches

    return ret

def get_amt_string_from_text(doc): # 提取包含罚金金额的完整字符串
    pattern = re.compile(rf'罚金.{{1,15}}元') # 一直匹配到‘元’字
    matches = re.findall(pattern, doc)
    ch_punct_pattern = re.compile(r'[，。！？、；：以已至（]') # 截取到标点符号之前
    for i in range(len(matches)):
        match = matches[i]
        ch_punct_pos = ch_punct_pattern.search(match)
        if ch_punct_pos:
            matches[i] = match[:ch_punct_pos.start()]
    for match in matches:
        if not '元' in match:
            matches.remove(match)
    for i in range(len(matches)):
        matches[i] = matches[i].replace(",", "")
    return matches

def get_time_from_text(doc):
    # print('-' * 80)
    full_doc = doc
    doc = get_judgment(doc)
    ret = get_time_string_from_text(doc)
    if len(ret) == 0:
        ret = get_time_string_from_text(full_doc)
    
    ret = list(set(ret))
    # print(ret)
    return ret

def get_amt_from_text(doc):
    full_doc = doc
    doc = get_judgment(doc)
    ret = get_amt_string_from_text(doc)
    if len(ret) == 0:
        ret = get_amt_string_from_text(full_doc)
    
    ret = list(set(ret))
    # print(ret)
    return ret

def calc_time_sum(doc):
    all_judge_time_str = get_time_from_text(doc)
    if len(all_judge_time_str) == 0: # 如果没有提取到刑期长度
        return -1
    
    time_sum = 0
    for judge_time_str in all_judge_time_str:
        num_list = c2d.takeNumberFromString(judge_time_str)['digitsStringList']
        num = 0
        if len(num_list) == 2:
            if '年' in judge_time_str and '月' in judge_time_str: # 如果是x年x月的格式
                num = int(num_list[0]) * 12 + int(num_list[1])
            else:
                print('发生错误：', judge_time_str)
                num = int(num_list[0]) # 取第一个
        elif len(num_list) == 1:
            if '年' in judge_time_str:
                num = int(num_list[0]) * 12
            elif '月' in judge_time_str:
                num = int(num_list[0])
        elif len(num_list) == 0:
            if '无期徒刑' in judge_time_str:
                num = 240
            elif '死刑' in judge_time_str:
                num = 10001 # 一会儿只需检查是否返回的数额大于10000，即可知道是否出现死刑了
            else:
                num = 0
        else:
            print('有不合规范的刑期长度：', judge_time_str)
        
        time_sum += num
        
    return time_sum

def calc_amt_sum(doc):
    all_amt_str = get_amt_from_text(doc)
    if len(all_amt_str) == 0: # 如果没有提取到罚金金额
        return -1
    amt_sum = 0
    for amt_str in all_amt_str:
        num_list = c2d.takeNumberFromString(amt_str)['digitsStringList']
        if len(num_list) >= 1:
            # 某些描述包含多段数字（如“一万元二千元”返回两个数），这里取所有数字之和
            try:
                # 支持带小数点的字符串，先转 float 再转 int（取整）
                amt_sum += sum(int(float(x)) for x in num_list)
            except ValueError:
                print('金额格式不对', amt_str)
        else:
            print('金额格式不对', amt_str)
    
    return amt_sum

if __name__ == '__main__':
    file = '/home/swh/ybq/casegen/process/input/multi/vx/qwen2.5-7B-Instruct.json'
    with open(file, 'r', encoding='utf-8') as myFile:
        data = json.load(myFile)
        
        for tmp in data:
            # 获取 gen_ans 和 exp_ans 字段的值
            gen_ans = tmp.get('gen_ans', '')
            exp_ans = tmp.get('exp_ans', '')
            
            # 对 gen_ans 和 exp_ans 分别调用 get_judge_from_text 函数
            time_gen = calc_time_sum(gen_ans)
            time_exp = calc_time_sum(exp_ans)
            amt_gen = calc_amt_sum(gen_ans)
            amt_exp = calc_amt_sum(exp_ans)
            
            # 创建输出字典
            out_dict = {
                'time_gen': time_gen,
                'time_exp': time_exp,
                'amt_gen': amt_gen,
                'amt_exp': amt_exp
            }
            print(out_dict)
