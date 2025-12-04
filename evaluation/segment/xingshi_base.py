# coding=utf-8
import os
from glob import glob
import traceback
import logging
import regex as re
import warnings
import cn2an
import pkg_resources


class DataSegmentXingshiBase():
    ALL_FIELDS = ["heading", "fact", "reason", "judgment", "appendix"]

    def __init__(self, punctuation_replace=True):
        self.punctuation_replace = punctuation_replace
        self.jiezhibiaodian = ["。", "）", ")", "”", "\n", "？"] # 截止到标点

    def punctuation_replace_fun(self, data): # 将英文标点全部替换成中文标点
        from bs4 import BeautifulSoup
        data = data.replace(':', '：')
        data = data.replace(' ', '')
        data = data.replace('(', '（')
        data = data.replace(')', '）')
        data = data.replace('\u3000', ' ')
        data = data.replace('\xa0', ' ')
        data = data.replace('<p>', '')
        data = data.replace('</p>', '')
        data = data.replace('<br>', '')
        data = data.replace('&#xD;', '')
        data = data.replace("\uFFFD", "")
        data = re.sub(r'(?<!\d),(?!\d)', '，', data) # 除了数字之间的,不改，其它的改成中文逗号

        soup = BeautifulSoup(data, "html.parser") # 去除所有html结构
        for data in soup(['style', 'script']):
            # Remove tags
            data.decompose()
        data = ' '.join(soup.stripped_strings)

        return data

    def parse(self, wenshu):
        if self.punctuation_replace:
            wenshu = self.punctuation_replace_fun(wenshu)
        wenshu = {"content": wenshu}
        # current content用于存储逐段删除后的剩余内容
        wenshu["current_content"] = wenshu["content"]

        for field in self.ALL_FIELDS:
            eval(f"self._set_{field}(wenshu)")

        del wenshu['current_content']
        return wenshu

    def del_fun(self, wenshu, field): # 如果 wenshu[field] 非空，则从 wenshu["current_content"] 中删除该字段内容。
        if wenshu[field].strip():
            wenshu["current_content"] = wenshu["current_content"].replace(wenshu[field], '')

    def text_end_itertools_min(self, end_list, content, end_supplement='?='): # 在 content 里查找 end_list 里所有可能的结束词，并返回最短匹配文本。
        return_text = ''
        min_len = 100000
        for pe in end_list:
            pattern_text = fr'.*?({end_supplement}{pe})'
            text_search = re.search(pattern_text, content, re.DOTALL)
            # print(text_search)
            if text_search:
                current_text = text_search.group()
                if current_text == "":
                    return ""
                tem_text = re.sub(r'\s+', '', current_text, re.DOTALL)
                if tem_text != '' and len(current_text) < min_len:
                    return_text = current_text
                    min_len = len(return_text)
        return return_text

    def text_end_itertools(self, end_list, content, end_supplement='?='): # 从 content 里查找 end_list 里的词，并返回第一个匹配项（而非最短匹配项）。
        return_text = ''
        for pe in end_list:
            pattern_text = fr'.*?({end_supplement}{pe})'
            text_search = re.search(pattern_text, content, re.DOTALL)
            if text_search:
                return_text = text_search.group()
                break
        return return_text
