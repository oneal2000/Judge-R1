import regex as re
import traceback
import cn2an
from xingshi_base import DataSegmentXingshiBase
class DataSegmentXingshiGongsuYishenPanjue(DataSegmentXingshiBase):

    def _set_heading(self, wenshu): # 头部的基础信息等
        wenshu["heading"] = ""
        pattern_list = [r'审理终结[\u4e00-\u9fa5]{0,10}。', r'公开开庭审理[\u4e00-\u9fa5]{0,10}。']
        wenshu["heading"] = self.text_end_itertools(pattern_list, wenshu["current_content"], '')
        if re.sub(r'\s+', '', wenshu["heading"]) == '':
            pattern_list = [r'[\u4e00-\u9fa5]{0,10}(公诉|抗诉)机关[\u4e00-\u9fa5，、]{0,30}(认为|指控)']
            wenshu["heading"] = self.text_end_itertools(pattern_list, wenshu["current_content"])
        self.del_fun(wenshu, "heading")

    def _set_fact(self, wenshu): # 经法庭审理查明的事实和据以定案的证据，到“本院认为”为止
        ###fact: 经法庭审理查明的事实和据以定案的证据
        wenshu["fact"] = ""
        pattern = [r'本院认为']
        wenshu["fact"] = self.text_end_itertools(pattern, wenshu["current_content"])

        self.del_fun(wenshu, "fact")

    def _set_reason(self, wenshu): # 裁判理由：包含本院认为、引用法条
        wenshu["reason"] = ''
        pattern = [
            r'判决如下[:：,，\n]',
            r'判决如下',
            r'裁定(如下)?[:：\n]',

        ]
        wenshu["reason"] = self.text_end_itertools(pattern, wenshu["current_content"], '')
        if not len(wenshu["reason"]): # 改成了如果找不到“判决如下”，就把后面所有的都设置成reason
            wenshu["reason"] = wenshu["current_content"]
        self.del_fun(wenshu, "reason")

    def _set_judgment(self, wenshu): # 判决结果
        ###panjuejieguo:判决结果
        pattern_list = [
            r'如[\u4e00-\u9fa5]{0,3}不服本判决',
            r'\n\s*本判决为终审判决',
            r'如[\u4e00-\u9fa5]{0,5}未按本判决指定的期间[\u4e00-\u9fa5]{0,5}给付金钱义务',
            r'\n\s*(代理)?审[\s]{0,3}判[\s]{0,3}长',
            r'\n\s*(代理)?审[\s]{0,3}判[\s]{0,3}员',
            '附录',
            '附：',
            r'附[\u4e00-\u9fa5]{0,10}法律[\u4e00-\u9fa5]{0,10}',
            r'附[\u4e00-\u9fa5]{0,10}：',
            r"\n\s*本案[^\n]{0,10}法律",
            r"\n《",
            '$',
        ]
        wenshu["judgment"] = self.text_end_itertools(pattern_list, wenshu["current_content"])
        if not len(wenshu['judgment']):
            wenshu["judgment"] = wenshu["current_content"]
        self.del_fun(wenshu, "judgment")

    def _set_appendix(self, wenshu): # 尾巴上那些内容
        wenshu["appendix"] = wenshu['current_content']
        self.del_fun(wenshu, "appendix")
        