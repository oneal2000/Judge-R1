# -*- coding: utf-8 -*-
import os
import json
import time
import argparse
from xingshi_yishenpanjue_gongsu import DataSegmentXingshiGongsuYishenPanjue


class DataSegmentXingshi():
    
    def __init__(self, punctuation_replace=False):
        self.yishengongsu = DataSegmentXingshiGongsuYishenPanjue(punctuation_replace)

    def parse(self, wenshu):
        wenshu = self.yishengongsu.parse(wenshu)
        return wenshu
