from collections import OrderedDict
from itertools import compress

import itertools
import ast
import yaml


class yaml_parser():

    def __init__(self, fname):

        self.fname = fname

        org_dict = yaml.safe_load(open(fname))
        self.final_dicts = org_dict['plots']

        if org_dict['cuts'] is not None:
            for key in self.final_dicts.keys():
                if 'cut' in self.final_dicts[key]:
                    self.final_dicts[key]['cut'] = tuple(sti.strip() for sti in self.final_dicts[key]['cut'].split(',')) + tuple(sti.strip() for sti in org_dict['cuts'].split(','))
                else:
                    self.final_dicts[key]['cut'] = tuple(sti.strip() for sti in org_dict['cuts'].split(','))

    @staticmethod
    def parse_var(var):

        if isinstance(var, (int, float, list)):
            return [var]

        try:
            return list(ast.literal_eval(str(var)))
        except (ValueError, TypeError):
            return [sti.strip() for sti in var.split(',')]

    @classmethod
    def get_combs(cls, dic):

        ord_dict = OrderedDict(dic)
        lis = [cls.parse_var(ord_dict[key]) for key in ord_dict.keys()]
        combs = list(itertools.product(*lis))
        dic_list = []
        for tpl in combs:
            litpl = list(tpl)
            dic = {}
            for key in ord_dict.keys():
                dic[key] = litpl[list(ord_dict.keys()).index(key)]
            dic_list.append(dic)

        return dic_list

    @staticmethod
    def clean_dict_list(dict_list):

        mask = [True] * len(dict_list)

        for i in range(len(dict_list)):
            for j in range(i+1, len(dict_list)):
                if dict_list[i] == dict_list[j]:
                    mask[j] = False

        return list(compress(dict_list, mask))

    def __call__(self):
        return self.clean_dict_list(list(itertools.chain.from_iterable([self.get_combs(self.final_dicts[key]) for key in self.final_dicts.keys()])))
