import random
import json
import numpy as np

class data_loader():
    def __init__(self, data_list) -> None:
        self.data = []
        for data_path in data_list:
            if 'dialogue' not in data_path:
                self.data += json.load(open(data_path, 'r', encoding='utf-8'))
            else:
                self.dialogue = json.load(open(data_path, 'r', encoding='utf-8'))
    
    def get_data_input(self):
        self.inputs =  []
        self.attention_mask = []
        self.token_type_ids = []

        for k in self.data:
            text = [h for h in k['text']] 
            answer =  [h for h in k['answer']]
            input = ['[CLS]'] +  text+ ['[SEP]'] + answer + ['[SEP]']
            mask = np.ones(len(input), dtype=np.int32) 
            segment = np.zeros(len(input), dtype=np.int32)
            j = input.index('[SEP]')
            segment[j+1:] = 1

            if len(input) < 555:
                self.inputs.append(input)
                self.attention_mask.append(mask)
                self.token_type_ids.append(segment)
        '''
        多轮对话数据处理
        [CLS]X1[SEP]X2[SEP]
        '''
        for d in self.dialogue:
            dia_input = ['[CLS]']
            dia_segment = [0]
            for a, line in enumerate(d):
                dia_input += [l  for l in line] + ['[SEP]']
                dia_segment += [a for _ in range(len(dia_input))]
            
            dia_mask = np.ones(len(dia_input), dtype=np.int32)
            dia_segment = np.array(dia_segment, dtype=np.int32)

            if len(dia_input) < 600:
                self.inputs.append(dia_input)
                self.attention_mask.append(dia_mask)
                self.token_type_ids.append(dia_segment)

        c = list(zip(self.inputs, self.attention_mask, self.token_type_ids))
        random.shuffle(c)
        self.inputs, self.attention_mask, self.token_type_ids = zip(*c)
        print('data loading')
        return self.inputs, self.attention_mask, self.token_type_ids
    
    def get_batch(self, ids, masks, segments,  global_batch_size):
        batch_num = len(ids) // global_batch_size
        print('batch_num', str(batch_num))
        for i in range(batch_num):
            input = ids[global_batch_size * i: global_batch_size * (i + 1)]
            mask = masks[global_batch_size * i: global_batch_size * (i + 1)]
            segment = segments[global_batch_size * i: global_batch_size * (i + 1)]
            yield input, mask, segment




# import os
# file_list = ['./new_data/'+key for key in os.listdir('./new_data')]
# Data_loader = data_loader(['new_data/dialogue_train_data.json'])
# inputs,mask, segment = Data_loader.get_data_input()


# print(len(inputs))
# print(inputs[100:103])

