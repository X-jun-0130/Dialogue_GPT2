from transformers import BertTokenizer, TextGenerationPipeline,TFGPT2LMHeadModel
import tensorflow as tf
from data_helper import data_loader
from rouge import Rouge
import logging
import os
import numpy as np
logging.disable(30)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


lr = 1e-5
epsilon = 1e-6
num_epochs = 10
n = 688860
# file_list = ['./new_data/'+key for key in os.listdir('./new_data')]
Data_loader = data_loader(['new_data/dialogue_train_data.json'])
input_ids, mask_ids, segment_ids = Data_loader.get_data_input()
print(len(input_ids))
inputs, masks, segments = input_ids[:n], mask_ids[:n], segment_ids[:n]
#拆分训练集和测试集
# masks和segments矩阵并没有用到 可删掉


#gpt2配置
checkpoint_path = "./bert_model/gpt-base/"
tokenizer = BertTokenizer.from_pretrained(checkpoint_path, lowercase=True, add_special_tokens=True)
#创建一个MirroredStrategy分发数据和计算图
strategy = tf.distribute.MirroredStrategy()
batch_size = 8
# 同时训练的Batch大小应该为：单个GPU的Batch * GPU个数
global_batch = batch_size * strategy.num_replicas_in_sync



def padding(inputs):
    max_len_input = max([len(key) for key in inputs])
    input = [tokenizer.convert_tokens_to_ids(token) for token in inputs]
    input = tf.keras.preprocessing.sequence.pad_sequences(input, max_len_input, padding='post', truncating='post')
    # mask = tf.keras.preprocessing.sequence.pad_sequences(masks, max_len_input, padding='post', truncating='post')
    # segment = tf.keras.preprocessing.sequence.pad_sequences(segments, max_len_input, padding='post', truncating='post')
    labels = input
    # labels[labels[: ,:] == 0 ] = -100 
    labels = tf.cast(labels, dtype=tf.float32)
    input = tf.cast(input, dtype=tf.int32)
    # mask = tf.cast(mask, dtype=tf.int32)
    # segment = tf.cast(segment, dtype=tf.int32)
    # labels = target[:, 1:]

    num_train = input.shape[0]
    db_train = tf.data.Dataset.from_tensor_slices((input, labels))
    db_train = db_train.shuffle(num_train).batch(global_batch, drop_remainder=True)
    return db_train

# def evaluate():
#     rouge = Rouge()
#     r =  1e-10
#     dev_inputs = input_ids[n:]
#     label_gt_new, label_pr_new = [], []
#     for i in range(len(dev_inputs)):
#         j = dev_inputs[i].index('[SEP]')
#         token = ''.join(dev_inputs[i][1:j])
#         try:
#             summary_ids = text_generator(token, max_length= len(token) + 250, repetition_penalty=1.3, do_sample=True, top_k=10, early_stopping=True)
#             generate_text = summary_ids[0]['generated_text'].replace(token, '')
#             label_gt_new.append(' '.join([k for k in generate_text]))
#             label_pr_new.append(' '.join(dev_inputs[i][j+1:-1]))
#         except:
#             pass
#     try:
#         metrics = rouge.get_scores(label_gt_new, label_pr_new, avg=True)
#         r = 100 * metrics['rouge-l']['r']
#     except:
#         pass
#     return  r
# def pred(input_sen):
#     summary_tokens = []
#     try:
#         summary_ids = text_generator(input_sen, max_length= len(input_sen) + 250, repetition_penalty=1.3, do_sample=True, top_k=10, early_stopping=True)
#         summary_tokens = summary_ids[0]['generated_text'].replace(' ', '')
#     except:
#         pass
#     return summary_tokens
def evaluate():
    rouge = Rouge()
    r =  1e-10
    dev_inputs = input_ids[n:]
    label_gt_new, label_pr_new = [], []
    for i in range(len(dev_inputs)):
        token = dev_inputs[i][:-1]
        x = [j for j, word in enumerate(token) if word == '[SEP]' ][-1]
        input = token[:x+4]
        label_pr_new.append(' '.join(dev_inputs[i][x+4:-1]))
        gene_str = ''
        for i in range(50):
            ids = tokenizer.convert_tokens_to_ids(input)
            ids = tf.constant(ids, dtype=tf.int32)[None, :]
            output = model(ids)
            logits = output.logits
            last_token_id = int(np.argmax(logits[0][-1].numpy()))
            if last_token_id == 102:
                break
            last_token = tokenizer.convert_ids_to_tokens(last_token_id)
            input.append(last_token)
            gene_str += last_token
        label_gt_new.append(' '.join([k for k in gene_str]))

    try:
        metrics = rouge.get_scores(label_gt_new, label_pr_new, avg=True)
        r = 100 * metrics['rouge-l']['r']
    except:
        pass
    return  r

def pred(input_sen):
    gene_str = ''

    for i in range(50):
        ids = tokenizer.convert_tokens_to_ids(input_sen)
        ids = tf.constant(ids, dtype=tf.int32)[None, :]
        output = model(ids)
        logits = output.logits
        last_token_id = int(np.argmax(logits[0][-1].numpy()))
        if last_token_id == 102:
            break
        last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        input_sen.append(last_token)
        gene_str += last_token

    return gene_str


sen_list = [['[CLS]', '患', '者', ':', '疾', '病', '：', '腺', '样', '体', '肥', '大', ' ', '内', '容', '：', '病', '情', '描', '述', '（', '主', '要', '症', '状', '、', '发', '病', '时', '间', '）', '：', '我', '女', '儿', '3', '3', '个', '月', '，', '白', '天', '的', '时', '候', '总', '是', '说', '鼻', '子', '难', '受', '，', '受', '挖', '鼻', '子', '，', '晚', '上', '睡', '觉', '打', '鼾', '。', '呼', '吸', '很', '重', '。', '曾', '经', '治', '疗', '情', '况', '和', '效', '果', '：', '在', '我', '们', '县', '人', '民', '医', '院', '看', '过', '，', '怀', '疑', '是', '腺', '样', '体', '肥', '大', '，', '但', '没', '有', '做', '检', '查', '，', '医', '生', '配', '了', '盐', '酸', '羟', '甲', '唑', '啉', '喷', '雾', '剂', '，', '用', '了', '有', '一', '个', '星', '期', '，', '张', '嘴', '呼', '吸', '好', '了', '一', '些', '，', '但', '其', '它', '的', '还', '是', '一', '样', '。', '想', '得', '到', '怎', '样', '的', '帮', '助', '：', '想', '知', '道', '到', '底', '是', '不', '是', '腺', '样', '体', '肥', '大', '，', '需', '要', '做', '什', '么', '样', '的', '检', '查', '？', '这', '么', '小', '好', '做', '检', '查', '吗', '？', '如', '果', '是', '怎', '么', '治', '？', '化', '验', '、', '检', '查', '结', '果', '：', '怀', '疑', '是', '腺', '样', '体', '肥', '大', '最', '后', '一', '次', '就', '诊', '的', '医', '院', '：', '长', '兴', '人', '民', '医', '院', '[SEP]', '患', '者', ':', '谢', '谢', '，', '在', '浙', '江', '是', '否', '可', '以', '做', '这', '样', '的', '摄', '片', '？', '[SEP]', '患', '者', ':'], 
            ['[CLS]', '患', '者', ':', '疾', '病', '：', '。', '颈', '椎', '病', '。', '所', '就', '诊', '医', '院', '科', '室', '：', '。', '江', '西', '骨', '科', '。', '检', '查', '及', '化', '验', '：', '。', '颈', '椎', '正', '常', '生', '理', '曲', '度', '尚', '好', '。', '诸', '椎', '体', '可', '见', '少', '许', '骨', '质', '增', '生', '。', '颈', '椎', '椎', '间', '盘', 'T', '2', '信', '号', '稍', '减', '低', '。', '颈', '椎', '轻', '度', '退', '变', '，', '椎', '间', '盘', '轻', '度', '变', '性', '。', 'C', '3', '/', '4', 'C', '4', '/', '5', 'C', '5', '/', '6', 'C', '6', '/', '7', '椎', '间', '盘', '突', '出', '，', '相', '应', '水', '平', '脊', '膜', '嚷', '受', '压', '变', '形', '，', '呈', '现', '弧', '形', '压', '迹', '。', 'C', '3', '/', '4', 'C', '4', '/', '5', 'C', '5', '/', '6', '段', '脊', '髓', '未', '见', '明', '显', '受', '压', '，', 'T', '2', 'W', '信', '号', '明', '显', '增', '高', '，', '第', '6', '颈', '椎', '椎', '体', '见', '有', '片', '状', '短', 'T', '1', '长', 'T', '2', '信', '号', '影', '，', '压', '脂', 'T', '2', 'W', '上', '呈', '明', '显', '高', '信', '号', '。', '第', '6', '颈', '椎', '椎', '体', '异', '常', '信', '号', '影', '考', '虑', '为', '血', '管', '瘤', '。', '颈', '椎', '3', '椎', '体', '后', '滑', '脱', '。', '右', '上', '肢', '麻', '木', '不', '适', '2', '月', '余', ',', '右', '上', '肢', '致', '右', '手', '各', '指', '麻', '木', '不', '适', '，', '感', '无', '力', ',', '右', '下', '肢', '行', '走', '异', '常', ',', '感', '无', '力', ',', '左', '侧', '上', '肢', '较', '右', '侧', '症', '状', '轻', '。', '左', '下', '肢', '偶', '感', '发', '热', ',', '大', '小', '便', '正', '常', ',', '胸', '部', '无', '束', '带', '感', ',', '左', '侧', 'h', 'o', 'f', 'f', 'm', 'a', 'n', '征', '弱', '阳', '性', '，', '右', '侧', 'h', 'o', 'f', 'f', 'm', 'a', 'n', '阳', '性', '，', '双', '上', '肢', '力', '体', '量', '尚', '可', '。', '颈', '椎', '3', '椎', '体', '后', '滑', '落', '，', '颈', '椎', '体', '，', '3', '/', '4', '4', '/', '5', '5', '/', '6', '6', '/', '7', '椎', '肩', '盘', '突', '出', '。', '第', '6', '椎', '椎', '体', '异', '常', '信', '号', '影', '考', '虑', '为', '血', '管', '瘤', '。', '颈', '椎', '失', '稳', '。', '。', '治', '疗', '情', '况', '：', '。', '发', '现', '这', '个', '颈', '椎', '病', '已', '经', '2', '个', '多', '月', '了', '，', '开', '了', '点', '药', '吃', '的', '没', '一', '点', '效', '果', '。', '前', '不', '久', '刚', '做', '过', '静', '脉', '曲', '张', '手', '术', '。', '几', '年', '前', '血', '压', '比', '较', '低', '，', '常', '常', '头', '晕', '。', '。', '病', '史', '：', '。', '几', '年', '前', '血', '压', '比', '较', '低', '，', '常', '常', '头', '晕', '。', '想', '得', '到', '怎', '样', '的', '帮', '助', '：', '（', '感', '谢', '医', '生', '为', '我', '快', '速', '解', '答', '—', '—', '该', '如', '何', '治', '疗', '和', '预', '防', '。', '）', '最', '好', '是', '能', '不', '能', '避', '免', '不', '做', '手', '术', '就', '不', '做', '手', '术', '。', '能', '不', '能', '开', '点', '有', '效', '的', '好', '药', '方', '子', '告', '诉', '我', '。', '本', '人', '万', '分', '感', '谢', '！', '[SEP]', '医', '生', ':']]
best = 10
with strategy.scope():
    # 定义模型、优化器、检查点
    model = TFGPT2LMHeadModel.from_pretrained(checkpoint_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, gen_model=model)
    def compute_loss(input_ids,labels):
        loss = model(input_ids=input_ids,  labels=labels).loss
        return tf.nn.compute_average_loss(loss, global_batch_size=global_batch)

with strategy.scope():
    def train_step(input_ids,  labels):
        with tf.GradientTape() as tape:
            loss = compute_loss(input_ids,  labels)
        grads = tape.gradient(loss, model.trainable_variables)
        # grads, _ = tf.clip_by_global_norm(grads, tf.cast(clip, tf.float32))
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

with strategy.scope():
    @tf.function 
    def distributed_train_step(dataset_inputs,  dataset_labels):
        per_replica_losses = strategy.experimental_run_v2(
            train_step, args=(dataset_inputs,  dataset_labels)
        )
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    

batch_index = 1
text_generator = TextGenerationPipeline(model, tokenizer)
with strategy.scope():
    for epoch in range(num_epochs):
        ave_loss = []
        print('Epoch:', epoch + 1)
        for input, mask, segment in Data_loader.get_batch(inputs, masks, segments, 229620): #分3次读入总数剧，减少内存开销
            dataset = padding(input)
            disset = strategy.experimental_distribute_dataset(dataset) #分发数据
            for record in disset:
                batch_index += 1
                pad_input,  labels = record
                loss = distributed_train_step(pad_input,  labels)
                ave_loss.append(loss)
                if (batch_index) % 2870 == 0:
                    print('Batch {} Loss {:.4f}'.format(batch_index, loss))

        print('\n')
        print('Ave_Loss {:.4f}'.format(np.mean(ave_loss)))
        print('\n')

        r = evaluate()
        print(r)

        for sen in sen_list:
            print(pred(sen))
            print('-----')

        if r > best:
            best = r
            print('saving_model')
            checkpoint.save('./save/genaration_model.ckpt')  
       

