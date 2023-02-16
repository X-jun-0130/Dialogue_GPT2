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
#gpt2中 masks和segments矩阵并没有用到 可删掉


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

# sen_list = ['患者1天前因个人原因自行停用胰岛素，于今晨出现口干、胸闷、气促不适，半天前开始出现恶心、呕吐症状，呕吐物为胃内容物，无呕血解黑便，无明显消瘦，无视物模糊，无头晕、头痛，无视物旋转，无耳鸣，无咳嗽、咳痰、发热，无腹痛、腹泻，无肢体麻木、抽搐，无肢体运动障碍，在外未予特殊处理，病情无好转，遂来我院急诊科就诊，急诊查指尖血糖为26.7 mmol/L，ECG示：窦性心动过速。为进一步诊治，急诊遂以“糖尿病”收住我科。病程中，患者精神可，饮食、睡眠可，大小便基本正常，体力、体重无明显改变。_抽取实体类型:症状_抽取结果:',
#             '怀孕了外阴骚痒需要怎么办',
#             '胰头部见一类圆形稍低密度影，边界显示欠清，大小约25*26mm，增强见病灶轻度强化，并低于胰腺实质。另动脉期胰头部可见一类圆形明显强化灶，大小约20*15mm，静脉期及延迟扫描见病灶密度比胰腺实质稍高。胰管略增宽，胆总管，肝内外胆管明显扩张，胆囊增大，胆囊内胆汁密度增高，另可见多发类圆形高密度影，胆囊壁增厚。肝脏大小形态未见明显异常，表面光整，肝叶比例协调，肝裂未见明显增宽。脾脏不大。双肾见多发大小不等的囊状低密度灶，最大的位于右肾，约57*74mm，增强未见明显强化。左肾周脂肪间隙稍模糊。腹膜后未见明显肿大淋巴结，腹腔未见积液征象。胰头部占位，胰腺癌可能，建议MR检查胰头部动脉期明显强化灶，胰岛细胞瘤？建议MR检查胆囊多发结石，胆囊炎双肾多发囊肿_问题:腹腔未见积液征象属于什么类别？_选项:阴性，阳性_答案:']


sen_list = [['[CLS]', '患', '者', ':', '疾', '病', '：', '腺', '样', '体', '肥', '大', ' ', '内', '容', '：', '病', '情', '描', '述', '（', '主', '要', '症', '状', '、', '发', '病', '时', '间', '）', '：', '我', '女', '儿', '3', '3', '个', '月', '，', '白', '天', '的', '时', '候', '总', '是', '说', '鼻', '子', '难', '受', '，', '受', '挖', '鼻', '子', '，', '晚', '上', '睡', '觉', '打', '鼾', '。', '呼', '吸', '很', '重', '。', '曾', '经', '治', '疗', '情', '况', '和', '效', '果', '：', '在', '我', '们', '县', '人', '民', '医', '院', '看', '过', '，', '怀', '疑', '是', '腺', '样', '体', '肥', '大', '，', '但', '没', '有', '做', '检', '查', '，', '医', '生', '配', '了', '盐', '酸', '羟', '甲', '唑', '啉', '喷', '雾', '剂', '，', '用', '了', '有', '一', '个', '星', '期', '，', '张', '嘴', '呼', '吸', '好', '了', '一', '些', '，', '但', '其', '它', '的', '还', '是', '一', '样', '。', '想', '得', '到', '怎', '样', '的', '帮', '助', '：', '想', '知', '道', '到', '底', '是', '不', '是', '腺', '样', '体', '肥', '大', '，', '需', '要', '做', '什', '么', '样', '的', '检', '查', '？', '这', '么', '小', '好', '做', '检', '查', '吗', '？', '如', '果', '是', '怎', '么', '治', '？', '化', '验', '、', '检', '查', '结', '果', '：', '怀', '疑', '是', '腺', '样', '体', '肥', '大', '最', '后', '一', '次', '就', '诊', '的', '医', '院', '：', '长', '兴', '人', '民', '医', '院', '[SEP]', '患', '者', ':', '谢', '谢', '，', '在', '浙', '江', '是', '否', '可', '以', '做', '这', '样', '的', '摄', '片', '？', '[SEP]', '患', '者', ':'], 
            ['[CLS]', '患', '者', ':', '疾', '病', '：', '。', '颈', '椎', '病', '。', '所', '就', '诊', '医', '院', '科', '室', '：', '。', '江', '西', '骨', '科', '。', '检', '查', '及', '化', '验', '：', '。', '颈', '椎', '正', '常', '生', '理', '曲', '度', '尚', '好', '。', '诸', '椎', '体', '可', '见', '少', '许', '骨', '质', '增', '生', '。', '颈', '椎', '椎', '间', '盘', 'T', '2', '信', '号', '稍', '减', '低', '。', '颈', '椎', '轻', '度', '退', '变', '，', '椎', '间', '盘', '轻', '度', '变', '性', '。', 'C', '3', '/', '4', 'C', '4', '/', '5', 'C', '5', '/', '6', 'C', '6', '/', '7', '椎', '间', '盘', '突', '出', '，', '相', '应', '水', '平', '脊', '膜', '嚷', '受', '压', '变', '形', '，', '呈', '现', '弧', '形', '压', '迹', '。', 'C', '3', '/', '4', 'C', '4', '/', '5', 'C', '5', '/', '6', '段', '脊', '髓', '未', '见', '明', '显', '受', '压', '，', 'T', '2', 'W', '信', '号', '明', '显', '增', '高', '，', '第', '6', '颈', '椎', '椎', '体', '见', '有', '片', '状', '短', 'T', '1', '长', 'T', '2', '信', '号', '影', '，', '压', '脂', 'T', '2', 'W', '上', '呈', '明', '显', '高', '信', '号', '。', '第', '6', '颈', '椎', '椎', '体', '异', '常', '信', '号', '影', '考', '虑', '为', '血', '管', '瘤', '。', '颈', '椎', '3', '椎', '体', '后', '滑', '脱', '。', '右', '上', '肢', '麻', '木', '不', '适', '2', '月', '余', ',', '右', '上', '肢', '致', '右', '手', '各', '指', '麻', '木', '不', '适', '，', '感', '无', '力', ',', '右', '下', '肢', '行', '走', '异', '常', ',', '感', '无', '力', ',', '左', '侧', '上', '肢', '较', '右', '侧', '症', '状', '轻', '。', '左', '下', '肢', '偶', '感', '发', '热', ',', '大', '小', '便', '正', '常', ',', '胸', '部', '无', '束', '带', '感', ',', '左', '侧', 'h', 'o', 'f', 'f', 'm', 'a', 'n', '征', '弱', '阳', '性', '，', '右', '侧', 'h', 'o', 'f', 'f', 'm', 'a', 'n', '阳', '性', '，', '双', '上', '肢', '力', '体', '量', '尚', '可', '。', '颈', '椎', '3', '椎', '体', '后', '滑', '落', '，', '颈', '椎', '体', '，', '3', '/', '4', '4', '/', '5', '5', '/', '6', '6', '/', '7', '椎', '肩', '盘', '突', '出', '。', '第', '6', '椎', '椎', '体', '异', '常', '信', '号', '影', '考', '虑', '为', '血', '管', '瘤', '。', '颈', '椎', '失', '稳', '。', '。', '治', '疗', '情', '况', '：', '。', '发', '现', '这', '个', '颈', '椎', '病', '已', '经', '2', '个', '多', '月', '了', '，', '开', '了', '点', '药', '吃', '的', '没', '一', '点', '效', '果', '。', '前', '不', '久', '刚', '做', '过', '静', '脉', '曲', '张', '手', '术', '。', '几', '年', '前', '血', '压', '比', '较', '低', '，', '常', '常', '头', '晕', '。', '。', '病', '史', '：', '。', '几', '年', '前', '血', '压', '比', '较', '低', '，', '常', '常', '头', '晕', '。', '想', '得', '到', '怎', '样', '的', '帮', '助', '：', '（', '感', '谢', '医', '生', '为', '我', '快', '速', '解', '答', '—', '—', '该', '如', '何', '治', '疗', '和', '预', '防', '。', '）', '最', '好', '是', '能', '不', '能', '避', '免', '不', '做', '手', '术', '就', '不', '做', '手', '术', '。', '能', '不', '能', '开', '点', '有', '效', '的', '好', '药', '方', '子', '告', '诉', '我', '。', '本', '人', '万', '分', '感', '谢', '！', '[SEP]', '医', '生', ':'], 
            ['[CLS]', '患', '者', ':', '疾', '病', '：', '髌', '骨', '错', '位', '手', '术', '。', '病', '情', '描', '述', '：', '徐', '大', '夫', '您', '好', '，', '我', '是', '吉', '林', '延', '吉', '的', '，', '髌', '骨', '错', '位', '想', '手', '术', '。', '我', '小', '姨', '1', '2', '月', '2', '号', '跟', '您', '联', '系', '过', '，', '您', '让', '1', '2', '月', '中', '下', '旬', '再', '提', '醒', '您', '一', '下', '，', '我', '们', '是', '1', '月', '5', '日', '放', '假', '，', '想', '1', '月', '6', '日', '前', '后', '手', '术', '，', '我', '现', '在', '已', '经', '是', '高', '二', '了', '，', '以', '后', '的', '学', '习', '就', '更', '紧', '张', '了', '，', '就', '没', '有', '时', '间', '手', '术', '了', '，', '所', '以', '请', '您', '一', '定', '给', '安', '排', '，', '谢', '谢', '，', '谢', '谢', '您', '了', '。', '（', '想', '1', '月', '6', '日', '前', '后', '手', '术', '）', '。', '希', '望', '提', '供', '的', '帮', '助', '：', '我', '们', '是', '1', '月', '5', '日', '放', '假', '，', '想', '1', '月', '6', '日', '前', '后', '手', '术', '，', '我', '现', '在', '已', '经', '是', '高', '二', '了', '，', '以', '后', '的', '学', '习', '就', '更', '紧', '张', '了', '，', '就', '没', '有', '时', '间', '手', '术', '了', '，', '所', '以', '请', '您', '一', '定', '给', '安', '排', '，', '谢', '谢', '，', '谢', '谢', '您', '了', '。', '（', '想', '1', '月', '6', '日', '前', '后', '手', '术', '）', '[SEP]', '医', '生', ':']]
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
       

