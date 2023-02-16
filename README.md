# Dialogue_GPT2
对话生成、多轮对话、GPT2微调、tensorflow2、gpu并行

# 任务类型
gpt2多轮对话。数据来自网络医疗对话

# 数据类型
```
[ "患者:疾病：小孩语言表达。病情描述：4岁半，女，现在上幼...。希望提供的帮助：怎么治疗（不能少于10个字",
  "医生:学习语言需要一个过程，可以请幼儿老师帮助训练...，有的对语言训练效果很好。",
  "患者:而且咬字很不清楚",
  "医生:咬字不清楚是因为...需要通过训练来纠正。"]
```
整理成：'[CLS]患者:X1[SEP]医生:X2[SEP]'  输入模型

预测时：'[CLS]患者:X1[SEP]医生:X2[SEP]患者:' 输入模型预测患者接下来的话
