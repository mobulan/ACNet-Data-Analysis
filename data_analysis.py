import matplotlib.pyplot as plt

from data_processing import *

with open('dataset/base_train_log.txt', 'r') as file:
	base_train_data = data_preparation(file)
with open('dataset/acb_train_log.txt', 'r') as file:
	acb_train_data = data_preparation(file)
with open('dataset/vc_acb_log.txt', 'r') as file:
	vgg_acb_data = data_preparation(file)

models = ['cifar-quick','VGG']
models_acb = ['cifar-quick-base','cifar-quick-acb','VGG-acb']
data = [base_train_data, acb_train_data, vgg_acb_data]

# 图像尺寸
plt.figure(figsize=(8, 5))
fig,ax = plt.subplots()

# Top-1 准确率

plt.axis([0,150,0,100])
ax.plot(base_train_data['epoch'], base_train_data['top1'], color='peru', linestyle='--', label="cifar-quick_base")
ax.plot(acb_train_data['epoch'], acb_train_data['top1'], '--', label="cifar-quick_acb")
ax.plot(vgg_acb_data['epoch'], vgg_acb_data['top1'], color='g', linestyle='--', label="VGG_acb")
plt.title("Top-1 Accuracy", fontsize=16)
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Epoch", fontsize=12)
plt.legend(loc='lower right')
save_fig('Top-1 Accuracy')
fig.show()

# Top-5 准确率
fig,ax = plt.subplots()
plt.axis([0,150,50,100])
ax.plot(base_train_data['epoch'], base_train_data['top5'], color='peru', linestyle='--', label="base_top5")
ax.plot(acb_train_data['epoch'], acb_train_data['top5'], '--', label="acb_top5")
ax.plot(vgg_acb_data['epoch'], vgg_acb_data['top5'], color='g', linestyle='--', label="VGG_acb")
plt.title("Top-5 Accuracy", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.legend(loc='best')
save_fig('Top-5 Accuracy')
fig.show()

# 损失曲线
fig,ax = plt.subplots()
plt.axis([0,150,0,3])
ax.plot(base_train_data['epoch'], base_train_data['loss'], color='peru', linestyle='--', label="base_loss")
ax.plot(acb_train_data['epoch'], acb_train_data['loss'], color='steelblue', linestyle='--', label="acb_loss")
ax.plot(vgg_acb_data['epoch'], vgg_acb_data['loss'], color='g', linestyle='--', label="VGG_acb")
plt.title("Loss", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(loc='best')
save_fig('Loss')
fig.show()

# 训练时间增加比率
fig,ax = plt.subplots()
train_time = [17.8155, 34.1450, 96.5, 209.2651]
time_increase = [(train_time[1] / train_time[0] - 1) * 100, (train_time[3] / train_time[2] - 1) * 100]
ax.bar(models, time_increase)
plt.title('Training time increse by using acb', fontsize=16)
plt.ylabel('incrasing rate', fontsize=12)
# 添加数值标签
for a,b in zip(models, time_increase):
	plt.text(a, b+0.05, '%.2f' % b, ha='center', va= 'bottom',fontsize=10)
save_fig('Training time increse')
fig.show()

# 最后Top-1准确率
fig,ax = plt.subplots()
final_accuracy = []
for i in data:
	final_accuracy.append(i.iloc[-1,i.columns.get_loc('top1')])
ax.bar(models_acb,final_accuracy)
plt.ylim(80)
plt.title('Final Top-1 Accuarcy', fontsize=16)
# 添加数值标签
for a,b in zip(models_acb,final_accuracy):
	plt.text(a, b+0.05, '%.2f' % b, ha='center', va= 'bottom',fontsize=10)
save_fig('Final Top-1 Accuracy')
fig.show()

# 使用ACB后准确率提升率
fig,ax = plt.subplots()
acc_increase = (final_accuracy[1]-final_accuracy[0],final_accuracy[2]-94)
ax.bar(models, acc_increase)
for a,b in zip(models, acc_increase):
	plt.text(a, b+0.007, '%.2f' % b, ha='center', va= 'bottom',fontsize=10)
plt.title('Top-1 Accuarcy increase by using acb', fontsize=16)
plt.ylabel('incrasing rate', fontsize=12)
save_fig('Top-1 Accuarcy increase')
fig.show()
