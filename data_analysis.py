from data_processing import *

# Cifar-quick data
with open('dataset/base_train_log.txt', 'r') as file:
	cifarquick_base = data_preparation(file)
with open('dataset/acb_train_log.txt', 'r') as file:
	cifarquick_acb = data_preparation(file)

# VGG data
with open('dataset/vgg_base_log.txt', 'r') as file:
	vgg_base = data_preparation(file)
with open('dataset/vgg_acb_log.txt', 'r') as file:
	vgg_acb = data_preparation(file)

# ResNet56 data
with open('dataset/res56_base_log.txt', 'r') as file:
	res56_base = data_preparation(file)
with open('dataset/res56_acb_log.txt', 'r') as file:
	res56_acb = data_preparation(file)

# wrn16 data
with open('dataset/wrnc16_base_log.txt', 'r') as file:
	wrn16_base = data_preparation(file)

models = ['Cifar-quick', 'VGG', 'ResNet-56', 'WRN-16']
models_3 = ['Cifar-quick', 'VGG', 'ResNet-56']
models_all = ['Cifar-quick-base', 'Cifar-quick-acb', 'VGG-base', 'VGG-acb', 'ResNet-56-base', 'ResNet-56-acb',
              'WRN-16-base']
data = [cifarquick_base, cifarquick_acb, vgg_base, vgg_acb, res56_base, res56_acb, wrn16_base]

# 图像尺寸
plt.figure(figsize=(8, 5))
fig, ax = plt.subplots()

# Top-1 准确率

plt.axis([0, 150, 0, 100])
ax.plot(cifarquick_acb['epoch'], cifarquick_acb['top1'], '--', label="cifar-quick_acb")
ax.plot(vgg_acb['epoch'], vgg_acb['top1'], color='g', linestyle='--', label="VGG_acb")
ax.plot(res56_acb['epoch'], res56_acb['top1'], color='peru', linestyle='--', label="cifar-quick_base")
ax.set_title("Top-1 Accuracy", fontsize=16)
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Epoch", fontsize=12)
plt.legend(loc='lower right')
save_fig('Top-1 Accuracy')
fig.show()

# Top-5 准确率
fig, ax = plt.subplots()
plt.axis([0, 150, 50, 100])
ax.plot(cifarquick_acb['epoch'], cifarquick_acb['top5'], '--', label="acb_top5")
ax.plot(vgg_acb['epoch'], vgg_acb['top5'], color='g', linestyle='--', label="VGG_acb")
ax.plot(res56_acb['epoch'], res56_acb['top5'], color='peru', linestyle='--', label="base_top5")
ax.set_title("Top-5 Accuracy", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.legend(loc='best')
save_fig('Top-5 Accuracy')
fig.show()

# 损失曲线
fig, ax = plt.subplots()
plt.axis([0, 150, 0, 3])
ax.plot(cifarquick_acb['epoch'], cifarquick_acb['loss'], color='steelblue', linestyle='--', label="acb_loss")
ax.plot(vgg_acb['epoch'], vgg_acb['loss'], color='g', linestyle='--', label="VGG_acb")
ax.plot(res56_acb['epoch'], res56_acb['loss'], color='peru', linestyle='--', label="base_loss")
ax.set_title("Loss", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
ax.legend(loc='best')
save_fig('Loss')
fig.show()

# 训练时间增加比率
fig, ax = plt.subplots()
plt.ylim(0, 500)
train_time = [17.8155, 34.1450, 46.97, 124.25, 171.24, 910.2]
time_increase = [(train_time[1] / train_time[0] - 1) * 100, (train_time[3] / train_time[2] - 1) * 100,
                 (train_time[5] / train_time[4] - 1) * 100]
ax.bar(models_3, time_increase, color=['peru', 'steelblue', 'rebeccapurple'])
ax.set_title('Training Time Increse by using acb', fontsize=16)
plt.ylabel('incrasing rate', fontsize=12)
# 添加数值标签
for a, b in zip(models_3, time_increase):
	plt.text(a, b + 0.05, '%.2f' % b + '%', ha='center', va='bottom', fontsize=10)
save_fig('Training time increse')
fig.show()

# 最后Top-1准确率
fig, ax = plt.subplots()
final_accuracy = []
base_accuracy = []
acb_accuracy = []
t = 0
for i in data:
	if t % 2:
		final_accuracy.append(np.max(i['top1']))
		acb_accuracy.append(np.max(i['top1']))
		t += 1
	else:
		final_accuracy.append(np.max(i['top1']))
		base_accuracy.append(np.max(i['top1']))
		t += 1
acb_accuracy.append(np.nan)
x = np.arange(len(models))
width = 0.4
rec1 = ax.bar(x - width / 2, base_accuracy, width, label='base')
rec2 = ax.bar(x + width / 2, acb_accuracy, width, label='acb')
plt.ylim(70)
ax.set_title('Final Top-1 Accuarcy', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models)
autolabel(rec1,ax)
autolabel(rec2,ax)
ax.legend()
save_fig('Final Top-1 Accuracy')
fig.show()

# 使用ACB后准确率提升率
fig, ax = plt.subplots()
plt.ylim(0, 1.1)

acc_increase = (
	final_accuracy[1] - final_accuracy[0], final_accuracy[3] - final_accuracy[2], final_accuracy[5] - final_accuracy[4])
ax.bar(models_3, acc_increase, color=['peru', 'steelblue', 'rebeccapurple'])
for a, b in zip(models_3, acc_increase):
	plt.text(a, b + 0.007, '%.2f' % b + '%', ha='center', va='bottom', fontsize=10)
ax.set_title('Top-1 Accuarcy Increase by using acb', fontsize=16)
plt.ylabel('incrasing rate', fontsize=12)
save_fig('Top-1 Accuarcy increase')
fig.show()
