import os
import sys
import json
import matplotlib.pyplot as plt

base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(base_project_dir)

history_dir = os.path.join(base_project_dir, 'Histories')

Org_VGG_BiLSTM_history_path = os.path.join(history_dir, 'Org_VGG_BiLSTM.json')
Org_VGG_BiLSTM_Pretrained_history_path = os.path.join(history_dir, 'ShifAug_VGG_BiLSTM.json')
Org_discriminator_Pretrained_history_path = os.path.join(history_dir, 'ShifAug_discriminator_PreTrained.json')

with open(Org_VGG_BiLSTM_history_path, 'r') as file:
    Org_VGG_BiLSTM_history = json.load(file)

with open(Org_VGG_BiLSTM_Pretrained_history_path, 'r') as file:
    Org_VGG_BiLSTM_Pretrained_history = json.load(file)

with open(Org_discriminator_Pretrained_history_path, 'r') as file:
    Org_discriminator_Pretrained_history = json.load(file)

Org_VGG_BiLSTM_loss = Org_VGG_BiLSTM_history['loss']
Org_VGG_BiLSTM_f1 = Org_VGG_BiLSTM_history['f1']
Org_VGG_BiLSTM_val_loss = Org_VGG_BiLSTM_history['val_loss']
Org_VGG_BiLSTM_val_f1 = Org_VGG_BiLSTM_history['val_f1']

Org_VGG_BiLSTM_Pretrained_loss = Org_VGG_BiLSTM_Pretrained_history['loss']
Org_VGG_BiLSTM_Pretrained_f1 = Org_VGG_BiLSTM_Pretrained_history['f1']
Org_VGG_BiLSTM_Pretrained_val_loss = Org_VGG_BiLSTM_Pretrained_history['val_loss']
Org_VGG_BiLSTM_Pretrained_val_f1 = Org_VGG_BiLSTM_Pretrained_history['val_f1']

Org_discriminator_Pretrained_loss = Org_discriminator_Pretrained_history['loss']
Org_discriminator_Pretrained_f1 = Org_discriminator_Pretrained_history['f1']
Org_discriminator_Pretrained_val_loss = Org_discriminator_Pretrained_history['val_loss']
Org_discriminator_Pretrained_val_f1 = Org_discriminator_Pretrained_history['val_f1']


plt.figure()
plt.plot(Org_VGG_BiLSTM_loss)
plt.plot(Org_VGG_BiLSTM_Pretrained_loss)
#plt.plot(Org_discriminator_Pretrained_loss)
plt.title('Simple vs Pre-trained losses')
plt.legend(['loss of simple VGG-BiLSTM', 'loss of pre-trained VGG-BLSTM', 'loss of pre-trained discriminator'])
plt.show()

plt.figure()
plt.plot(Org_VGG_BiLSTM_f1)
plt.plot(Org_VGG_BiLSTM_Pretrained_f1)
plt.plot(Org_discriminator_Pretrained_f1)
plt.title('Simple vs Pre-trained f1s')
plt.legend(['f1 of simple VGG-BiLSTM', 'f1 of pre-trained VGG-BLSTM', 'f1 of pre-trained discriminator'])
plt.show()

plt.figure()
plt.plot(Org_VGG_BiLSTM_val_loss)
plt.plot(Org_VGG_BiLSTM_Pretrained_val_loss)
plt.plot(Org_discriminator_Pretrained_val_loss)
plt.title('Simple Training vs  Pre-trained val losses')
plt.legend(['val loss of simple VGG-BiLSTM', 'val loss of pre-trained VGG-BLSTM', 'val loss of pre-trained discriminator'])
plt.show()

plt.figure()
plt.plot(Org_VGG_BiLSTM_val_f1)
plt.plot(Org_VGG_BiLSTM_Pretrained_val_f1)
plt.plot(Org_discriminator_Pretrained_val_f1)
plt.title('Simple vs Pre-trained val f1s')
plt.legend(['val f1 of simple VGG-BiLSTM', 'val f1 of pre-trained VGG-BLSTM', 'val f1 of pre-trained discriminator'])
plt.show()




