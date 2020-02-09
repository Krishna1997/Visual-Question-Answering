import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from six.moves import cPickle as pickle
from datetime import datetime
from torchvision import models


class Trainer(object):

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=16, lr=0.001):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 100
        self._test_freq = 10
        self._save_freq = 3
        self._print_freq = 50
        self._batch_size = batch_size
        self._lr = lr

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.DEVICE)
        if self.DEVICE == "cuda":
            self._model = self._model.cuda()

        if self.method == 'simple':
            self.optimizer = optim.Adam([{'params': self._model.embed.parameters(), 'lr': 0.08},
                                        {'params': self._model.gnet.parameters(), 'lr': 1e-3},
                                        {'params': self._model.fc.parameters(), 'lr': 1e-3}
                                       ], weight_decay=1e-8)
        else:
            self.optimizer = optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=1e-8)
        
        self.criterion = nn.CrossEntropyLoss()
        self.initialize_weights()


        self.total_validation_questions = 200.0

        if self.method == 'simple':
            self.chk_dir = './chk_simple/'
        else:
            self.chk_dir = './chk_coattention_test/'
            print('Creating Image Encoder')
            self.img_enc = models.resnet18(pretrained=True)
            modules = list(self.img_enc.children())[:-2]
            self.img_enc = nn.Sequential(*modules)
            for params in self.img_enc.parameters():
                params.requires_grad = False
            if self.DEVICE == "cuda":
                self.img_enc = self.img_enc.cuda()
            self.img_enc.eval()

        if not os.path.exists(self.chk_dir):
            os.makedirs(self.chk_dir)

    def _optimize(self, predicted_answers, true_answers):
        raise NotImplementedError()

    def validate(self):
        accuracy = 0.0
        batches = 0
        with open('./data/a2i.pkl', 'rb') as f:
            a2i = pickle.load(f)
        with open('./data/i2a.pkl', 'rb') as f:
            i2a = pickle.load(f)
        wc = open("correct_predictions_simple.txt",'w')
        ww = open("wrong_predictions_simple.txt",'w')
        for batch_id, (imgT, quesT, gT, img_name, ques, answer) in enumerate(self._val_dataset_loader):
            batches+=1
            self._model.eval()
            if not self.method == 'simple':
                quesT = rnn.pack_sequence(quesT)
                imgT = imgT.to(self.DEVICE)
                imgT = self.img_enc(imgT)
                imgT = imgT.view(imgT.size(0), imgT.size(1), -1)

            imgT, quesT, gT = imgT.to(self.DEVICE), quesT.to(self.DEVICE), gT.to(self.DEVICE)
            gT = torch.squeeze(gT)
            pd_ans = self._model(imgT, quesT) # TODO
            #print(len(i2a))
            #print("Pd len: ",pd_ans.shape[1])
            for i in range(gT.shape[0]):
                #print("Pd len: ",pd_ans[i].shape[0])
                #print(img_name[i],ques[i],i2a[torch.argmax(pd_ans[i]).item()],answer[i])
                if torch.argmax(pd_ans[i]).item() == gT[i]:
                    accuracy = accuracy + 1.0
                    #print(ques[i],answer[i],i2a[torch.argmax(pd_ans[i]).item()],a2i[answer[i]],gT[i].item())
                    wc.write("{:}\t{:}\t{:}\t{:}\n".format(img_name[i],ques[i],i2a[torch.argmax(pd_ans[i]).item()],answer[i]))
                else:
                    #print(ques[i],i2a[torch.argmax(pd_ans[i]).item()],a2i[answer[i]],gT[i].item())
                    ww.write("{:}\t{:}\t{:}\t{:}\n".format(img_name[i],ques[i],i2a[torch.argmax(pd_ans[i]).item()],answer[i]))
            if (batch_id + 1) % self._print_freq == 0:
                print('Validation Accuracy: %f' % (accuracy / ((batch_id + 1)*self._batch_size)))
        wc.write("\n")
        ww.write("\n")
        wc.close()
        ww.close()
        accuracy = accuracy / (batches*self._batch_size)
        return accuracy

    def train(self):
        print('Started Training.\n')
        tr_iter = 0
        val_iter = 0
        best_prec = 0.0
        for epoch in range(self._num_epochs):
            print("Epoch: ", epoch)
            #if (epoch + 1) // 3 == 0:
                #self.adjust_learning_rate(epoch + 1)
            num_batches = len(self._train_dataset_loader)
            print(num_batches)
            for batch_id, (imgT, quesT, gT, img_name, ques, answer) in enumerate(self._train_dataset_loader):
                self._model.train()
                current_step = epoch * num_batches + batch_id

                # ============
                if not self.method == 'simple':
                    quesT = rnn.pack_sequence(quesT)
                    imgT = imgT.to(self.DEVICE)
                    imgT = self.img_enc(imgT)
                    imgT = imgT.view(imgT.size(0), imgT.size(1), -1)
                else:
                    imgT = imgT.to(self.DEVICE)

                quesT, gT = quesT.to(self.DEVICE), gT.to(self.DEVICE)
                predicted_answer = self._model(imgT, quesT) # TODO
                ground_truth_answer = torch.squeeze(gT)     # TODO
                # ============

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if (current_step + 1) % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    tr_iter = tr_iter + 1
                #if (current_step + 1) % self._test_freq == 0:
                    #self._model.eval()
                    #val_accuracy = self.validate()
                    #print("Epoch: {} has val 1 accuracy {}".format(epoch, val_accuracy))

            if (epoch + 1) % self._save_freq == 0 or epoch == self._num_epochs - 1:
                val_accuracy = self.validate()
                print("Epoch: {} has val 2 accuracy {}".format(epoch, val_accuracy))
                val_iter = val_iter + 1

                is_best = val_accuracy > best_prec
                best_prec = max(val_accuracy, best_prec)
                self.save_checkpoint({'epoch': epoch + 1,
                                      'state_dict': self._model.state_dict(),
                                      'best_prec': best_prec},
                                      #'optimizer': optimizer.state_dict()}, is_best,
                                      is_best, self.chk_dir + 'checkpoint_' + str(epoch + 1) + '.pth.tar')

    def initialize_weights(self):
      for layer in self._model.modules():
          if not isinstance(layer, (nn.Conv2d, nn.Linear)):
              continue
          try:
              torch.nn.init.xavier_normal_(layer.weight)
              try:
                  nn.init.constant_(layer.bias.data, 0)
              except:
                  pass
          except:
              pass

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10
