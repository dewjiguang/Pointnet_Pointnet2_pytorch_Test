from matplotlib import pyplot as plt

Loss_list = []  #存储每次epoch损失值

def draw_fig(list,name,epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch+1)
    print(x1)
    y1 = list
    if name=="loss":
        plt.cla()
        plt.title('Train loss vs. epoch', fontsize=20)
        yresult = []
        for i in y1:
            yresult.append(i.cpu().detach().numpy())
        plt.plot(x1, yresult, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.grid()
        plt.savefig("data/pic/Train_loss.png")
        plt.show()
    elif name =="acc":
        plt.cla()
        plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.grid()
        plt.savefig("data/pic/Train _accuracy.png")
        plt.show()
    f = open('data/draw_' + name + '.txt', mode='w')
    data = list
    for i in data:
        f.write(str(i) + '\n')
    f.close()
