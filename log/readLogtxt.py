from matplotlib import pyplot as plt

from data_utils.Draw import draw_fig
def draw_fig1(list,name,epoch,picdir,str1):
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
        plt.savefig(picdir + "/Train_loss.png")
        plt.show()
    elif name =="acc":
        plt.cla()
        plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.grid()
        plt.savefig(picdir + "/"+str1+".png")
        plt.show()

def draw_fig2(list1,list2,name,epoch,picdir,str1,str2):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch+1)
    print(x1)
    y1 = list1
    y2 = list2

    if name =="acc":
        plt.cla()
        # plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '^-')
        plt.plot(x1, y2, 's-')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
        plt.rcParams['axes.unicode_minus'] = False
        # plt.legend(['str1','str2'],loc='upper left')
        plt.legend(['baseline','ours'],loc='lower right')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.grid()
        plt.savefig(picdir + "/"+str1+"_"+str2+"ACC.png")
        plt.show()

def draw_fig3(list1,list2,list3,name,epoch,picdir,str1,str2,str3):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch+1)
    print(x1)
    y1 = list1
    y2 = list2
    y3 = list3
    if name =="acc":
        plt.cla()
        # plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '^-')
        plt.plot(x1, y2, 's-')
        plt.plot(x1, y3, '.-')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
        plt.rcParams['axes.unicode_minus'] = False
        # plt.legend(['str1','str2'],loc='upper left')
        plt.legend(['空间注意力','CBAM','baseline'],loc='lower right')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.grid()
        plt.savefig(picdir + "/"+str1+"_"+str2+"ACC.png")
        plt.show()
def openreadtxt(file_name):
    data = []
    file = open(file_name, 'r')  # 打开文件
    file_data = file.read() # 读取所有行
    tmp_list = file_data.split('Model - INFO - Training accuracy: ')
    # 验证
    # tmp_list = file_data.split('Model - INFO - eval point accuracy: ')
    del tmp_list[0]
    str=[]
    for i in range(len(tmp_list)):
        str.append(float(tmp_list[i][0:8]))
    return str


if __name__ == "__main__":

    str1='2023-03-11_21-02_空间注意力30次'
    str2='2023-03-11_15-19_CBAM30次'
    str3='2023-03-12_13-55_CBAM池化前三十次'
    data1 = openreadtxt('sem_seg/'+str1+'/logs/pointnet_sem_seg.txt')
    data2 = openreadtxt('sem_seg/'+str2+'/logs/pointnet_sem_seg.txt')
    data3 = openreadtxt('sem_seg/'+str3+'/logs/pointnet_sem_seg.txt')
    print(data1)
    sum=0;
    for i in range(len(data1)):
        sum+=data1[i]
    print(sum/30)
    sum = 0;
    for i in range(len(data2)):
        sum += data2[i]
    print(sum/30)

    sum = 0;
    for i in range(len(data3)):
        sum += data3[i]
    print(sum/30)
    # draw_fig1(data1,'acc', 30, 'pic/',str1)
#  俩图的
#     draw_fig2(data1,data2, 'acc', 30, 'pic/', str1,str2)
    draw_fig3(data1,data2,data3,'acc', 30, 'pic/', str1,str2,str3)





