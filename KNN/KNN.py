import numpy as np
#读取数据集的图片部分
def read_image(filename, offset, amount):
    read_num = 0
    image = np.zeros((amount,28*28)) #创建图片数组
    with open(filename, 'rb') as pf:
        pf.seek(16 + 28 * 28 * offset)#定位第offset张图片
        for ind in range(amount):
            for row in range(28*28):
                data = pf.read(1)
                pix = int.from_bytes(data, byteorder='big')
                if pix > 50:#以50灰度区分黑白
                    image[ind][row] = 1
            read_num = read_num+1
            if read_num%1000 == 0:
                print('已读取第', read_num, '张图像')
        print('读取图像结束')
    return image
#读取数据集的标签部分
def read_label(filename, offset, amount):
    read_num = 0
    image = np.zeros(amount) #创建amount个数据的标签数组
    with open(filename, 'rb') as pf:
        pf.seek(8 + offset)#定位到第offset图片的标签
        for ind in range(amount):
                data = pf.read(1)
                pix = int.from_bytes(data, byteorder='big')
                image[ind] = pix
                read_num = read_num + 1
                if read_num % 1000 == 0:
                    print('已读取第', read_num, '张图像的标签')
        print('读取标签结束')
    return image

def classify(test_arr,train_arr, label_arr, K):
    sum_arr = np.sum(np.abs(test_arr - train_arr) ** 2, axis=1) ** (1/2)#axis为1，用于对行求和，即欧式求和
    find_arr = np.argsort(sum_arr)[:K]#排序得出前k个最小的数字的下标
    label = label_arr[find_arr]
    l_sort = np.zeros(10)
    for i in range(9):
        p = label[label == i].size
        l_sort[i] = p
    return np.argsort(l_sort)[9]#列表数字最大的作为识别结果

train_image = './train-images.idx3-ubyte'
train_lable = './train-labels.idx1-ubyte'
test_image = './t10k-images.idx3-ubyte'
test_label = './t10k-labels.idx1-ubyte'
offset, amount =map(int,input('要从第几张图片开始训练，训练图片数量为多少？').split())
train_image_arr = read_image(train_image, offset, amount)
train_label_arr = read_label(train_lable, offset, amount)
offset, amount =map(int,input('要从第几张图片开始测验，测验图片数量为多少？').split())
test_image_arr = read_image(train_image, offset, amount)
test_label_arr = read_label(train_lable, offset, amount)
Wrong_num_arr = np.zeros(10)
WRONG_NUM = 0
for i in range(amount):
    label = classify(test_image_arr[i], train_image_arr, train_label_arr, 3)
    if test_label_arr[i] == label:
        print(i,'TURE')
    else:
        Wrong_num_arr[label] += 1
        print(i, 'WRONG!!!!!')
        WRONG_NUM += 1
print("已成功识别结束%d张图片<<<<"%(amount),'错误率为', (float(WRONG_NUM/amount) * 100), '%', sep='')
print("各数字错误识别次数如下：")
for i in range(10):
    print(i,"   ",Wrong_num_arr[i])
