#coding=utf-8
#paddle 1.3
import paddle
import paddle.fluid as fluid
import numpy as np
import cv2
np.set_printoptions(suppress=True)
#数据生成器
def train_data_generator():
    img0=cv2.imread('./img0.jpg').astype(np.float32)
    img1=cv2.imread('./img1.jpg').astype(np.float32)
    img0=img0.transpose(2,0,1)
    img1=img1.transpose(2,0,1)
    img_shape=img0.shape
    img0=img0/256-0.5
    img1=img1/256-0.5
    for i in range(200):
        img0 = img0+np.random.random(img_shape)/255
        img1 = img1+np.random.random(img_shape)/255
        if(np.random.random((1))>0.5):
            yield [np.array(img0),np.array([0]).astype(np.int64)]
        else:
            yield [np.array(img1),np.array([1]).astype(np.int64)]

def test_data_generator():
    img0=cv2.imread('./img0.jpg').astype(np.float32)
    img1=cv2.imread('./img1.jpg').astype(np.float32)
    img0=img0.transpose(2,0,1)
    img1=img1.transpose(2,0,1)
    img_shape=img0.shape
    img0=img0/256-0.5
    img1=img1/256-0.5
    for i in range(8):
        img0 = img0+np.random.random(img_shape)/255
        img1 = img1+np.random.random(img_shape)/255
        if(np.random.random((1))>0.5):
            yield [np.array(img0),np.array([0]).astype(np.int64)]
        else:
            yield [np.array(img1),np.array([1]).astype(np.int64)]

def build_net(phase,data_generator,main_program,startup_program):
    #建立train的异步reader，和建立test的异步reader
    with fluid.program_guard(main_program,startup_program):
        image_input = fluid.layers.data(name='image_input',shape=[-1,3,112,112],dtype='float32')
        label_input = fluid.layers.data(name='label_input',shape=[-1,1],dtype='int64')
        if(phase=='train'):
            reader = fluid.layers.create_py_reader_by_data(capacity=64,feed_list=[image_input,label_input])
            reader.decorate_paddle_reader(paddle.batch(data_generator,4))
        else:
            reader = fluid.layers.create_py_reader_by_data(capacity=64,feed_list=[image_input,label_input])
            reader.decorate_paddle_reader(paddle.batch(data_generator,2))
        #在unique_name保护下建立train和test统一的网络，具体含义参考官网api
        with fluid.unique_name.guard():
            img,label = fluid.layers.read_file(reader)

            output1 = fluid.layers.conv2d(input=img, num_filters=16, filter_size=3, stride=2,padding=1, act='relu')
            output2 = fluid.layers.pool2d(input=output1, pool_size=2, pool_type='max',pool_stride=2)
            output3 = fluid.layers.conv2d(input=output2,num_filters=24,filter_size=3,stride=1,padding=1,act='relu')
            output4 = fluid.layers.pool2d(input=output3, pool_size=2,pool_type='max',pool_stride=2)
            output5 = fluid.layers.conv2d(input=output4, num_filters=24,filter_size=3,stride=1,padding=1,act='relu')
            output6 = fluid.layers.conv2d(input=output5, num_filters=24,filter_size=3,stride=1,padding=1,act='relu')
            output7 = fluid.layers.conv2d(input=output6, num_filters=24,filter_size=3,stride=1,padding=1,act='relu')
            output8 = fluid.layers.pool2d(input=output7, pool_size=2,pool_type='max',pool_stride=2)
            output9 = fluid.layers.conv2d(input=output8, num_filters=24,filter_size=3,stride=1,padding=1,act='relu')
            output10= fluid.layers.conv2d(input=output9, num_filters=24,filter_size=3,stride=1,padding=1,act='relu')
            output11= fluid.layers.fc(output10,2)
            softmax_out = fluid.layers.softmax(output11)
            loss = fluid.layers.cross_entropy(softmax_out,label)
            average_loss = fluid.layers.mean(loss)
            accuracy = fluid.layers.accuracy(input=softmax_out, label=label)
            
            optimizer = fluid.optimizer.AdamOptimizer(0.001)
            opts = optimizer.minimize(average_loss)
            if(phase=='train'):
                return reader,average_loss,img,softmax_out,label
            elif(phase=='test'):
                return reader,img,accuracy,label

def main():
    #定义program
    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()

    #定义设备和exe
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    #定义train网络及其reader、返回值；定义test网络及其reader、返回值
    train_reader,train_average_loss, train_img, train_predict,train_label = build_net('train',train_data_generator,train_program,startup_program)
    test_reader, test_img, test_accuracy, test_label = build_net('test',test_data_generator,test_program,startup_program)

    #全局启动（具体含义参考官网api介绍）
    exe.run(startup_program)

    #train的reader的启动，开始训练
    for epoch_id in range(100):
        train_reader.start()  
        try:
            while(True):
                fetched_train_average_loss,fetched_train_img, fetched_train_predict, fetched_train_label = exe.run(program = train_program, fetch_list=[train_average_loss,train_img,train_predict,train_label])
                print('fetched_train_average_loss\n',fetched_train_average_loss)
                print('fetched_train_img\n',fetched_train_img.shape)
                print('fetched_train_predict\n',fetched_train_predict)
                print('fetched_train_label\n',fetched_train_label)
                print('')
        except fluid.core.EOFException:
            print("end of train===============")
            #train的reader的重置，使得下一个batch可以继续训练
            train_reader.reset()

        #模型保存
        model_param_path="./model_parameter/params"+str(epoch_id).zfill(3)
        train_param_path="./train_parameter/persistables"+str(epoch_id).zfill(3)
        fluid.io.save_params(executor=exe,dirname=model_param_path,main_program=train_program)
        fluid.io.save_persistables(executor=exe,dirname=train_param_path,main_program=train_program)

        test_reader.start()
        try:
            while(True):
                fetched_test_img,fetched_test_accuracy,fetched_test_label = exe.run(program = test_program,fetch_list=[test_img,test_accuracy,test_label])
                print('fetched_test_img:\n',fetched_test_img.shape)
                print('fetched_test_accuracy:\n',fetched_test_accuracy)
                print('fetched_test_label:\n',fetched_test_label)
                print('')
        except fluid.core.EOFException:
            print("end of test===============")
            #reset貌似只有清空缓存（即capacity）的功能，生成器无法重置，只能用完再重开始
            test_reader.reset()

if __name__=="__main__":
    main()
    #其余的学习率调整，加载参数恢复训练等，均可在此代码基础上修改完成