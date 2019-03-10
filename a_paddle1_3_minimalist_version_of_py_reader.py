#coding=utf-8
#paddle 1.3
import paddle
import paddle.fluid as fluid
import numpy as np

#数据生成器
def train_data_generator():
    for i in range(80):
        yield [np.array([i%10]),np.array([(i%10)*2])]
def test_data_generator():
    for i in range(10):
        yield [np.array([i%10]),np.array([(i%10)*2])]

def build_net(phase,data_generator,main_program,startup_program):
    #建立train的异步reader，和建立test的异步reader
    with fluid.program_guard(main_program,startup_program):
        image_input = fluid.layers.data(name='image_input',shape=[-1,1],dtype='float32')
        label_input = fluid.layers.data(name='label_input',shape=[-1,1],dtype='float32')
        if(phase=='train'):
            reader = fluid.layers.create_py_reader_by_data(capacity=64,feed_list=[image_input,label_input])
            reader.decorate_paddle_reader(paddle.batch(data_generator,4))
        else:
            reader = fluid.layers.create_py_reader_by_data(capacity=64,feed_list=[image_input,label_input])
            reader.decorate_paddle_reader(paddle.batch(data_generator,1))
        #在unique_name保护下建立train和test统一的网络，具体含义参考官网api
        with fluid.unique_name.guard():
            img,label = fluid.layers.read_file(reader)
            hidden = fluid.layers.fc(input = img, size=10, act='relu')
            predict = fluid.layers.fc(input = hidden, size=1, act=None)
            loss = fluid.layers.square_error_cost(input = predict,label=label)
            average_loss = fluid.layers.mean(loss)
            #期望的网络功能：将输入数字乘以2再输出
            optimizer = fluid.optimizer.SGDOptimizer(0.001)
            opts = optimizer.minimize(average_loss)
            if(phase=='train'):
                return reader,average_loss,img,predict
            elif(phase=='test'):
                return reader,img,predict

def main():
    #定义program
    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = fluid.Program()

    #定义设备和exe
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    #定义train网络及其reader、返回值；定义test网络及其reader、返回值
    train_reader,train_average_loss, train_img, train_predict = build_net('train',train_data_generator,train_program,startup_program)
    test_reader, test_img, test_predict = build_net('test',test_data_generator,test_program,startup_program)

    #全局启动（具体含义参考官网api介绍）
    exe.run(startup_program)

    #train的reader的启动，开始训练
    for epoch_id in range(3):
        train_reader.start()  
        try:
            while(True):
                fetched_train_average_loss,fetched_train_img, fetched_train_predict = exe.run(program = train_program, fetch_list=[train_average_loss,train_img,train_predict])
                print('fetched_train_average_loss\n',fetched_train_average_loss)
                print('fetched_train_img\n',fetched_train_img)
                print('fetched_train_predict\n',fetched_train_predict)
                print('')
        except fluid.core.EOFException:
            print("end of train===============")
            #train的reader的重置，使得下一个batch可以继续训练
            train_reader.reset()

        #模型保存
        model_param_path="./model_parameter"
        train_param_path="./train_parameter"
        fluid.io.save_params(executor=exe,dirname=model_param_path,main_program=train_program)
        fluid.io.save_persistables(executor=exe,dirname=train_param_path,main_program=train_program)

        test_reader.start()
        try:
            while(True):
                fetched_test_img,fetched_test_predict = exe.run(program = test_program,fetch_list=[test_img,test_predict])
                print('fetched_test_img:\n',fetched_test_img)
                print('fetched_test_predict:\n',fetched_test_predict)
                print('')
        except fluid.core.EOFException:
            print("end of test===============")
            #reset貌似只有清空缓存（即capacity）的功能，生成器无法重置，只能用完再重开始
            test_reader.reset()

if __name__=="__main__":
    main()
    #其余的学习率调整，加载参数恢复训练等，均可在此代码基础上修改完成