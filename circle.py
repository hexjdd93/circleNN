import numpy as np
import tensorflow as tf
import random
import matplotlib
import matplotlib.pyplot as plt

# x,y
circle = np.array([[0.5],[0.5]])
#radio
r = 3
#data to train
x_data = np.array([])
y_data = np.array([])

# data count
amountData = 5000

# Variables for training
learning_rate = 0.5
epochs = 100
batch_size = 100 # This time, it will not use

'''
    This Method gets a point (x,y) and returns 1 if the point is inside o 0 if it is outside the circle
    The circle is above this method
'''
def inside(point):
    if (point.shape != (2,1)):
        return -1
    d = np.sqrt( np.power(circle[0] - point[0], 2) + np.power(circle[1] - point[1], 2) )
    return 1 if d <= r else 0


if __name__ == "__main__":
    # print( inside(np.array( [[-1],[-1]] )) )
    '''
    This for cicle is used to get the data training
    '''
    for i in range(amountData):
        # varible to get a array with values from -5 to 5
        point = np.random.uniform(-5,5,(2,1))
        # Verify if the point is inside or outside, then append to y_data
        y_data = np.append( y_data, inside(point))
        '''
        We want to lear if a point (x,y) is inside or outside of a circle
        so the weights will be [ [x], [y], [x^2], [y^2] ], this is because of
        circle equation (x-h)^2 + (y-k)^2 = r^2
        '''
        toWeight = np.array( [[point[0][0]],[point[1][0]],[point[0][0]*point[0][0]],[point[1][0]*point[1][0]]] )
        # append toWeight to x_data
        x_data = np.append(x_data, toWeight)
        
    # y_data to reshape to (amountData,1)
    y_data = np.reshape(y_data, (amountData, 1))
    # x_data will have a shape (amountData, 4)
    x_data = np.reshape(x_data, (amountData, 4) )

    print("--------- init tensorflow ---------")
    '''
    The equation will be y = W*X+b
    '''
    # X as input an Y as output
    x = tf.placeholder(tf.float32, [None, 4])
    y = tf.placeholder(tf.float32, [None, 1])

    # # weights of the NN and b
    W = tf.Variable(tf.random_normal([4,1], stddev=0.03), name='W')
    b = tf.Variable(tf.random_normal([1]), name='b')

    # # output
    y_ = tf.add(tf.matmul(x, W), b)
    # The sigmoid function works better for this problem than relu
    # y_ = tf.nn.relu(y_)
    y_ = tf.sigmoid(y_)
    # the output is clipped to avoid get a NaN
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    
    # #Cost function
    costF = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
    # # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(costF)

    #init
    init_op = tf.global_variables_initializer()

    #Variable for get the weights initialize with zeros
    weights = np.zeros((5,1))
    bias = np.zeros((1,1))

    with tf.Session() as sess:
        #DON'T FORGET TO INITIALIZE
        sess.run(init_op)
        print(init_op)
        avgCost = 0
        for epoch in range(epochs):
            _, c = sess.run([optimiser, costF], feed_dict={x: x_data, y: y_data})
            # #If want to see the progress with the cost uncomment the follow lines
            # if(epoch % 100 == 0):
            #     print(c)
            avgCost += c
        # the average cost is:
        print(avgCost/epochs)
        weights = sess.run(W)
        bias = sess.run(b)[0]
        print(weights, bias)

    print("--------- init matplotlib ---------")
    # Variables used for plot the training data
    inside_x = np.array([])
    inside_y = np.array([])
    outside_x = np.array([])
    outside_y = np.array([])
    
    for i in range(amountData):
        # if y_data[i][0] == 1 the point is inside
        if(int(y_data[i][0]) == 1):
            inside_x = np.append(inside_x, np.array([ [x_data[i][0]] ]))
            inside_y = np.append(inside_y, np.array([ [x_data[i][1]] ]))
        else:
            outside_x = np.append(outside_x, np.array([ [x_data[i][0]] ]))
            outside_y = np.append(outside_y, np.array([ [x_data[i][1]] ]))
    # reshape the inside and outside variables
    inside_x = np.reshape(inside_x, (inside_x.shape[0], 1 ))
    inside_y = np.reshape(inside_y, (inside_y.shape[0], 1 ))
    outside_x = np.reshape(outside_x, (outside_x.shape[0], 1 ))
    outside_y = np.reshape(outside_y, (outside_y.shape[0], 1 ))
    
    # subplots
    fig, ax = plt.subplots()
    #plot all the inside points
    ax.scatter(x=inside_x, y=inside_y,c='red')
    #plot all the outside points
    ax.scatter(x=outside_x, y=outside_y,c='green')

    #now plot the result of NN
    nn_x = np.linspace(-9, 9, 400)
    nn_y = np.linspace(-5, 5, 400)
    nn_x, nn_y = np.meshgrid(nn_x, nn_y)
    a, b, c, d = weights
    ax.contour(nn_x, nn_y,(a[0]*nn_x + b[0]*nn_y + c[0]*nn_x**2 + d[0]*nn_y**2 + bias), [0], colors='blue')

    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()