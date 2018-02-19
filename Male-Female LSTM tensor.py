
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.contrib import rnn
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv('gender_data.csv',header=None)


# In[3]:


data.columns = ['name','m_or_f']
data['namelen'] = [len(str(i)) for i in data['name']]


# In[4]:


data = data[(data['namelen'] >= 2) ]


# In[5]:


names = data['name']
gender = data['m_or_f']
vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)


# In[6]:


char_index = dict((c, i) for i, c in enumerate(vocab))


# In[7]:


def set_flag(i):
    tmp = np.zeros(39);
    tmp[i] = 1
    return(tmp)


# In[8]:


data['name'] = data['name'].apply(lambda x: str(x)[:30])
data['namelen'] = [len(str(i)) for i in data['name']]


# In[9]:


maxlen = 30
def encoded_name(name):
    tmp = [set_flag(char_index[j]) for j in str(name)]
    for k in range(0,maxlen - len(str(name))):
        tmp.append(set_flag(char_index["END"]))
    tmp = np.array(tmp)
    return tmp


# In[10]:


data['encode_name'] = data['name'].apply(encoded_name)


# In[11]:


data['encode_result'] = data['m_or_f'].apply(lambda x: [1,0] if x=='m' else [0,1])


# In[12]:


data.head()


# In[13]:


def next_batch(X_data,Y_data,steps):
    
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0,len(X_data)-steps) 

    x_batch = np.array([X_data[rand_start:rand_start+steps+1]])[0]
    y_batch = np.array([Y_data[rand_start:rand_start+steps+1]])[0]
    return x_batch,y_batch


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(data['encode_name'], data['encode_result'], test_size=0.3, random_state=101)


# In[15]:


# Just one feature, the time series
num_inputs = 39
# Num of steps in each batch
num_time_steps = 30
# 100 neuron layer, play with this
num_neurons = 200
# Just one output, predicted time series
num_outputs = 2

## You can also try increasing iterations, but decreasing learning rate
# learning rate you can play with this
learning_rate = 0.001 
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 10000
# Size of the batch of data
batch_size = 10
display_step = 200


# In[16]:


X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
Y = tf.placeholder(tf.float32, [None, num_outputs])


# In[17]:


# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_neurons, num_outputs]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_outputs]))
}


# In[18]:


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, num_time_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_neurons, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# In[19]:


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# In[46]:


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    for step in range(1, num_train_iterations):
        batch_x, batch_y = next_batch(X_train, y_train, batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " +                   "{:.4f}".format(loss) + ", Training Accuracy= " +                   "{:.3f}".format(acc))

    print("Optimization Finished!")
    
    # Calculate accuracy for 128 mnist test images
    test_len = 10
    test_data,test_label = next_batch(X_test,y_test,test_len)
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    
    print("prediction: ", sess.run(prediction, feed_dict={X: np.array([encoded_name('kamal')]), Y: np.array([np.array([1,0])])}))


# In[44]:





# In[39]:




