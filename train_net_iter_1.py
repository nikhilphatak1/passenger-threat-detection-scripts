
# TODO
# FIGURE OUT HOW TO IMPORT/FORMAT/CLEAN DATA

from google.cloud import storage
client = storage.Client()
# https://console.cloud.google.com/storage/browser/[bucket-id]/
bucket = client.get_bucket('nikhil-kaggle-data')
# Then do other things...
blob = bucket.get_blob('remote/path/to/file.txt')

blob2 = bucket.blob('remote/path/storage.txt')
blob2.upload_from_filename(filename='/local/path.txt')


# very important
import tensorflow as tf

# parameters TODO
learning_rate = 0.1
num_steps = 500
batch_size = 1000
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons

 #TODO this is not the true input size, will be smaller
num_input = 512*512*660

# TODO is this actually 2 and we have
# a separate neural net for every threat zone?
num_classes = 2

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


# store layer weights and biases (seems arbitrary but is necessary)
# bias units in every layer are what actually result in complex
# hypotheses, otherwise you just get tensor products everywhere
weights =
{
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases =
{
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# create model
def neural_net(x):
    # hidden layer: fully connected, 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # hidden layer: fully connected, 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# construct model
logits = neural_net(X)

# define loss and optimizer to prepare for training
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initialize the variables
init = tf.global_variables_initializer()

# start training
# fancy python aliasing syntax
with tf.Session() as sess:

    # run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = #TODO get next data batch here
        # run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy],
                                  feed_dict={X: batch_x, Y: batch_y})
            #TODO print step number, loss value
            # and this iteration's training accuracy

    print('''optimization complete message''')

    # Calculate accuracy TODO
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: '''TEST IMAGES GO HERE''',
                                      Y: '''TEST LABELS GO HERE'''}



#TODO
'''''''''''''''''''''''''''''''''
-ask why matlab symmetry checking does not work as expected
-clean data
-format data
-figure out how to stream from google bucket
(use scanner python equivalent)
-learn tf.train documentation
'''''''''''''''''''''''''''''''''
