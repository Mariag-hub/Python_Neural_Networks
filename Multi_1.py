import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt
from Progetto import load
from datetime import datetime


#mini batch [10-150], eta [0.00001-1], nh [15-90]

X_train, y_train, X_test, y_test = load()

class MLP(object):
    
    def __init__(self, n_hidden, eta, mini_batch_size):
        self.n_hidden = n_hidden
        self.eta = eta
        self.minibatch_size = mini_batch_size
        self.eval_ = {'loss': []}
        
    def create_batch_generator(self,X, y, batch_size=128, shuffle=False):
        X_copy = np.array(X)
        y_copy = np.array(y)
    
        if shuffle:
            data = np.column_stack((X_copy, y_copy))
            np.random.shuffle(data)
            X_copy = data[:, :-1]
            y_copy = data[:, -1].astype(int)
    
        for i in range(0, X.shape[0], batch_size):
            yield (X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])
            
    def fit(self, X_train, y_train, X_test, y_test, epochs):
           
            n_features = X_train.shape[1]
            n_classes = 10
            random_seed = 123
            np.random.seed(random_seed)
    
            g1 = tf.Graph()
            with g1.as_default():
                tf.set_random_seed(random_seed)
                tf_x = tf.placeholder(dtype=tf.float32,
                           shape=(None, n_features),
                           name='tf_x')
    
                tf_y = tf.placeholder(dtype=tf.int32, 
                            shape=None, name='tf_y')
                y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)
    
                h1 = tf.layers.dense(inputs=tf_x, units=self.n_hidden,
                             activation=tf.sigmoid,
                             name='layer1')
    
                logits = tf.layers.dense(inputs=h1, 
                                 units=10,
                                 activation=None,
                                 name='layer3')
    
                predictions = {
                        'classes' : tf.argmax(logits, axis=1, 
                                  name='predicted_classes'),
                        'probabilities' : tf.nn.softmax(logits, 
                                  name='softmax_tensor')
                        }
            with g1.as_default():
                cost = tf.losses.mean_squared_error(labels=y_onehot, predictions = logits)
    
                optimizer = tf.train.GradientDescentOptimizer(
                        learning_rate=self.eta)
    
                train_op = optimizer.minimize(loss=cost)
    
                init_op = tf.global_variables_initializer()
                sess =  tf.Session(graph=g1)
                sess.run(init_op)
                
                t1 = datetime.now()
                training_costs = []
                for epoch in range(epochs):
                    batch_generator = self.create_batch_generator(
                            X_train, y_train, 
                            batch_size=self.minibatch_size, shuffle = False)
                    for batch_X, batch_y in batch_generator:
            
                        feed = {tf_x:batch_X, tf_y:batch_y}
                        _, batch_cost = sess.run([train_op, cost],
                                     feed_dict=feed)
                        training_costs.append(batch_cost)
                    print(' -- Epoch %2d  '
                             'Avg. Training Loss: %.4f' % (
                                     epoch+1, np.mean(training_costs)
                            ))
                    self.eval_['loss'].append(np.mean(training_costs))
    
                feed = {tf_x : X_test}
                y_pred = sess.run(predictions['classes'], 
                                  feed_dict=feed)
     
                acc = np.sum(y_pred == y_test)/y_test.shape[0]
                t2 = datetime.now()
                print(t2-t1)
                sess.close()
                return acc
    
    def plot(self):
        
        plt.plot(range(len(self.eval_['loss'])), self.eval_['loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.show()
        
