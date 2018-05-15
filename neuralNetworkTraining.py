import tensorflow as tf
import numpy as np
import random

import trainingMap

# Environment Parameters
n_epochs = 5000
n = 0
n_actions = 3
average = []
step = 1
batch_size = 5000
sensor_number = 1

# Define our three actions of moving forward, turning left & turning right
choice = ['forward', 'left', 'right']

# Hyper Parameters
alpha = 1e-4
gamma = 0.99
normalize_r = True
save_path='models/test.ckpt'
value_scale = 0.5
entropy_scale = 0.00
gradient_clip = 40

# Apply discount to episode rewards & normalize
def discount(r, gamma, normal):
    discount = np.zeros_like(r)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * gamma + r[i]
        discount[i] = G
    # Normalize 
    if normal:
        mean = np.mean(discount)
        std = np.std(discount)
        if (std == 0):
            std = 0.001
        discount = (discount - mean) / (std)
    return discount

# Layers
fc = 256
activ = tf.nn.elu

# Tensorflow Variables
X = tf.placeholder(tf.float32, (None, sensor_number), name='X')
Y = tf.placeholder(tf.int32, (None,), name='actions')
R = tf.placeholder(tf.float32, (None,), name='reward')
N = tf.placeholder(tf.float32, (None), name='episodes')
D_R = tf.placeholder(tf.float32, (None,), name='discounted_reward')

dense = tf.layers.dense(
        inputs = X, 
        units = fc, 
        activation = activ,
        name = 'fc')

logits = tf.layers.dense(
         inputs = dense, 
         units = n_actions, 
         name='logits')

value = tf.layers.dense(
        inputs=dense, 
        units = 1, 
        name='value')

calc_action = tf.multinomial(logits, 1)
aprob = tf.nn.softmax(logits)

tf.trainable_variables()

def rollout(batch_size):
    
    states, actions, rewards, rewardsFeed, discountedRewards = [], [], [], [], []
    episode_num = 1
    action_repeat = 1
    reward = 0

    trainingMap.generateMap()
    currentPos = trainingMap.randomInitalPos()
    direction = trainingMap.randomInitialDir()
    state = trainingMap.getState(currentPos, direction)
    
    while True: 
        # Run State Through Policy & Calculate Action
        feed = {X: state}
        prob = sess.run(aprob, feed_dict=feed)
        action = np.random.choice([0,1,2], p=prob[0])
        
        # Perform Action
        for i in range(action_repeat):
            state2, reward2, done, currentPos, direction = trainingMap.nextStep(choice[action], currentPos, direction)
            reward += reward2
            if done:
                break
        
        # Store Results
        states.append(state[0])
        rewards.append(reward)
        actions.append(action)
        
        # Update Current State
        reward = 0
        state = state2.reshape(1, sensor_number)
        
        if done:
            # Track Discounted Rewards
            rewardsFeed.append(rewards)
            discountedRewards.append(discount(rewards, gamma, normalize_r))
            
            if len(np.concatenate(rewardsFeed)) > batch_size:
                break
                
            # Reset Environment
            currentPos = trainingMap.randomInitalPos()
            direction = trainingMap.randomInitialDir()
            state = trainingMap.getState(currentPos, direction)
            rewards = []
            episode_num += 1
                         
    return np.stack(states), np.stack(actions), np.concatenate(rewardsFeed), np.concatenate(discountedRewards), episode_num

mean_reward = tf.divide(tf.reduce_sum(R), N)

# Define Losses
pg_loss = tf.reduce_mean((D_R - value) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
value_loss = value_scale * tf.reduce_mean(tf.square(D_R - value))
entropy_loss = -entropy_scale * tf.reduce_sum(aprob * tf.exp(aprob))
loss = pg_loss + value_loss - entropy_loss

# Create Optimizer
optimizer = tf.train.AdamOptimizer(alpha)
grads = tf.gradients(loss, tf.trainable_variables())
grads, _ = tf.clip_by_global_norm(grads, gradient_clip) # gradient clipping
grads_and_vars = list(zip(grads, tf.trainable_variables()))
train_op = optimizer.apply_gradients(grads_and_vars)

# Initialize Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(tf.global_variables())
load_was_success = True 

try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
except:
    print("No saved model to load. Starting new session")
    load_was_success = False
else:
    print("Loaded Model: {}".format(load_path))
    saver = tf.train.Saver(tf.global_variables())
    step = int(load_path.split('-')[-1])+1

while step < n_epochs+1:
    # Gather Training Data
    print('Epoch', step)
    s, a, r, d_r, n = rollout(batch_size)
    mean_reward = np.sum(r)/n
    average.append(mean_reward)
    print('Training Episodes: {}  Average Reward: {:4.2f}  Total Average: {:4.2f}'.format(n, mean_reward, np.mean(average)))
          
    # Update Network
    sess.run(train_op, feed_dict={X:s, Y:a, D_R: d_r})
          
    # Save Model
    if step % 10 == 0:
          print("SAVED MODEL")
          saver.save(sess, save_path, global_step=step)
          
    step += 1