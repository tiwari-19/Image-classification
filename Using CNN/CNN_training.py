from CNN_model import *
import _pickle
from sklearn.model_selection import train_test_split

data = _pickle.load(open(os.path.join(CHECKPOINT_PATH, 'dataset_feats.pkl'), 'rb'))
Labels = _pickle.load(open(os.path.join(CHECKPOINT_PATH, 'labels.pkl'), 'rb'))

X_train, X_val, y_train, y_val = train_test_split(data, Labels, shuffle=True, test_size=0.10, stratify=Labels)
print("Training samples =", X_train.shape[0])
print("Validation samples =", X_val.shape[0])


sess = tf.InteractiveSession()
initializer = tf.global_variables_initializer()
sess.run(initializer)

batch_size = 16
num_batches = X_train.shape[0] // batch_size

for epoch_counter in range(30):
    train_loss = 0
    start = 0
    end = start + batch_size
    for batch_counter in range(num_batches):
        batch_x = X_train[start:end]
        batch_y = y_train[start:end]
        start = end
        end = start + batch_size
        train_dict = {X: batch_x, y_true: batch_y}
        _, batch_loss = sess.run([train_step, loss], feed_dict=train_dict)
        train_loss += batch_loss
    train_loss /= num_batches
    val_dict = {X: X_val, y_true: y_val}
    val_loss, val_acc = sess.run([loss, accuracy], feed_dict=val_dict)

    print("Epoch %2d:  Train loss = %0.3f   Val loss = %0.3f   Val Acc = %0.3f" % (epoch_counter+1,
           train_loss,
           val_loss,
           val_acc))

saver = tf.train.Saver()
saver.save(sess, 'checkpoint_dir/trained_cnn')