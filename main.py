import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))




r_learning = 1e-4
BATCH_SIZE = 16
p_keep = 0.5
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    graph = tf.get_default_graph()
    w1= graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    
    return w1, keep, w3, w4, w7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # one by one convolution:  layer 7 conv 1x1
    rand_std = 1e-2
    l2_scale = 1e-3
    conv1x1_l7 = tf.layers.conv2d(vgg_layer7_out,num_classes, 1, 1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(0,rand_std),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
    # do this according to https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
#    vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001)
#    vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.01)
    conv1x1_l4 = tf.layers.conv2d(vgg_layer4_out,num_classes, 1, 1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(0,rand_std),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
    
    conv1x1_l3 = tf.layers.conv2d(vgg_layer3_out,num_classes, 1, 1, padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(0,rand_std),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
    
    
    # deconvolution: 2Xconv7
    output = tf.layers.conv2d_transpose(conv1x1_l7, num_classes, 4, 2, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(0,rand_std),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
    
    # pool4 + 2xconv7
    add_l4_l7 = tf.add(conv1x1_l4, output)
    # deconvolution: (conv4+2xconv7)*2
    output = tf.layers.conv2d_transpose(add_l4_l7, num_classes, 4, 2, padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(0,rand_std),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
    
    tf.Print(output, [tf.shape(output)])
    # pool3 + 2x conv4 + 4x conv7
    add_l3_l4_l7 = tf.add(conv1x1_l3, output)
    output = tf.layers.conv2d_transpose(add_l3_l4_l7, num_classes, 16, 8, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale))
    
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels = labels,logits=logits))
#    total_loss = cross_entropy_loss+ sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    nsample = []
    nloss = []
    for epoch in tqdm(range(epochs)):
        i=0
        for image, label in get_batches_fn(batch_size):
            # training
            _train_op, loss = sess.run([train_op, cross_entropy_loss],
                                       feed_dict= {input_image: image, 
                                                   correct_label: label,
                                                   keep_prob: p_keep,
                                                   learning_rate: r_learning})
            i=i+1
            print("(%10d): %.20f"%(i, loss))
            nsample.append(i*batch_size)
            nloss.append(loss)  
        print("Epoch: %d Loss: %f"%(epoch,loss))
    #plt.plot(nsample,nloss, 'ro')
    #plt.savefig('runs/Bsize%d_Pkeep%f.jpg'%(batch_size, p_keep))
    with open ('runs/Bsize%d_Pkeep%f.txt'%(batch_size, p_keep),'w') as f:  
        for s,l in zip(nsample,nloss):
            f.write("%d   %f\n"%(s,l))
    
    
tests.test_train_nn(train_nn)

 
def run():
    global p_keep,r_learning,BATCH_SIZE
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    Epochs = 30
    #BATCH_SIZE = 16
    if len(sys.argv)>1:
        BATCH_SIZE = int(sys.argv[1])
    if len(sys.argv)>2:
            p_keep = float(sys.argv[2])    
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    tf.reset_default_graph()

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out,layer7_out = load_vgg(sess, vgg_path)
        tf.Print(keep_prob,[keep_prob])
        #p_keep = keep_prob.get_Variable()
        
        
        #print("keep_prob from VGG: %f  keep_prob in project: %f"%(keep_prob,p_keep))
        final_out = layers(layer3_out, layer4_out,layer7_out, num_classes)
        correct_label = tf.placeholder( tf.float32,  (None, None, None, num_classes))
        learning_rate = tf.placeholder( tf.float32)
        logits, train_op, cross_entropy_loss = optimize(final_out, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, Epochs, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, 
                 input_image, correct_label, keep_prob, learning_rate)
        saver = tf.train.Saver()
        saver.save(sess, "./runs/Batch%d_Pkeep%f.ckpt"%( BATCH_SIZE, p_keep))

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        # OPTIONAL: Apply the trained model to a video
        video_name = 'dataroad_marked.mp4'
        helper.gen_test_output_video(  data_dir, sess, image_shape, logits, keep_prob, input_image,video_name)
        print('Writing video Finished.')


if __name__ == '__main__':
    run()
