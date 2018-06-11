import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
from skimage import transform, exposure
import cv2
from moviepy.editor import *

from scipy.ndimage import uniform_filter

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
				   
					  
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image0 = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image0 = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
																													   
                gt_bg = np.all(gt_image0 == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
										 							
                images.append(image0)
                gt_images.append(gt_image)
                  # flip image
                images.append(image0[:,::-1,:])
                gt_images.append(gt_image[:,::-1,:])     
                # change brightness
                randGamma = random.uniform(0.5,1.5)
                image = exposure.adjust_gamma(image0, randGamma, 1)                    
                images.append(image)
                gt_images.append(gt_image)
                # geometric aug
#                randShear = random.uniform(-10,10)*np.pi/180
#                randTranslateX = random.uniform(-100,100)   # 8% of x length
#                randTranslateY = random.uniform(-30,30)   # 8% of y length
#                randrot = random.uniform(-5,5)*np.pi/180
#                image = transform.warp(image0, \
#                             transform.AffineTransform(scale=None,\
#                                              rotation=randrot, \
#                                              shear=randShear,\
#                                              translation=(randTranslateX,randTranslateY)),mode='edge') 
#                gt_image = transform.warp(gt_image0, \
#                             transform.AffineTransform(scale=None,\
#                                              rotation=randrot, \
#                                              shear=randShear,\
#                                              translation=(randTranslateX,randTranslateY)),mode='edge')
#                gt_bg = np.all(gt_image == background_color, axis=2)
#                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
#                images.append(image)
#                gt_images.append(gt_image) 
                
            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def denoise_img(img):
    for i in range(img.shape[-1]):
        img[...,i] = uniform_filter(img[...,i],3)
    img = img.astype('uint8')
    return img


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        segmentation_r = denoise_img(segmentation)
        mask = np.dot(segmentation_r, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        
        
def gen_test_output_video(data_dir, sess, image_shape, logits, keep_prob, input_image, video_filename):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param video_filename: file name of the video
    :param image_shape: Tuple - Shape of image
    :return: n/a
    """
    imglist=[]
    for image_file in glob(os.path.join(data_dir, 'data_road/testing', 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        segmentation_r = denoise_img(segmentation)
        mask = np.dot(segmentation_r, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        imglist.append(np.array(street_im))
   
    # When everything done, release the capture
    clips = [ImageClip(m).set_duration(0.1)
         for m in imglist]                

    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(video_filename, fps=24)

def gen_test_output_video2(sess, logits, keep_prob, image_pl, video_filename, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param video_filename: file name of the video
    :param image_shape: Tuple - Shape of image
    :return: n/a
    """
    
    vc = cv2.VideoCapture(video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, image_shape)
    

    while True:
        ret, frame = vc.read()
        if frame is None:
            break
        image = scipy.misc.imresize(frame, image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        segmentation_r = denoise_img(segmentation)
        mask = np.dot(segmentation_r, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.imresize(mask, frame.shape)
        mask = scipy.misc.toimage(mask, mode="RGBA")
        
        street_im = scipy.misc.toimage(frame)
        street_im.paste(mask, box=None, mask=mask)
        
        out.write(street_im)


    # When everything done, release the capture
    vc.release()
    out.release()
    cv2.destroyAllWindows()	   
    
def gen_test_output_video1(sess, logits, keep_prob, image_pl, video_filename, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param video_filename: file name of the video
    :param image_shape: Tuple - Shape of image
    :return: n/a
    """
    vc = cv2.VideoCapture(video_filename)
   
    
    counter=0
    while True:
        ret, frame = vc.read()
        if frame is None:
            break
        image = scipy.misc.imresize(frame, image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        segmentation_r = denoise_img(segmentation)
        mask = np.dot(segmentation_r, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.imresize(mask, frame.shape)
        mask = scipy.misc.toimage(mask, mode="RGBA")
        
        street_im = scipy.misc.toimage(frame)
        street_im.paste(mask, box=None, mask=mask)

        cv2.imwrite("runs/image%d.jpg"%counter,np.array(street_im))
        counter=counter+1

    # When everything done, release the capture
    vc.release()
    cv2.destroyAllWindows()	       