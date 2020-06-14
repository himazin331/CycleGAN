import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

def create_dataset(data_dir):

    print("\n___Creating a dataset...")
    
    cnt = 0
    cnt_t = 0
    prc = ['/', '-', '\\', '|']
    
    flg = 0

    for c in os.listdir(data_dir):
        d = os.path.join(data_dir, c)
        print("Number of image in a domain \"{}\": {}".format(c, len(os.listdir(d))))
        
    train_X = [] # ドメインX
    train_Y = [] # ドメインY

    
    for c in os.listdir(data_dir):
     
        print("\nclass: {}".format(c))   
        
        d = os.path.join(data_dir, c)            
        imgs = os.listdir(d)
        
        for i in [f for f in imgs if ('jpg'or'JPG' in f)]:     

            if i == 'Thumbs.db':
                continue

            img = tf.io.read_file(os.path.join(d, i))      
            img = tf.image.decode_image(img, channels=3)  
            img = (img.numpy() - 127.5) / 127.5

            if flg == 0:
                train_X.append(img)
            else:
                train_Y.append(img)
            
            cnt += 1
            cnt_t += 1
            
            print("\r   Loading a images...{}    ({} / {})".format(prc[cnt_t%4], cnt, len(os.listdir(d))), end='')
            
        print("\r   Loading a images...Done    ({} / {})".format(cnt, len(os.listdir(d))), end='')
        
        cnt = 0
        flg = 1
        
    print("\n___Successfully completed\n")
        
    train_X = tf.convert_to_tensor(train_X, np.float32) 
    train_X = train_X.numpy()
    train_Y = tf.convert_to_tensor(train_Y, np.float32) 
    train_Y = train_Y.numpy()
    return train_X, train_Y