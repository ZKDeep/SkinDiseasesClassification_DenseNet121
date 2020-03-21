# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:54:27 2020

@author: zubair
"""
import input_config
def model():
    n_classes =    input_config.n_classes
    import keras
    from keras import Model
    from keras.layers import GlobalAveragePooling2D, Dense
    #from sklearn.metrics import classification_report, confusion_matrix    
    dense121 = keras.applications.DenseNet121(include_top=False, weights='imagenet')    
    new_model=dense121.output
    new_model=GlobalAveragePooling2D(name='globalavg')(new_model)    
    #new_model=Dense(512,activation='relu')(new_model) #dense layer 3
    preds=Dense(n_classes,activation='softmax', name='output_layer_last')(new_model) #final layer with softmax activation
    
    model=Model(inputs=dense121.input,outputs=preds)
        
    for layer in model.layers:
        layer.trainable=True
    model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
    
    print("..")
    print("....")
    print(".....")
    print("......")
    print("........")
    
    return(model)



#
#
#img.save("resized_img.jpg")
#
#loss = []
#acc = []
#for epoch in range(2):
#    lo = []
#    ac = []
#    for img in range(len(images_path)):
#        im = PIL.Image.open(images_path[img])
#    
#        width, height = im.size   # Get dimensions
#        print(images_path[img])
#            
#        if width > 512 and height > 512:
#            print(width)
#            print(height)
#        
#    
#    
#            left = (width - 512)/2
#            top = (height - 512)/2
#            right = (width + 512)/2
#            bottom = (height + 512)/2
#            
#            # Crop the center of the image
#            im = im.crop((left, top, right, bottom))
#        im = np.array(im)
#        im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
#        print(im.shape)
#        im = im/255
#        la = classes[img]
#        lab = np.zeros((7,))
#        lab[la] = 1
#        lab = lab.reshape(1,7)
#        try:
#        #step_size_train = train_batches.n//train_batches.batch_size
#            #hist = model.fit(im, lab, epochs=1)
#            hist = model.evaluate(im, lab, batch_size = 1)
#            ac.append(hist[1])
#            lo.append(hist[0])
##            ac.append(hist.history['acc'])
##            lo.append(hist.history['loss'])
#            
#        except:
#            print("Skiping")
#    ac = np.average(np.array(ac))
#    lo = np.average(np.array(lo))
#    loss.append(lo)
#    acc.append(ac)
#            
                


#batc = batch_class(batch_size, complete, normalize)
#for train_batches in batc.batch(batch_size, complete, normalize):
#        for im in range(len(train_batches[0])):
#            imj = train_batches[0][im].reshape((1,train_batches[0][im].shape[0], 
#                              train_batches[0][im].shape[1], train_batches[0][im].shape[2]))
#            la = train_batches[1][im]
#            lab = np.zeros((7,))
#            lab[la] = 1
#            lab = lab.reshape(1,7)
#            try:
#            #step_size_train = train_batches.n//train_batches.batch_size
#                hist = model.fit(imj, lab,
#                               epochs=5)
#            except:
#                print("Skiping")
#                
