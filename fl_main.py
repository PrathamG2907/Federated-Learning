from utilities import *




folder_path = 'trainingSet'
image_paths = list(paths.list_images(folder_path))
image_list, label_list = load_data(image_paths)

#converting the label to 1x10 array
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.1, random_state=42)

batched_users = make_batched_users(x_train, y_train)
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

lr = 0.01 
global_epochs = 100
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(learning_rate=lr,decay=lr /global_epochs,  momentum=0.9) 

global_model = build_model(784, 10)

for epoch in range(global_epochs):
            
    # get weights of the global model
    global_weights = global_model.get_weights()
    
    #list to collect local model weights after scalling
    scaled_local_weight_list = []

    usernames= list(batched_users.keys())
    random.shuffle(usernames)

    for user in usernames:
        local_model = build_model(784, 10)
        local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        
        #fit local model with user's data
        local_model.fit(batched_users[user], epochs=1, verbose=0)
        
        #scale the model weights and add to list
        scaled_weights = scale_model_weights(local_model.get_weights(),batched_users,user)
        scaled_local_weight_list.append(scaled_weights)
        
        #clear session to free memory after each communication round
        K.clear_session()
        
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    
    #update global model 
    global_model.set_weights(average_weights)

    #test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, epoch)
