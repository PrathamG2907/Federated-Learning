from fl_P2P_utilities import *
import random


folder_path = 'trainingSet'
image_paths = list(paths.list_images(folder_path))
image_list, label_list = load_data(image_paths)

#converting the label to 1x10 array
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.1, random_state=42)

batched_users = make_batched_users(x_train, y_train)
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

users= list(batched_users.keys())
user_models=make_user_models(users)
model_versions=make_user_versions(users)


lr = 0.01 
total_iterations = 1000
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(learning_rate=lr,decay=lr /total_iterations,  momentum=0.9) 



for iter in range(total_iterations):
    usernames= list(batched_users.keys())
    random.shuffle(usernames)

    for user in usernames:
        user_models[user].fit(batched_users[user], epochs=5, verbose=0)
    
    random_user=random.choice(usernames)
    scaled_weights_list=[]
    updated_users=[]
    for user in usernames:
        if(model_versions[user][user]>model_versions[random_user][user]):
            model_versions[random_user][user]=model_versions[user][user]
            updated_users.append(user)
    
    updated_users.append(random_user)


    for user in updated_users:
        scaled_weights=scale_model_weights(user_models[user].get_weights(), batched_users, user ,updated_users)
        scaled_weights_list.append(scaled_weights)
    new_weights = sum_scaled_weights(scaled_weights_list)
    user_models[random_user].set_weights(new_weights)

    for(X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, user_models[random_user], iter)

