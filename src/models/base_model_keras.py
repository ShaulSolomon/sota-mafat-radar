from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

class base_model_keras():
    '''
    The original keras Architecture that was given to us.
    ### **CNN Model**
    The model has two convolutional layers, both followed by max-pooling layers.    
    Those layers are followed by 2 fully-connected (dense) layers, activated with a ReLU function and regularized with   
    L2 regularization, followed by a final output layer with a single neuron with a Sigmoid activation function,   
    used for final binary classification.

    '''

    def create_model(input_shape, init):
        """
        CNN model.

        Arguments:
            input_shape -- the shape of our input
            init -- the weight initialization

        Returns:
            CNN model    
        """
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer = init, bias_regularizer='l2', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer = init, bias_regularizer='l2'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, kernel_regularizer = 'l2', activation='relu', kernel_initializer = init))
        model.add(Dense(32, kernel_regularizer = 'l2', activation='relu', kernel_initializer = init))
        model.add(Dense(1, activation='sigmoid', kernel_initializer = init))
        return model

