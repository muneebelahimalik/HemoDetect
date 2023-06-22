import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_model(num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(416, 416, 3))
    base_output = base_model.output

    pooled_regions = tf.keras.layers.GlobalAveragePooling2D()(base_output)
    roi_classification = Dense(256, activation='relu')(pooled_regions)
    output_scores = Dense(num_classes, activation='softmax')(roi_classification)

    model = Model(inputs=base_model.input, outputs=output_scores)
    return model

def compile_and_train_model(model, train_dataset, validation_dataset, loss_fn, optimizer, num_epochs):
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=validation_dataset)
    return history

def evaluate_model(model, validation_dataset):
    evaluation = model.evaluate(validation_dataset)
    return evaluation[0], evaluation[1]
