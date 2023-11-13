import sys
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf

print(tfds.__version__)

data_augment_flag = sys.argv[1].lower() == "true"
print(data_augment_flag)

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
def draw_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
 
def augment_image(image, label):
    """
    이미지 데이터에 대한 증강을 수행하는 함수입니다 (TensorFlow 사용).
    
    :param image: TensorFlow 이미지 텐서.
    :param label: 이미지에 대한 라벨.
    :return: 증강된 이미지와 원래 라벨.
    """
    # 이미지를 좌우 플립
    image = tf.image.random_flip_left_right(image)

    # 이미지를 상하 플립
    image = tf.image.random_flip_up_down(image)

    # 밝기 조절
    image = tf.image.random_brightness(image, max_delta=0.2)

    # 대비 조절
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    return image, label    

# tf_flowers data load
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    name='tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    data_dir='~/aiffel/tf_flowers/',
    #download=False,
    download=True,
    with_info=True,
    as_supervised=True,
)

# print("### raw data  ")    
# print(raw_train)
# print(raw_validation)
# print(raw_test)

# augement if flag on
if (data_augment_flag):
    raw_train = raw_train.map(augment_image)

# show 10 data    
plt.figure(figsize=(10, 5))

get_label_name = metadata.features['label'].int2str

for idx, (image, label) in enumerate(raw_train.take(10)):  # 10개의 데이터를 가져 옵니다.
    plt.subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title(f'label {label}: {get_label_name(label)}')
    plt.axis('off')

# image resize
import tensorflow as tf

IMG_SIZE = 160 # 리사이징할 이미지의 크기

def format_example(image, label):
    image = tf.cast(image, tf.float32)  # image=float(image)같은 타입캐스팅의  텐서플로우 버전입니다.
    image = (image/127.5) - 1 # 픽셀값의 scale 수정
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

print("Image resized")

# format example 실행
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# print(train)
# print(validation)
# print(test)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

print("Batches perpared")

#image_batch, label_batch = train_batches.take(1).next()

for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    #predictions = model.predict(image_batch)
    break