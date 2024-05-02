ITER_N = 120
CHANNEL_N = 16 # Number of CA state channels
CELL_FIRE_RATE = 0.5
SEED_STD = 0.1
batch_size = 32
target_digit = 6

import numpy as np
import matplotlib.pylab as pl

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import Constraint
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
from datetime import datetime
# Get the current date in YYYYMMDD format
current_date = datetime.now().strftime("%Y_%m_%d_outwards")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load MNIST data
(original_train_images, original_train_labels), (original_test_images, original_test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape the entire dataset before splitting
normalized_images = original_train_images.reshape(original_train_images.shape[0], -1).astype('float32') / 255
normalized_test_images = original_test_images.reshape(original_test_images.shape[0], -1).astype('float32') / 255

# Define a split ratio for the dataset
split_ratio = 0.8  # e.g., 80% for training, 20% for validation / testing

# Calculate the number of images to include in the training set
num_train_images = int(len(normalized_images) * split_ratio)

# Split the images and labels into training and test sets
train_images = normalized_images[:num_train_images]
train_labels = original_train_labels[:num_train_images]
val_images = normalized_images[num_train_images:]
val_labels = original_train_labels[num_train_images:]

test_images = normalized_test_images
test_labels = original_test_labels

# make it train on all digits
# train_labels = np.full(len(train_images), target_digit)
# val_labels = np.full(len(val_labels), target_digit)
# test_labels = np.full(len(test_labels), target_digit)
                     
print(f"Train / val / test split {train_images.shape, val_images.shape, test_images.shape}")

# Create a TensorFlow dataset for each digit in the training set
datasets = []
for digit in range(10):
    idx = train_labels == digit
    digit_images = train_images[idx]
    datasets.append(tf.data.Dataset.from_tensor_slices(digit_images).shuffle(1000).batch(60))
    
#@title CA model and utils
def to_greyscale(x):
    y = tf.clip_by_value(x[..., 0:1], 0.0, 1.0)
    return y

def get_living_mask(x):
    alpha = x[:, :, :, 0:1]
    return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1

def make_seed(num_examples, channel_n=CHANNEL_N, seed_std=SEED_STD):
    h, w = 28,28
    seed = np.zeros([num_examples, h, w, channel_n], np.float32)
    for i in range(h//2 - 1, h//2 + 1):
        for j in range(w//2-1, w//2 + 1):
            seed[:, i, j, 0] = np.random.uniform(1, 1, size = num_examples)
            seed[:, i, j, 1:] = np.random.normal(0, seed_std, size = seed[:, i, j, 1:].shape)
#     seed = np.random.normal(0.5, 0.5, size = seed.shape)
    return seed

# Gaussian initialization
class CustomInitializer(Initializer):
    def __init__(self, mean=0.0, stddev=0.01):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=np.float32):
        return tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

class build_generator(tf.keras.Model):

    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE, f1 = 100, f2 = 100):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.f1 = f1
        self.f2 = f2 

        self.perceive = tf.keras.Sequential([
            Conv2D(self.f1, 3, activation=tf.nn.relu, padding="SAME"), # 80 filters, 3x3 kernel
        ])

        self.dmodel = tf.keras.Sequential([
            Conv2D(self.f2, 1, activation=tf.nn.relu),
            Conv2D(self.channel_n, 1, activation=tf.nn.tanh,
                kernel_initializer=tf.zeros_initializer),
        ])

        self(tf.zeros([1, 3, 3, channel_n]))  # dummy call to build the model

    @tf.function
    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x)
        dx = self.dmodel(y)*step_size
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
        x += dx * tf.cast(update_mask, tf.float32)

        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask

        return x * tf.cast(life_mask, tf.float32)

build_generator().dmodel.summary()

# TODO: TRY SIGMOID, square root loss 


def compute_kernel(x, y, sigma_list):
    """Computes a Gaussian kernel between two sets of samples using multiple bandwidth parameters."""
    beta_list = [1.0 / (2.0 * sigma**2) for sigma in sigma_list]
    x_expanded = tf.expand_dims(x, 1)
    y_expanded = tf.expand_dims(y, 0)
    kernel_val = 0.
    for beta in beta_list:
        beta = tf.cast(beta, tf.float32)
        squared_diff = tf.reduce_sum(tf.square(x_expanded - y_expanded), 2)
        kernel_val += tf.exp(-beta * squared_diff)
    return kernel_val / tf.cast(tf.size(sigma_list), tf.float32)

def compute_mmd(x, y, sigma_list=[2, 5, 10, 20, 40, 80]):
    """Computes the Maximum Mean Discrepancy (MMD) between two sets of samples, x and y."""
    x_kernel = compute_kernel(x, x, sigma_list)
    y_kernel = compute_kernel(y, y, sigma_list)
    xy_kernel = compute_kernel(x, y, sigma_list)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


def log_mean_exp(a):
    """Compute the log of the mean of exponentials of input elements."""
    max_ = tf.reduce_max(a, axis=1, keepdims=True)
    return max_ + tf.math.log(tf.reduce_mean(tf.exp(a - max_), axis=1))

def tensorflow_parzen_estimator(mu, sigma):
    """Constructs a Parzen window estimator using TensorFlow."""
    mu = tf.convert_to_tensor(mu, dtype=tf.float32)
    sigma = tf.constant(sigma, dtype=tf.float32)
    
    def parzen_estimator(x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        a = (tf.expand_dims(x, 1) - tf.expand_dims(mu, 0)) / sigma
        E = log_mean_exp(-0.5 * tf.reduce_sum(tf.square(a), axis=2))
        Z = mu.shape[1] * tf.math.log(sigma * tf.sqrt(2 * np.pi))
        return E - Z
    
    return parzen_estimator

def compute_log_likelihood(parzen, data, batch_size=100):
    """Computes log-likelihood of data given a Parzen window estimator."""
    n_batches = int(np.ceil(data.shape[0] / batch_size))
    log_likelihoods = []
    for i in range(n_batches):
        batch = data[i*batch_size:(i+1)*batch_size]
        log_likelihood = parzen(batch)
        log_likelihoods.append(log_likelihood)
    return tf.reduce_mean(log_likelihoods)

def find_best_sigma(samples, data, sigma_range, batch_size=100, verbose=True):
    """Finds the best sigma value over a range by optimizing log-likelihood."""
    best_log_likelihood = float('-inf')
    best_sigma = 0
    
    for sigma in sigma_range:
        parzen = tensorflow_parzen_estimator(samples, sigma)
        log_likelihood = compute_log_likelihood(parzen, data, batch_size)
        
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_sigma = sigma
            
        if verbose:
            print(f'sigma={sigma}, log_likelihood={log_likelihood.numpy():.2f}')
    
    if verbose:
        print('====================')
        print(f'Best log_likelihood={best_log_likelihood.numpy():.2f} for sigma={best_sigma}')
        print('')
    
    return best_log_likelihood, best_sigma

def generate_images(model, channel_n=CHANNEL_N, seed_std=SEED_STD, iter_n=ITER_N):
    x = make_seed(16, channel_n=channel_n, seed_std=seed_std)
    for i in range(iter_n):
        x = model(x, training=False) 
    generated_images = x
    fig, axes = plt.subplots(1, 16, figsize=(20, 2))
    greyscale_images = to_greyscale(generated_images)
    for i, ax in enumerate(axes):
        ax.imshow(greyscale_images[i, :, :, 0], cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Generated Images for Digit {target_digit}')
    return fig, axes


def plot_training_metrics(epochs, mmd_losses, parzen_log_likelihoods, parzen_eval_interval, target_digit, params, train_version=False):
    """
    Plots and saves the training metrics including MMD Loss and Parzen Window Log Likelihood
    for a specific target digit.
    
    Parameters:
    - epochs: Total number of epochs trained.
    - mmd_losses: List of MMD losses recorded after each epoch.
    - parzen_log_likelihoods: List of Parzen window log likelihoods recorded at specified intervals.
    - parzen_eval_interval: Interval at which Parzen window log likelihoods were evaluated.
    - target_digit: The target digit for which the model was trained.
    """
    
    # Create directory if it doesn't exist
    save_dir = f'evaluation/{current_date}/{target_digit}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "train_metrics"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "generated_images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "log_likelihoods"), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Plotting MMD Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), mmd_losses, label='MMD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MMD Loss')
    plt.title('MMD Loss over Epochs')
    plt.legend()
    
    # Plotting Parzen Window Log Likelihood
    plt.subplot(1, 2, 2)
    epochs_evaluated = list(range(parzen_eval_interval, epochs + 1, parzen_eval_interval))
    plt.plot(epochs_evaluated, parzen_log_likelihoods, label='Parzen Log Likelihood', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Log Likelihood')
    plt.title('Parzen Window Log Likelihood over Epochs')
    plt.legend()
    
    plt.tight_layout()
    
    if not train_version: 
        # Save the figure
        filename = os.path.join(save_dir, f'train_metrics/train_metrics_{params["fire_rate"]}_{params["channel_n"]}_{params["lr"]}_{params["epochs"]}_{params["iter_n"]}_{params["seed_std"]}_VALIDATION_VERSION.png')
    else: 
        filename = os.path.join(save_dir, f'train_metrics/train_metrics_{params["fire_rate"]}_{params["channel_n"]}_{params["lr"]}_{params["epochs"]}_{params["iter_n"]}_{params["seed_std"]}_TRAIN_VERSION.png')
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory
    
    if not train_version:
        # Save the generated images
        (fig, axes) = generate_images(params["model"], params["model"].channel_n, params["seed_std"], params["iter_n"])
        fig.savefig(filename.replace('train_metrics/train_metrics', 'generated_images/generated_images'))
        plt.close()

        # Save the model
        model_filename = filename.replace("train_metrics/train_metrics", "model/model")
        params["model"].save_weights(model_filename)
    


def train_model_for_digit(target_digit, params, sigma_range=np.arange(0.1, 0.5, 0.05), parzen_eval_interval=1):
    model = params["model"]
    optimizer = params["optimizer"]
    epochs = params["epochs"]
    seed_std = params["seed_std"]
    channel_n = model.channel_n
    iter_n = params["iter_n"]
    
    fig, axes = generate_images(model, channel_n=channel_n, seed_std=seed_std, iter_n=iter_n)
    plt.show()
    
    log_likelihood = -10000.0

    digit_dataset = datasets[target_digit]
    
    idx = val_labels == target_digit
    digits = val_images[idx]
    val_data = digits[:len(digits) - len(digits) % 100]
    
    idx = train_labels == target_digit
    digits = train_images[idx]
    train_test_data = digits[:len(val_data)]

    mmd_losses = []
    parzen_log_likelihoods = []
    train_log_likelihoods = []

    for epoch in range(epochs):
        epoch_mmd_losses = []
        for real_images in digit_dataset:
            x = make_seed(batch_size, channel_n=channel_n, seed_std=seed_std)
            
            with tf.GradientTape() as tape:
                for i in tf.range(iter_n):
                    x = model(x, training=True)
                generated_images = tf.reshape(x[..., 0], [batch_size, 28*28])
                mmd_loss = compute_mmd(real_images, generated_images)

            grads = tape.gradient(mmd_loss, model.weights)
            grads = [g/(tf.norm(g)+1e-8) for g in grads]
            optimizer.apply_gradients(zip(grads, model.weights))
            epoch_mmd_losses.append(mmd_loss.numpy())
        
        epoch_mmd_loss_avg = np.mean(epoch_mmd_losses)
        mmd_losses.append(epoch_mmd_loss_avg)
        
        # Evaluate using Parzen window estimator periodically
        if (epoch+1) % parzen_eval_interval == 0:
            fig, axes = generate_images(model, channel_n=channel_n, seed_std=seed_std, iter_n=iter_n)
            plt.close()
            x = make_seed(val_data.shape[0], channel_n=channel_n, seed_std=seed_std)
            for i in range(iter_n): 
                x = model(x, training=False)
            samples = tf.reshape(x[..., 0], [x.shape[0], 28*28])
            log_likelihood, best_sigma = find_best_sigma(samples, val_data, sigma_range, verbose=False)
            train_log_likelihood, train_best_sigma = find_best_sigma(samples, train_test_data, sigma_range, verbose=False)
            parzen_log_likelihoods.append(log_likelihood.numpy())
            train_log_likelihoods.append(train_log_likelihood.numpy())
#             print(f'Epoch {epoch+1}, Parzen Log Likelihood: {log_likelihood.numpy()}')
#             print(f'Epoch {epoch+1}, MMD Loss: {epoch_mmd_loss_avg}')
    
        
    plot_training_metrics(epochs, mmd_losses, parzen_log_likelihoods, parzen_eval_interval, target_digit, params)
    plot_training_metrics(epochs, mmd_losses, train_log_likelihoods, parzen_eval_interval, target_digit, params, train_version=True)
    
    
    assert log_likelihood != -10000
    return model, log_likelihood


# params = {
#     "fire_rate": [0.2, 0.5, 0.8],
#     "channel_n": [4, 8, 16],
#     "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
#     "epochs": [100],
#     "iter_n": [70, 100, 130, 160],
#     "seed_std": [0.01, 0.1, 1.0, 10.0],
# }

params = {
    "channel_n": [8, 16, 32, 64, 90],
    "f2": [32, 64, 128, 256, 512],
    "iter_n": [70, 100, 130, 160],
#     "seed_std": [0.01, 0.1, 1.0, 10.0],
#     "fire_rate": [0.2, 0.5, 0.8, 1],
#     "lr": [3e-5, 1e-4, 2e-4, 3e-4, 1e-3, 3e-3],
#     "epochs": [50, 100, 150, 200],
}


best_params = {
    "fire_rate": 0.5,
    "channel_n": 16,
    "model": build_generator(),
    "lr": 2e-4,
    "optimizer": tf.keras.optimizers.Adam(learning_rate=2e-4),
    "epochs": 100,
    "iter_n": 120,
    "seed_std": SEED_STD,
    "f1": 100, 
    "f2": 128, 
}

best_log_likelihood = float('-inf')

# # Define the path for the output text file
output_text_file = f'evaluation/{current_date}/{target_digit}/training_log.txt'
os.makedirs(os.path.dirname(output_text_file), exist_ok=True)  # Ensure the directory exists

# Open the text file for writing
with open(output_text_file, 'w') as log_file:
    
    idx = val_labels == target_digit
    digits = val_images[idx]
    test_data = digits[:len(digits) - len(digits) % 100]
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    for param_name, param_values in params.items():
        print(f'Testing {param_name}s', file=log_file)
        current_params = best_params.copy()
        likelihoods = []
        param_saved_vals = []

        for param in param_values: 
            print(f"Currently on {param_name} = {param}", file=log_file)
            current_params[param_name] = param

            current_params["model"] = build_generator(fire_rate=current_params["fire_rate"], channel_n=current_params["channel_n"], f2=current_params["f2"])
            current_params["optimizer"] = tf.keras.optimizers.Adam(learning_rate=current_params["lr"])

            current_model, likelihood = train_model_for_digit(target_digit, current_params)
            likelihoods.append(likelihood)
            param_saved_vals.append(param if not isinstance(param, tf.keras.optimizers.Optimizer) else param.learning_rate.numpy())  # Handling optimizer objects differently

            if likelihood > best_log_likelihood:
                best_log_likelihood = likelihood
                best_params[param_name] = param
                print(f"New best {param_name} = {param} with loss {likelihood}", file=log_file)
            else:
                print(f"Current {param_name} = {param} with loss {likelihood}", file=log_file)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(param_saved_vals, likelihoods, marker='o', linestyle='-')
        plt.title(f'Log Likelihood vs. {param_name}')
        plt.xlabel(param_name if param_name != 'optimizers' else 'Learning Rate')
        plt.ylabel('Log Likelihood')
        plt.grid(True)

        # Save the plot
        # Create directory if it doesn't exist
        save_dir = f'evaluation/{current_date}/{target_digit}'
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f'log_likelihoods/log_likelihood_vs_{param_name}.png')
        plt.savefig(filename)
        plt.show()
        plt.close()  # Close the figure to free memory
        
        # Generate a load of sample images using this model. 
        # These can be loaded later to generate statistics about the model's ability 
        
        x = make_seed(5000, channel_n=current_params["channel_n"], seed_std=current_params["seed_std"])

        for i in range(current_params["iter_n"]):
            x = current_params["model"](x, training=False) 

        generated_images = x
        greyscale_images = to_greyscale(generated_images)
        
        os.makedirs(os.path.join(save_dir, "image_samples"), exist_ok=True)
        np.save(os.path.join(save_dir, f'image_samples/generated_images_{current_params["fire_rate"]}_{current_params["channel_n"]}_{current_params["lr"]}_{current_params["epochs"]}_{current_params["iter_n"]}_{current_params["seed_std"]}.npy'), greyscale_images)  

    print("Best params: ", best_params, file=log_file) 
    print("Best log likelihood: ", best_log_likelihood, file=log_file)



idx = test_labels == target_digit
digits = test_images[idx]
test_data = digits[:len(digits) - len(digits) % 100]
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

save_dir = f'evaluation/{current_date}/{target_digit}'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "image_samples"), exist_ok=True)
np.save(os.path.join(save_dir, "image_samples", f"test_images.npy"), test_data)

