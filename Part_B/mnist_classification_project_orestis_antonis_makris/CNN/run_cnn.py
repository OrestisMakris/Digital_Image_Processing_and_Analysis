import torch
import matplotlib.pyplot as plt
import numpy as np


from data_loader import load_mnist_data,  visualize_mnist_samples , visualize_batch_samples
from cnn_model import MNISTConvNet

from train_cnn import train_model,  plot_training_history,  save_model
from eval_cnn import get_predictions, plot_confusion_matrix,  print_classification_report , plot_per_class_metrics , plot_conv_filters

def main():
    """
    main function to run the complete CNn pipeline.
    """
    # --- Configuration --

    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3 

    MODEL_SAVE_PATH = 'mnist_cnn_model_state.pth'
    
    print(f"Using device: {DEVICE}")


    print("\n--- Step 1: Loading and Visualizing Data ---")

    train_loader, test_loader, classes = load_mnist_data(batch_size=BATCH_SIZE)
    
    
    visualize_mnist_samples(train_loader, classes)
    
    #
    visualize_batch_samples(train_loader, num_samples=16)

    
    print("\n--- Step 2: Training the CNN Model ---")

    model = MNISTConvNet().to(DEVICE)
    
    history = train_model(model, train_loader, test_loader, 
                          num_epochs=NUM_EPOCHS, 
                          learning_rate=LEARNING_RATE, 
                          device=DEVICE)
                          
    plot_training_history(history)
    save_model(model, MODEL_SAVE_PATH)

    
    print("\n--- Step 3: Evaluating the Trained Model ---")

    eval_model = MNISTConvNet()

    eval_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    eval_model.to(DEVICE)

    predictions, true_labels = get_predictions(eval_model, test_loader, DEVICE)
    
    print_classification_report(true_labels, predictions, classes)
    
    
    plot_confusion_matrix(true_labels, predictions, classes, normalize=False, title='Confusion Matrix')
    plot_confusion_matrix(true_labels, predictions, classes, normalize=True, title='Normalized Confusion Matrix')
    
    
    plot_per_class_metrics(true_labels, predictions, classes)


    plot_conv_filters(eval_model, num_filters=6)

    sample, _ = next(iter(test_loader))
    img = sample[0].unsqueeze(0).to(DEVICE)  
    fmap = eval_model.get_feature_maps(img)



if __name__ == "__main__":

    main()