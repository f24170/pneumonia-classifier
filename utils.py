import matplotlib.pyplot as plt

def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img, label = dataset[i]
        img = img.squeeze(0)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    plt.show()


def print_prediction_with_confidence(output, class_names):
    import torch
    probs = torch.softmax(output, dim=1)[0]
    conf, pred = torch.max(probs, 0)
    print(f"預測結果：{class_names[pred.item()]}（信心值：{conf.item():.2%}）")

def plot_accuracy(history):
    import matplotlib.pyplot as plt
    plt.plot(history['acc'], label='Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.show()
