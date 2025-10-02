
dataset = InsectsDataset(
    csv_file="../data/Insects.csv",
    root_dir="../data/Insects",
    transform=transform
)

trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=0)

dataiter = iter(trainloader)
images, labels = next(dataiter)

for image, label in zip(images, labels):
    plt.figure()
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(dataset.idx_to_species[label.item()])
    plt.axis("off")
    plt.show()

