import matplotlib.pyplot as plt

def plot_gan_loss(gen_loss, disc_loss, num_epochs):
    plt.plot(range(len(gen_loss)), gen_loss, label='generator loss')
    plt.plot(range(len(disc_loss)), disc_loss, label='discriminator loss')
    plt.legend()
    plt.show()