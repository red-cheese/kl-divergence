from numpy import concatenate, linspace
from numpy.random import normal
from scipy.stats import gaussian_kde, entropy, ks_2samp
import matplotlib.pyplot as plt


MAX_ENTROPY_ALLOWED = 1e6  # A hack to never deal with inf entropy values that happen when the PDFs don't intersect


def run_test(p_samples, q_samples, p_label="P", q_label="Q"):
    n1 = len(p_samples)
    n2 = len(q_samples)
    if n1 == 0 or n2 == 0:
        return 0

    # Plot the samples
    plt.hist(p_samples, normed=True, bins=25, color='b', alpha=0.5, label=p_label)
    plt.hist(q_samples, normed=True, bins=25, color='g', alpha=0.5, label=q_label)
    plt.legend(loc="upper right")
    plt.show()

    # Estimate the PDFs using Gaussian KDE
    pdf1 = gaussian_kde(p_samples)
    pdf2 = gaussian_kde(q_samples)

    # Calculate the interval to be analyzed further
    a = min(min(p_samples), min(q_samples))
    b = max(max(p_samples), max(q_samples))

    # Plot the PDFs
    lin = linspace(a, b, max(n1, n2))
    p = pdf1.pdf(lin)
    q = pdf2.pdf(lin)
    plt.plot(lin, p, color='b', label="Estimated PDF(P)")
    plt.plot(lin, q, color='g', label="Estimated PDF(Q)")
    plt.legend(loc="upper right")
    plt.show()

    # Return the Kullback-Leibler divergence
    return min(MAX_ENTROPY_ALLOWED, entropy(p, q))


def run_normal_unimodal(n1=1000, n2=1000, mu1=0., sigma1=1., mu2=0., sigma2=1.):
    s1 = normal(mu1, sigma1, n1)
    s2 = normal(mu2, sigma2, n2)
    label1 = "P ~ N(%.2f, %.2f), %d samples" % (mu1, sigma1, n1)
    label2 = "Q ~ N(%.2f, %.2f), %d samples" % (mu2, sigma2, n2)
    kl_div = run_test(s1, s2, label1, label2)
    ks_test = ks_2samp(s1, s2)
    print("%s\n%s\nKL-divergence: %.10f\nKS 2-sample test: %.10f\n\n" % (label1, label2, kl_div, ks_test[0]))


def run_normal_bimodal(n1=1000, n2=1000, mu11=0., mu12=10., sigma11=1., sigma12=.5, mu2=0., sigma2=1.):
    s11 = normal(mu11, sigma11, n1)
    s12 = normal(mu12, sigma12, n1)
    s1 = concatenate((s11, s12))
    s2 = normal(mu2, sigma2, n2)
    label1 = "P ~ N(%.2f, %.2f) + N(%.2f, %.2f), %d samples" % (mu11, sigma11, mu12, sigma12, n1)
    label2 = "Q ~ N(%.2f, %.2f), %d samples" % (mu2, sigma2, n2)
    kl_div = run_test(s1, s2, label1, label2)
    ks_test = ks_2samp(s1, s2)
    print("%s\n%s\nKL-divergence: %.10f\nKS 2-sample test: %.10f\n\n" % (label1, label2, kl_div, ks_test[0]))


def main():
    run_normal_unimodal(5000, 500)
    run_normal_unimodal(mu1=2., sigma1=.5)
    run_normal_unimodal(mu1=20., sigma1=.5)
    run_normal_bimodal()


if __name__ == "__main__":
    main()
